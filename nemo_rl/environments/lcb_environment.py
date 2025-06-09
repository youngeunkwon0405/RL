# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class LCBEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]]
    livecodebench_tests_path: Optional[str] 


class LCBEnvironmentMetadata(TypedDict):
    global_id: str
    question_id: str
    tests: list
    starter_code: Optional[str]


@ray.remote
class LCBVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, livecodebench_tests_path: Optional[str] = None):
        if livecodebench_tests_path:
            os.environ['LIVECODEBENCH_TESTS'] = livecodebench_tests_path
        
        from nemo_rl.environments import livecodebench_v5
        self.compute_scores = livecodebench_v5.compute_scores
    

    
    def verify(
        self,
        pred_responses: List[str],
        prompts: List[str],
        metadata: List[LCBEnvironmentMetadata],
    ) -> List[float]:
        """Verify the correctness of the generated code against test cases.

        Args:
            pred_responses: List[str]. The generated responses from the policy.
            prompts: List[str]. The prompts corresponding to the responses.
            metadata: List[LCBEnvironmentMetadata]. The metadata containing test cases.

        Returns:
            List[float]. The rewards for each predicted response (1.0 for pass, 0.0 for fail).
        """
        
        jobs = []
        for i, (response, metadata_item) in enumerate(zip(pred_responses, metadata)):
            job = {
                'tests': metadata_item['tests'],
                'gen': response,
                'attempt_idx': i
            }
            jobs.append(job)
        
        cache_file = f"lcb_cache_nemo_rl_{os.getpid()}.jsonl"
        try:
            accuracy = self.compute_scores(jobs, cache_file)
            
            # use cache file to get reward per response
            results = [0.0] * len(pred_responses)
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_jobs = [json.loads(line) for line in f]
                
                for cached_job in cached_jobs:
                    idx = cached_job['attempt_idx']
                    if idx < len(results):
                        is_pass = cached_job.get('pass-1', False)
                        results[idx] = 1.0 if is_pass else 0.0
                
                os.remove(cache_file)
            
            return results
        except Exception as e:
            logging.error(f"Error in LCB verification: {e}")
            return [0.0] * len(pred_responses)


@ray.remote
class LCBEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: LCBEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        livecodebench_tests_path = cfg.get("livecodebench_tests_path")
        
        self.workers = [
            LCBVerifyWorker.options(
                runtime_env={"py_executable": LCBVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote(livecodebench_tests_path)
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[LCBEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the LCB environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs.
            metadata: List[LCBEnvironmentMetadata]. Contains test cases and problem info.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        assistant_response_batch = []
        prompts = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append(assistant_responses[-1])
            prompts.append(conversation[0]["content"])

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_prompts = chunk_list_to_workers(prompts, self.num_workers)
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(response, prompt, metadata)
            for i, (response, prompt, metadata) in enumerate(
                zip(chunked_assistant_response_batch, chunked_prompts, chunked_metadata)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result > 0.5
                else "Environment: incorrect",
            }
            for result in results
        ]

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        
        Note: This function is not used in the current implementation.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )
        
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics 