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
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch
from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    MathEnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env


@ray.remote
class HFVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, extraction_config: Optional[List[str]] = None):
        if extraction_config:
            configs = []
            for config_name in extraction_config:
                if config_name == "latex":
                    configs.append(LatexExtractionConfig)
                elif config_name == "expr":
                    configs.append(ExprExtractionConfig)
                elif config_name == "string":
                    configs.append(StringExtractionConfig)
                else:
                    raise ValueError(f"Unknown extraction config `{config_name}`")
        else:
            configs = None
        self.extraction_config = configs

    def verify(
        self, pred_responses: List[str], ground_truths: List[str]
    ) -> List[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            ground_truths: List[str]. The ground truth responses.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        rewards = []
        has_predictions = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                if self.extraction_config is not None:
                    gold = parse(ground_truth, self.extraction_config)
                    pred = parse(
                        response[-100:], self.extraction_config
                    )  # avoid looking at the whole string
                else:
                    gold = parse(ground_truth)
                    pred = parse(response[-100:])  # avoid looking at the whole string
                rewards.append(float(verify(gold, pred)))
                has_predictions.append(len(pred) > 0)
            except Exception:
                rewards.append(0)
                has_predictions.append(False)
        return rewards, has_predictions


class MathEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote
class MathEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": HFVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote(cfg["extraction_config"])
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[MathEnvironmentMetadata],
    ) -> MathEnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # Unpack the results
        rewards = []
        has_predictions = []
        for rew, has_pred in results:
            rewards += rew
            has_predictions += has_pred

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if reward
                else "Environment: incorrect",
            }
            for reward in rewards
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(rewards).cpu()
        done = torch.ones_like(rewards).cpu()
        has_predictions = torch.tensor(has_predictions).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return MathEnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            has_predictions=has_predictions,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
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
            # "table": table, TODO @sahilj WIP
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
