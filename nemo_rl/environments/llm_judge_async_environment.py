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
import logging
import os
import uuid
import re # Import re module for regex
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import ray
import torch
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.environments.utils import extract_answer_from_box
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)

# Configuration and Metadata types (can be reused or adapted from existing llm_judge_environment)
class LLMJudgeAsyncConfig(TypedDict):
    num_workers: int
    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Default sampling parameters for the judge
    temperature: Optional[float]
    max_tokens: Optional[int]
    stop: Optional[List[str]] # Note: vLLM's AsyncEngine uses 'stop'
    # Any other vllm.SamplingParams can be added here

class LLMJudgeEnvironmentMetadata(TypedDict): # Reusing from previous
    reference_answer: Optional[str]
    evaluation_criteria: Optional[str]
    judge_prompt_template: Optional[str] # Added for per-sample judge prompts
    extract_box: Optional[bool]

@ray.remote
class AsyncVLLMWorker:
    """
    Worker that serves an LLM using vllm.AsyncLLMEngine for judging responses.
    Each call to judge() handles a single prompt asynchronously.
    """
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.VLLM
    DEFAULT_JUDGE_PROMPT_TEMPLATE = """You are an expert judge. You will be given a question, a model's prediction to evaluate, a reference answer, and evaluation criteria.

Question:
{question}

Model's Prediction:
{response}

Reference Answer:
{reference}

Now, evaluate if the response is correct based on the following evaluation criteria:
{criteria}

Answer yes or no, then give your reasoning.
"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        disable_log_stats: bool = True, # Reduce verbose VLLM logging
        **engine_kwargs, # Allow passing other EngineArgs
    ):
        # Imports moved here to be within the Ray actor's context,
        # ensuring they are resolved correctly in the worker environment.
        import os
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        
        self.AsyncLLMEngine = AsyncLLMEngine
        self.SamplingParams = SamplingParams
        self.default_judge_prompt_template = self.DEFAULT_JUDGE_PROMPT_TEMPLATE # Store the default

        # Attempt to use HF_HOME from env, otherwise default to huggingface_hub's default cache
        # This ensures the worker tries to use the same cache path as the driver.
        hf_home_cache_path = os.environ.get("HF_HOME", HUGGINGFACE_HUB_CACHE)
        if not os.path.isdir(hf_home_cache_path):
            try:
                os.makedirs(hf_home_cache_path, exist_ok=True)
                logging.info(f"Created HF cache directory for worker: {hf_home_cache_path}")
            except OSError as e:
                logging.warning(f"Worker could not create HF cache directory {hf_home_cache_path}: {e}. "
                                 "This might lead to download issues if the default cache is not writable.")

        # It's critical that download_dir is set for vLLM if HF_HOME is being customized,
        # or if there are any doubts about vLLM picking up the environment variable.
        # Also add ignore_patterns to prevent issues with problematic aux files.
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            download_dir=hf_home_cache_path, # Explicitly tell vLLM where to download/look for models
            ignore_patterns=["*.safetensors.index.json", "*.pt", "*.bin.index.json", "*.gitattributes"], # Ignore common problematic files
            **engine_kwargs
        )
        self.engine = self.AsyncLLMEngine.from_engine_args(engine_args)
        logging.info(f"AsyncVLLMWorker initialized with model: {model_name}")
        
    async def judge(
        self,
        request_id: str,
        question: str,
        response_to_judge: str,
        metadata: LLMJudgeEnvironmentMetadata,
        sampling_params_dict: dict,
    ) -> Tuple[str, float]:
        """
        Judges a single response using the LLM.

        Args:
            request_id: A unique ID for this generation request.
            question: The question string.
            response_to_judge: The assistant's response string.
            metadata: Metadata containing reference answer and criteria.
            sampling_params_dict: Dictionary to initialize vllm.SamplingParams.

        Returns:
            Tuple of (request_id, score)
        """
        reference = metadata.get("reference_answer", "")
        criteria = metadata.get("evaluation_criteria", "")
        extract_box = metadata.get("extract_box", False)

        response_to_judge = response_to_judge.split("</think>")[-1].strip()
        response_to_judge = extract_answer_from_box(response_to_judge) if extract_box else response_to_judge
        response_to_judge = "None" if response_to_judge is None else response_to_judge
        # Prioritize metadata's judge_prompt_template, then default
        # Note that if you want to use a custom judge_prompt_template, you may need to change the verdict extraction logic accordingly
        current_judge_prompt = metadata.get("judge_prompt_template") or self.default_judge_prompt_template
        
        prompt = current_judge_prompt.format(
            question=question,
            response=response_to_judge,
            reference=reference,
            criteria=criteria,
        )
        logging.info(f"Prompt: {prompt}")
        logging.info(f"question: {question}")
        logging.info(f"response_to_judge: {response_to_judge}")
        logging.info(f"reference: {reference}")
        logging.info(f"criteria: {criteria}")
        sampling_params = self.SamplingParams(**sampling_params_dict)
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        score = 0.0
        if final_output and not final_output.finished:
            logging.info(f"Request {request_id} did not finish within the token limit, but we will score it anyway. Output: {final_output}")
        elif final_output:
            generated_text = final_output.outputs[0].text.strip()
            generated_text_lower = generated_text.lower()

            has_yes = "yes" in generated_text_lower

            if has_yes:
                score = 1.0
                logging.info(f"Parsed 'yes' for request {request_id}. Score: {score}. Output: '{generated_text}'")
            else:
                score = 0.0
                logging.info(f"No 'yes' found in {request_id}. Score: {score}. Output: '{generated_text}'")
            
        else:
            logging.warning(f"No output received from LLM for request {request_id}.")

        return request_id, score


    async def judge_batch(self, batch: List[Tuple[str, str, str, LLMJudgeEnvironmentMetadata, dict]]):
        """Process a list of (request_id, question, response, metadata, sampling_params).

        Returns: List[Tuple[str, float]] in the same order.
        """
        results = []
        for req_id, q, r, meta, params in batch:
            res = await self.judge(req_id, q, r, meta, params)
            results.append(res)
        return results

@ray.remote(max_concurrency=16) # Allow concurrent processing of step calls
class LLMJudgeAsyncEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM # The environment actor itself uses system python

    def __init__(self,cluster: RayVirtualCluster, cfg: LLMJudgeAsyncConfig, name_prefix: str = "vllm_judge", workers_per_node: Optional[Union[int, List[int]]] = None,):
        self.cfg = cfg
        # self.num_workers = cfg["num_workers"]
        
        tensor_parallel_size = cfg.get("tensor_parallel_size", 1)
        self.tensor_parallel_size = tensor_parallel_size
        
        # Pass down critical environment variables (HF cache, etc.) to workers.
        env_vars_to_pass = {}
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "WANDB_API_KEY",
            "HUGGINGFACE_HUB_DISABLE_XET",  # Often set to "1" to bypass xet errors.
            "HF_TOKEN",
        ]:
            if key in os.environ:
                env_vars_to_pass[key] = os.environ[key]

        # Ensure xet is disabled to avoid CAS service issues if not explicitly set.
        env_vars_to_pass.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")


        worker_builder = RayWorkerBuilder(AsyncVLLMWorker, model_name=cfg["model_name"],
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
                max_model_len=cfg.get("max_model_len"))

        if self.tensor_parallel_size > 1:
            # For tensor parallelism, create node-aware worker groups
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)

            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=node_bundle_indices,
            )
        else:
            # Use standard worker group creation for non-TP case
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
            )

        # Number of data parallel groups is the number of tied worker groups
        self.dp_size = self.worker_group.group_count

        logging.info(f"Created {self.dp_size} tied worker groups for LLM judge")

        self._request_counter = 0
        self._actor_id_prefix = str(uuid.uuid4())[:8]
        
    def _get_tied_worker_bundle_indices(self, cluster):
        """Calculate bundle indices for tensor parallel workers."""
        # Get the placement groups (nodes) from the cluster
        placement_groups = cluster.get_placement_groups()

        tied_worker_groups = []

        # For each node (placement group), create tied worker groups of size tensor_parallel_size
        for node_idx, pg in enumerate(placement_groups):
            # How many bundles (GPUs) are on this node
            bundles_on_node = pg.bundle_count
            tied_worker_groups_on_node = bundles_on_node // self.tensor_parallel_size

            if tied_worker_groups_on_node > 0:
                for group_idx in range(tied_worker_groups_on_node):
                    # Local bundle indices for this tied worker group (consecutive GPUs on this node)
                    start_idx = group_idx * self.tensor_parallel_size
                    end_idx = start_idx + self.tensor_parallel_size
                    local_bundle_indices = list(range(start_idx, end_idx))
                    tied_worker_groups.append((node_idx, local_bundle_indices))

        if not tied_worker_groups:
            raise ValueError(
                f"Cannot create any tensor parallel tied worker groups with size {self.tensor_parallel_size}. "
                f"Make sure each node has at least {self.tensor_parallel_size} GPUs."
            )

        return tied_worker_groups
    
    def shutdown(self):
        """Shut down all vLLM workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[LLMJudgeEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """
        message_log_batch: List[List[Dict[str, str]]] - List of conversations, each conversation is a list of messages
        metadata: List[LLMJudgeEnvironmentMetadata] - List of metadata for each conversation

        Returns:
            EnvironmentReturn: A NamedTuple containing the following fields:
                - observations (List[Dict[str, str]]): A list of observations, where
                  each observation is a dictionary representing the judge's feedback
                  (e.g., `{"role": "environment", "content": "Environment: Score = X.XX"}`).
        """
        # ----------------------------------------------------------
        # Pre-process batch into lists
        # ----------------------------------------------------------

        samples = []  # list of dicts per sample
        for idx, conversation in enumerate(message_log_batch):
            assert len(conversation) == 2, (
                "LLMJudgeAsyncEnvironment only supports single-turn conversations"
            )
            user_messages = conversation[0]["content_in_oai_format"]
            assert isinstance(user_messages, list)
            question = next(
                m["content"] for m in user_messages if m["role"] == "user"
            )
            response = conversation[-1]["content"]
            samples.append((idx, question, response, metadata[idx]))

        # Default sampling params template
        default_sampling_params = {
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 512),
            "stop": self.cfg.get("stop", None),
        }

        # ----------------------------------------------------------
        # Shard samples across data-parallel vLLM engines
        # ----------------------------------------------------------
        dp = self.dp_size if getattr(self, "dp_size", 0) else 1
        shards: List[list] = [[] for _ in range(dp)]

        for sample in samples:
            shard_id = sample[0] % dp
            idx, q, r, meta = sample
            req_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_{idx}"
            shards[shard_id].append(
                (req_id, q, r, meta, default_sampling_params.copy())
            )

        futures = self.worker_group.run_all_workers_multiple_data(
            "judge_batch",
            shards,
            only_on="tied_leader",
        )

        shard_results: List[List[Tuple[str, float]]] = self.worker_group.get_all_worker_results(
            futures
        )

        # Flatten and restore original ordering
        result_dict = {}
        for shard in shard_results:
            for req_id, score in shard:
                # req_id includes original idx at the end after last '_'
                idx = int(req_id.split("_")[-1])
                result_dict[idx] = score

        scores = [result_dict[i] for i in range(len(samples))]
        
        # Increment request counter for next step
        self._request_counter += 1
        
        observations = [
            {"role": "environment", "content": f"Environment: Score = {score:.2f}"}
            for score in scores
        ]
        
        rewards_tensor = torch.tensor(scores, dtype=torch.float32).cpu()
        terminateds_tensor = torch.ones_like(rewards_tensor).cpu()
        
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata, 
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
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