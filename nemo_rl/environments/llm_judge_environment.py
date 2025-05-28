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


# def extract_verdict(critique: str) -> int:
#     """
#     Extract the verdict from the critique.
#     Returns 1 if the verdict is "right" or "correct", and 0 otherwise.
#     """
#     try:
#         verdict_text_to_parse = ""
        
#         # Try to find "Conclusion: " (case-insensitive)
#         conclusion_marker = "Conclusion:"
#         idx = critique.lower().find(conclusion_marker.lower())
#         if idx != -1:
#             verdict_text_to_parse = critique[idx + len(conclusion_marker):].strip()
#         else:
#             # Fallback to "Verdict: " (case-insensitive)
#             verdict_marker = "Verdict:"
#             idx = critique.lower().find(verdict_marker.lower())
#             if idx != -1:
#                 verdict_text_to_parse = critique[idx + len(verdict_marker):].strip()
#             else:
#                 # If no clear marker is found, a verdict cannot be reliably extracted.
#                 return 0

#         processed_verdict_text = verdict_text_to_parse.lower().replace("**", "")
        
#         # Check for positive keywords (e.g., "right", "correct")
#         # Ensure they are not negated (e.g., "not right")
#         is_positive = (
#             ("right" in processed_verdict_text or "correct" in processed_verdict_text) and \
#             ("not right" not in processed_verdict_text and "not correct" not in processed_verdict_text)
#         )

#         if is_positive:
#             return 1
        
#         # For any other case (including explicit "wrong", "incorrect", or ambiguity), return 0.
#         return 0

#     except Exception:
#         # In case of any error during parsing, default to 0.
#         return 0


def extract_verdict(critique):
    """
    Extract the verdict from the critique.
    Returns 1.0 if the verdict is "right" and 0.0 if the verdict is "wrong".
    Returns -1.0 if the verdict is not found or if there is an error.
    We always consider -1.0 as a failed verdict and do not use it in our training.
    """
    try:
        # Find the verdict after "Conclusion: "
        conclusion_start = critique.find("Conclusion")
        if conclusion_start == -1:
            return -1.0

        verdict_text = critique[conclusion_start + len("Conclusion: "):].strip().lower()
        verdict_text = verdict_text.replace("**", "")
        verdict_text = verdict_text

        # Check if verdict is either "right" or "wrong"
        if "right" in verdict_text or "correct" in verdict_text:
            return 1.0
        elif "wrong" in verdict_text or "incorrect" in verdict_text:
            return 0.0
        else:
            return -1.0
    except:
        return -1.0


class LLMJudgeEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env


@ray.remote
class VerdictExtractionWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self):
        pass

    def verify_critiques(self, critiques: List[str]) -> List[float]:
        """
        Extracts verdicts from a list of critiques.
        Returns a list of rewards (1.0 for "right"/"correct", 0.0 otherwise).
        """
        results = []
        for critique in critiques:
            try:
                verdict_score = extract_verdict(critique)
                results.append(float(verdict_score))
            except Exception:
                results.append(0.0) 
        return results


class LLMJudgeEnvironmentMetadata(TypedDict, total=False):
    """
    Metadata for the LLM Judge Environment. Can be extended as needed.
    It's passed through the step function.
    """
    # Example: prompt_id: str
    # Example: original_query: str
    pass


@ray.remote
class LLMJudgeEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: LLMJudgeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            VerdictExtractionWorker.options(
                runtime_env={"py_executable": VerdictExtractionWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[LLMJudgeEnvironmentMetadata],
    ) -> EnvironmentReturn:
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        chunked_assistant_critiques = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )

        futures = [
            self.workers[i].verify_critiques.remote(chunk)
            for i, chunk in enumerate(chunked_assistant_critiques)
        ]

        results = ray.get(futures)
        results = [item for sublist in results for item in sublist]  # Flatten results

        observations = [
            {
                "role": "environment",
                "content": "Environment: verdict right"
                if result == 1.0
                else "Environment: verdict wrong",
            }
            for result in results
        ]

        rewards = torch.tensor(results, dtype=torch.float32).cpu()
        # Assuming each critique step is a terminal step in this environment's context
        terminateds = torch.ones_like(rewards, dtype=torch.bool).cpu()

        next_stop_strings = [None] * len(message_log_batch)


        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        
        # Mask rewards for sequences that didn't end properly (e.g., truncated)
        if "is_end" in batch and batch["is_end"].numel() == batch["rewards"].numel():
             batch["rewards"] = batch["rewards"] * batch["is_end"].float()
        else:
            # if is_end is not available or mismatched, proceed without masking but maybe log a warning
            pass


        generation_lengths_for_right_verdict = 0.0
        if "rewards" in batch and batch["rewards"].numel() > 0 and (batch["rewards"] == 1).float().sum() > 0:
            if (
                "generation_lengths" in batch and "prompt_lengths" in batch and
                batch["generation_lengths"].numel() == batch["rewards"].numel() and
                batch["prompt_lengths"].numel() == batch["rewards"].numel()
            ):
                valid_indices = batch["rewards"] == 1
                generation_lengths_for_right_verdict = (
                    (batch["generation_lengths"][valid_indices] - batch["prompt_lengths"][valid_indices])
                    .float()
                    .mean()
                    .item()
                )

        accuracy = batch["rewards"].mean().item() if "rewards" in batch and batch["rewards"].numel() > 0 else 0.0
        
        pass_rate = 0.0
        if "text" in batch and "rewards" in batch and batch["text"] and batch["rewards"].numel() > 0:
             # Ensure batch["text"] is a list of strings as expected by calculate_pass_rate_per_prompt
             if isinstance(batch["text"], list) and all(isinstance(t, str) for t in batch["text"]):
                pass_rate = calculate_pass_rate_per_prompt(batch["text"], batch["rewards"])


        fraction_ended = 0.0
        if "is_end" in batch and batch["is_end"].numel() > 0:
            fraction_ended = batch["is_end"].float().mean().item()
        
        num_problems = 0
        if "is_end" in batch and hasattr(batch["is_end"], "shape"):
             num_problems = batch["is_end"].shape[0]
        elif "rewards" in batch and hasattr(batch["rewards"], "shape"):
             num_problems = batch["rewards"].shape[0]


        gen_len_mean = 0.0
        if "generation_lengths" in batch and batch["generation_lengths"].numel() > 0:
            gen_len_mean = batch["generation_lengths"].float().mean().item()

        prompt_len_mean = 0.0
        if "prompt_lengths" in batch and batch["prompt_lengths"].numel() > 0:
            prompt_len_mean = batch["prompt_lengths"].float().mean().item()


        metrics = {
            "accuracy": accuracy,
            "pass@samples_per_prompt": pass_rate,
            "fraction_of_samples_properly_ended": fraction_ended,
            "num_problems_in_batch": num_problems,
            "generation_lengths_mean": gen_len_mean,
            "prompt_lengths_mean": prompt_len_mean,
            "generation_lengths_for_right_verdict": generation_lengths_for_right_verdict,
        }
        
        # Ensure all metric values are Python floats/ints
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()

        return batch, metrics

