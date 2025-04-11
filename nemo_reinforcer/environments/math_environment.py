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
from itertools import tee
import re
import random
from typing import Dict, List, Tuple, TypedDict

import ray
import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
from sympy import zoo, nan

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.environments.interfaces import EnvironmentInterface
from nemo_reinforcer.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_reinforcer.environments.utils import chunk_list_to_workers
from nemo_reinforcer.distributed.virtual_cluster import PY_EXECUTABLES


class MathEnvConfig(TypedDict):
    num_workers: int


@ray.remote
class HFVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def calculate_accuracy_reward(self, completion, solution):
        """Reward function that checks if the completion is the same as the ground truth."""
        if completion.strip() == "":
            return -1.0

        last_boxed_str = self.last_boxed_only_string(completion)
        if last_boxed_str is None:
            return -1.0

        # remove \boxed
        if last_boxed_str[7:-1].strip() == solution.strip():
            return 1.0

        def _is_valid(parsed_result):
            if parsed_result is None:
                return False
            if len(parsed_result) == 0:
                return False
            if (
                parsed_result[0] is zoo
                or parsed_result[0] is nan
                or parsed_result[1] is zoo
                or parsed_result[1] is nan
            ):
                return False
            return True

        gold_parsed = parse(
            f"\\boxed{{{solution}}}",
            extraction_mode="first_match",
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        # equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
                # StringExtractionConfig()
            ],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                last_boxed_str,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            # equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    ),
                    ExprExtractionConfig(),
                    # StringExtractionConfig()
                ],
                extraction_mode="first_match",
            )
            if (
                len(answer_parsed) == 2
                and len(gold_parsed) == 2
                and _is_valid(answer_parsed)
                and _is_valid(gold_parsed)
            ):
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                if verify(answer_parsed, gold_parsed):
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            return 1.0

    def is_format_correct(self, completion):
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        # pattern = r"^<think>.*?</think>"
        if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            return False
        # check if all tags only appear once
        tags = ["<think>", "</think>", "<answer>", "</answer>"]
        # tags = ["<think>", "</think>"]
        for tag in tags:
            if completion.count(tag) != 1:
                return False

        # check if <think>...</think> is empty
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
        if think_match and think_match.group(1).strip() == "":
            return False

        return True

    def extract_answer_part(self, response):
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1)
        return ""

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
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                final_answer = self.extract_answer_part(response)
                do_print = False
                if random.randint(0, 512) == 1:
                    do_print = True
                if not self.is_format_correct("<think>" + response):
                    if do_print:
                        print(f"[Invalid Format] Response Case: {response}")
                    results.append(0.0)
                    continue
                accuracy_reward = self.calculate_accuracy_reward(
                    final_answer, ground_truth
                )
                if do_print:
                    print(f"Response Case: {response}")
                    print(
                        f"[Reward: {accuracy_reward}] Answer Case: {final_answer} <====> GT: {ground_truth}"
                    )
                if accuracy_reward == 1.0:
                    results.append(1.0)
                else:
                    results.append(0.0)
            except Exception:
                results.append(0)
        return results


class MathEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote
class MathEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: Dict):
        self.num_workers = cfg["num_workers"]
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": HFVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
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
    ):
        """Runs a step in the math environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
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

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {"role": "user", "content": "correct" if result else "incorrect"}
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        return observations, metadata, rewards, done

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
