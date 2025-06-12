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
import ast
import contextlib
import io
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from Levenshtein import ratio

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


class BFCLEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None

class BFCLEnvironmentMetadata(TypedDict):
    ground_truth: object


def decode_unicode(value):
    """Helper to decode Unicode escape sequences properly"""
    if isinstance(value, str):
        try:
            # Decode Unicode escape sequences
            decoded = value.encode().decode("unicode_escape")
            # Handle double-encoded UTF-8 characters
            return decoded.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return value    # If decoding fails, return the original value
    return value


def extract_function_calls(node):
    if isinstance(node, ast.Expression):  # Handle root expression
        return extract_function_calls(node.body)
    elif isinstance(node, ast.Expr):  # Handle top-level expressions
        return extract_function_calls(node.value)
    elif isinstance(node, ast.List):  # If it's a list, process its elements
        return [extract_function_calls(elt) for elt in node.elts]
    elif isinstance(node, ast.Call):  # If it's a function call
        function_name = node.func.id  # Extract function name
        arguments = {}
        for keyword in node.keywords:
            key = keyword.arg
            value = extract_function_calls(keyword.value)
            # Ensure lists remain lists inside arguments
            if isinstance(value, (dict, list)):
                arguments[key] = [value]  # Wrap dictionaries and lists in a list
            else:
                arguments[key] = [decode_unicode(value)]  # Decode Unicode and wrap scalars in a list
        return {function_name: arguments}
    elif isinstance(node, ast.Dict):  # Handle dictionary literals
        return {extract_function_calls(k): extract_function_calls(v) for k, v in zip(node.keys, node.values)}
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):  # Handle negative numbers
        value = extract_function_calls(node.operand)
        return -value if isinstance(value, (int, float)) else None
    elif isinstance(node, ast.Constant):  # If it's a constant value (int, str, float, bool, etc.)
        return decode_unicode(node.value) if isinstance(node.value, str) else node.value
    elif isinstance(node, ast.NameConstant):  # Handle booleans (Python 3.6 and below)
        return node.value
    elif isinstance(node, ast.Name):  # Handle booleans (True/False in Python 3.8+)
        if node.id == "true":
            return True
        if node.id == "false":
            return False
    return None  # Default case


def parsing(result):
    # Nemotron specific
    result = result.replace("\n", "").strip()
    if "<TOOLCALL>" in result and "</TOOLCALL>" in result:
        extracted = re.findall(r'<TOOLCALL>(.*?)</TOOLCALL>', result)
        result = [item[1:-1] if item.startswith('[') and item.endswith(']') else item for item in extracted]
        
    elif "<functioncall>" in result and "</functioncall>" in result:
        extracted = re.findall(r'<functioncall>(.*?)</functioncall>', result)
        result = [item[1:-1] if item.startswith('[') and item.endswith(']') else item for item in extracted]

    result = str(result).strip()

    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    if result.startswith("['"):
        result = result.replace("['", "[")
        result = result.replace("', '", ", ")
        result = result.replace("','", ", ")
    if result.endswith("']"):
        result = result.replace("']", "]")
    
    if result.startswith("[\""):
        result = result.replace("[\"", "[")
        result = result.replace("\", \"", ", ")
        result = result.replace("\",\"", ", ")
    if result.endswith("\"]"):
        result = result.replace("\"]", "]")
    
    result = result.replace("\"(", "(")
    result = result.replace("\"[", "[")
    result = result.replace("]\"", "]")

    return result


def match_percentage(l1, l2):
    try:
        j1 = json.dumps(l1, sort_keys=True, indent=1).replace("\n", " ")
        clean_j1 = re.sub(r"[{}\[\],:]", "", j1).strip().split()
        j2 = json.dumps(l2, sort_keys=True, indent=1).replace("\n", " ")
        clean_j2 = re.sub(r"[{}\[\],:]", "", j2).strip().split()
    except Exception as e:
        clean_j1 = 0.0
        clean_j2 = 0.0
    return ratio(clean_j1, clean_j2)


@ray.remote
class BFCLVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BFCL

    def __init__(self):
        logging.getLogger("bfcl_verify").setLevel(logging.CRITICAL)

    def ast_rewards(self, model_response: str, ground_truth: object) -> float:
        print(f"\n--------------------------------\nraw model response: {model_response}", flush=True)
        
        try:
            model_response = parsing(model_response)
        except Exception as e:
            print(f"{e} TOOLCALL error in response \ns:{model_response}")
            return 0.0

        functions = []
        try:
            parsed = ast.parse(model_response, mode="eval")
            functions = extract_function_calls(parsed)
        except Exception as e:
            print(f"{e} in \ns:{model_response}")
            return 0.0
            
        exact_match = int(ground_truth == functions)
        percentage_result = match_percentage(ground_truth, functions)
        percentage_result = percentage_result**2
        percentage_result = 0.0 
        result = max(exact_match, percentage_result)
        
        arg_gt_str = json.dumps(ground_truth, sort_keys=True)
        functions_str = json.dumps(functions, sort_keys=True)
        
        print(f"s={model_response}", flush=True)
        print(f"model_answer={functions}", flush=True)
        print(f"expected_answer={ground_truth}", flush=True)
        print(f"arg_gt_str={arg_gt_str}", flush=True)
        print(f"functions_str={functions_str}", flush=True)
        print(f"exact_match={exact_match}, similarity={percentage_result}", flush=True)
        print(f"result={result}", flush=True)
        
        return float(result)

    def verify(
        self, pred_responses: List[str], ground_truths: List[object]
    ) -> List[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            ground_truths: List[object]. The ground truth function call structures.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                reward = self.ast_rewards(response, ground_truth)
                results.append(reward)
            except Exception as e:
                print(f"Error in verify: {e}")
                results.append(0.0)
        return results


@ray.remote
class BFCLEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BFCL

    def __init__(self, cfg: BFCLEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            BFCLVerifyWorker.options(
                runtime_env={"py_executable": BFCLVerifyWorker.DEFAULT_PY_EXECUTABLE}
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
        metadata: List[BFCLEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the BFCL environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[BFCLEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

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
            # Take the last assistant response, handling thinking tags
            response = assistant_responses[-1] if assistant_responses else ""
            # Remove thinking tags if present
            response = response.split("</think>")[-1].strip()
            assistant_response_batch.append(response)

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # Process each chunk in parallel
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
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
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