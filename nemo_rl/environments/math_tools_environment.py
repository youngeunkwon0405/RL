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

import re
import tempfile
from typing import Any, Optional, TypedDict
import json

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.tools.bash_tool import BashTool
from nemo_rl.environments.tools.file_tool import FileTool
from nemo_rl.environments.math_grader import extract_answer, math_equal


class MathToolsConfig(TypedDict):
    max_turns: int
    max_bash_timeout: int  # s
    max_file_size: int  # bytes
    memory_limit: int  # MB
    cpu_limit: float  # cores


class MathToolsMetadata(TypedDict):
    problem_id: str
    problem_text: str
    ground_truth: str  
    working_dir: str
    current_turn: int
    max_turns: int
    files_created: list[str]
    bash_history: list[tuple[str, str]]  # command, output
    tool_calls_count: dict[str, int]  


@ray.remote
class MathToolsEnvironment(EnvironmentInterface):
    def __init__(self, cfg: MathToolsConfig):
        self.cfg = cfg
        
        # TODO: use sandbox code execution
        self.bash_tool = BashTool(
            timeout=cfg.get("max_bash_timeout", 30),
            memory_limit=cfg.get("memory_limit", 512),
            cpu_limit=cfg.get("cpu_limit", 1.0),
        )
        self.file_tool = FileTool(
            max_file_size=cfg.get("max_file_size", 1024 * 1024)  # 1MB default
        )

    @property
    def TOOLS(self):
        """tool definitions in qwen format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute bash commands in a controlled environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_edit",
                    "description": "Create a new file or edit an existing file. If the file doesn't exist, it will be created. Replace old content with new content, or leave old_content empty to append.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file to create/edit."
                            },
                            "old_content": {
                                "type": "string",
                                "description": "The content to replace. Leave empty to append or when creating a new file."
                            },
                            "new_content": {
                                "type": "string",
                                "description": "The new content to replace the old content with."
                            }
                        },
                        "required": ["path", "old_content", "new_content"]
                    }
                }
            }
        ]

    def shutdown(self) -> None:
        pass

    def get_tools_definition(self) -> list[dict[str, Any]]:
        return self.TOOLS

    def _parse_tool_calls(self, content: str) -> list[tuple[str, dict[str, Any]]]:
        tool_calls = []
        
        # parse qwen json format: {"name": "tool_name", "parameters": {...}}
        # find potential json objects that contain "name" and "parameters" keys
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    potential_json = content[start_idx:i+1]
                    
                    try:
                        tool_call_data = json.loads(potential_json)
                        if isinstance(tool_call_data, dict) and "name" in tool_call_data and "parameters" in tool_call_data:
                            tool_name = tool_call_data.get("name")
                            parameters = tool_call_data.get("parameters", {})
                            
                            # ensure parameters is a dictionary
                            if not isinstance(parameters, dict):
                                parameters = {}
                            
                            if tool_name:
                                tool_calls.append((tool_name, parameters))
                                
                    except json.JSONDecodeError:
                        continue
                    
                    start_idx = -1
        
        return tool_calls

    def _extract_final_answer(self, content: str) -> Optional[str]:
        """extract answer from \\boxed{} LaTeX format"""
        return extract_answer(content, extract_from_boxed=True)

    def _check_answer(self, submitted: str, ground_truth: str) -> bool:
        if submitted is None:
            return False
        return math_equal(ground_truth, submitted)

    def _execute_tool(
        self, tool_name: str, tool_args: dict[str, Any], working_dir: str
    ) -> str:
        try:
            if tool_name == "bash":
                return self.bash_tool.execute(
                    command=tool_args["command"],
                    working_dir=working_dir
                )
            elif tool_name == "file_edit":
                return self.file_tool.edit(
                    path=tool_args["path"],
                    old_content=tool_args["old_content"],
                    new_content=tool_args["new_content"],
                    working_dir=working_dir
                )
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata_batch: list[MathToolsMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: list[MathToolsMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        observations = []
        rewards = []
        terminateds = []
        next_metadata = []
        next_stop_strings = []
        
        for message_log, metadata in zip(message_log_batch, metadata_batch):
            # get last assistant message
            last_assistant_msg = ""
            if message_log and message_log[-1]["role"] == "assistant":
                last_assistant_msg = message_log[-1]["content"]
            
            # check max turns
            if metadata["current_turn"] >= metadata["max_turns"]:
                observations.append({
                    "role": "environment",
                    "content": f"Error: Maximum turns ({metadata['max_turns']}) reached."
                })
                rewards.append(0.0)
                terminateds.append(True)
                next_metadata.append(None)
                next_stop_strings.append(None)
                continue
            
            # check for final answer
            final_answer = self._extract_final_answer(last_assistant_msg)
            if final_answer is not None:
                is_correct = self._check_answer(final_answer, metadata["ground_truth"])
                observations.append({
                    "role": "environment",
                    "content": f"Final answer submitted: {final_answer}. "
                              f"{'Correct!' if is_correct else f'Incorrect. Expected: {metadata['ground_truth']}'}"
                })
                rewards.append(1.0 if is_correct else 0.0)
                terminateds.append(True)
                next_metadata.append(None)
                next_stop_strings.append(None)
                continue
            
            # parse and execute tool calls
            tool_calls = self._parse_tool_calls(last_assistant_msg)
            
            if not tool_calls:
                observations.append({
                    "role": "environment",
                    "content": "No valid tool calls or final answer found. Please use the provided tools or submit your final answer using \\boxed{} format."
                })
                rewards.append(0.0)
                terminateds.append(False)
                updated_metadata = metadata.copy()
                updated_metadata["current_turn"] += 1
                next_metadata.append(updated_metadata)
                next_stop_strings.append(["</tool_call>", "<|im_end|>"]) # not sure we need to append here
                continue
            
            # execute tools
            tool_results = []
            updated_metadata = metadata.copy()
            
            for tool_name, tool_args in tool_calls:
                # validate that tool_args is a dictionary
                if not isinstance(tool_args, dict):
                    result = f"Error: Invalid tool arguments format for {tool_name}. Expected dictionary, got {type(tool_args).__name__}"
                    tool_results.append(f"[{tool_name}]\n{result}")
                    continue
                
                result = self._execute_tool(tool_name, tool_args, metadata["working_dir"])
                tool_results.append(f"[{tool_name}]\n{result}")
                
                # update metadata
                updated_metadata["tool_calls_count"][tool_name] = (
                    updated_metadata["tool_calls_count"].get(tool_name, 0) + 1
                )
                
                if tool_name == "bash":
                    command = tool_args.get("command", "")
                    updated_metadata["bash_history"].append((command, result))
                elif tool_name in ["file_edit"]:
                    path = tool_args.get("path", "")
                    if path and path not in updated_metadata["files_created"]:
                        updated_metadata["files_created"].append(path)
            
            # create observation
            observation_content = "\n\n".join(tool_results)
            observations.append({
                "role": "environment",
                "content": observation_content
            })
            rewards.append(0.0)  # no reward until final answer
            terminateds.append(False)
            updated_metadata["current_turn"] += 1
            next_metadata.append(updated_metadata)
            next_stop_strings.append(["</tool_call>", "<|im_end|>"])
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminateds_tensor = torch.tensor(terminateds, dtype=torch.bool)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        accuracy = batch["rewards"].mean().item()
        
        total_tool_calls = 0
        tool_usage = {"bash": 0, "file_edit": 0}
        total_turns = 0
        num_problems = 0
        
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
        
        if "extra_env_info" in batch:
            for info in batch["extra_env_info"]:
                if info:
                    if "tool_calls_count" in info:
                        for tool, count in info["tool_calls_count"].items():
                            if tool in tool_usage:
                                tool_usage[tool] += count
                            total_tool_calls += count
                    
                    if "current_turn" in info:
                        total_turns += info["current_turn"]
                        num_problems += 1
        
        avg_turns_per_problem = total_turns / num_problems if num_problems > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch.get("text", []), batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "avg_turns_per_problem": avg_turns_per_problem,
            "total_tool_calls": total_tool_calls,
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
            **{f"tool_usage_{k}": v for k, v in tool_usage.items()},
        }
        
        return batch, metrics


def create_working_directory() -> str:
    return tempfile.mkdtemp(prefix="math_tools_") 

