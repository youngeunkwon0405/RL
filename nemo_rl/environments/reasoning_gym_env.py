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

from typing import Any, Optional, TypedDict

import random
import re
import ray
import torch
from reasoning_gym.utils import extract_answer
from reasoning_gym.composite import CompositeDataset, CompositeConfig, DatasetSpec

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

class ReasoningGymConfig(TypedDict, total=False):
    tasks: list[dict[str, Any]]  # List of tasks with name, config, weight


class ReasoningGymMetadata(TypedDict):
    puzzle_entry: dict[str, Any]
    dataset_name: str


class ReasoningGymGameLogic:    
    @staticmethod
    def generate(config: dict[str, Any]) -> dict[str, Any]:
        """Generate a new reasoning-gym problem."""
        puzzle_seed = random.randint(0, 2**31 - 1)
        
        if "tasks" in config:
            dataset_specs = []
            for task in config["tasks"]:
                task_config = task.get("dataset_config", {})
                filtered_config = {k: v for k, v in task_config.items() if k != "size"}
                dataset_specs.append(DatasetSpec(
                    name=task["dataset_name"],
                    weight=task.get("weight", 1.0),
                    config=filtered_config
                ))
            
            composite_config = CompositeConfig(
                size=1, # generate one puzzle at a time on the fly, rather than a pre-generated dataset. Might switch to pre-generated.
                seed=puzzle_seed,
                datasets=dataset_specs
            )
            dataset = CompositeDataset(composite_config)
            puzzle_entry = dataset[0]
            
            return {
                "puzzle_entry": puzzle_entry,
                "dataset_name": puzzle_entry["metadata"]["source_dataset"],
                "dataset": dataset
            }
        
        else:
            raise ValueError("Must specify 'tasks' in config")
    

    @staticmethod
    def init(game_state: dict[str, Any]) -> str:
        """Initialize the task and return the question."""
        puzzle_entry = game_state["puzzle_entry"]
        return puzzle_entry["question"]
    
    @staticmethod
    def step(
        answer: str, game_state: dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """Process the answer and return response, reward, done, new_state."""
        puzzle_entry = game_state["puzzle_entry"]
        dataset = game_state["dataset"]
        
        score = dataset.score_answer(answer, puzzle_entry)
        
        # single-turn RL (though maybe puzzles should be multi turn...)
        is_terminated = True
        
        if score >= 1.0:
            response = "Correct! Puzzle solved."
        elif score > 0:
            response = f"Partially correct (score: {score:.2f})"
        else:
            response = "Incorrect answer."
            
        return response, float(score), is_terminated, game_state
    
    @staticmethod
    def render(game_state: dict[str, Any]) -> str:
        """Render the current game state (just the question)."""
        puzzle_entry = game_state["puzzle_entry"]
        return puzzle_entry["question"]


class ReasoningGymRunner:
    """Handles reasoning-gym task interactions."""
    
    def __init__(self):
        pass  # No initialization needed as game methods are static

    def _parse_answer(self, text: str) -> Optional[str]:
        """Parse the answer from the model's response text."""
        extracted = extract_answer(text, tag_name="answer")
        if extracted is not None:
            return extracted
        
        # TODO: reasoning gym uses answer tags. Model is trained to use \boxed{...}. Maybe we shouldnt do this, and let model learn format.
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        return text.strip() if text.strip() else None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: ReasoningGymMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[ReasoningGymMetadata],
    ]:
        """Process a solution to a reasoning-gym problem (single turn)."""
        game_state = metadata["puzzle_entry"]
        
        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</answer>"]
        next_metadata = metadata.copy()
        next_observation_content = ""

        last_assistant_msg_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg_content = message_log[-1]["content"].strip()

        parsed_answer = self._parse_answer(last_assistant_msg_content)

        if parsed_answer is None:
            next_observation_content = "<environment>\nNo valid answer found. Please provide your answer in <answer></answer> tags.\n</environment>"
            next_metadata = None
            is_terminated = True
        else:
            step_response, reward, game_over, next_game_state = (
                ReasoningGymGameLogic.step(parsed_answer, game_state)
            )

            turn_reward = reward
            is_terminated = game_over
            
            next_observation_content = f"<environment>\n{step_response}\n</environment>"

            if is_terminated:
                next_metadata = None

        return (
            {"role": "environment", "content": next_observation_content + "\n"},
            turn_reward,
            is_terminated,
            next_stop_strings if not is_terminated else None,
            next_metadata,
        )


@ray.remote
class ReasoningGymEnv(EnvironmentInterface):
    """Reasoning-gym environment"""

    def __init__(self, cfg: Optional[ReasoningGymConfig] = None):
        self.game_config = cfg if cfg else {}
        self.runner = ReasoningGymRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[ReasoningGymMetadata],
    ) -> EnvironmentReturn:
        """Process a batch of reasoning-gym task interactions."""
        # Process each interaction sequentially (can parallelize if needed later)
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata_batch)
        ]

        # Unpack results and format according to EnvironmentReturn NamedTuple
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        for obs, rew, term, stops, meta in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminateds, dtype=torch.bool)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
        )

    def shutdown(self):
        pass

    # not used anywhere yet
    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards >= 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        
        avg_score = final_rewards.mean().item() if len(final_rewards) > 0 else 0.0
        
        return batch, {
            "reasoning_gym_success_rate": success_rate,
            "reasoning_gym_avg_score": avg_score,
        } 