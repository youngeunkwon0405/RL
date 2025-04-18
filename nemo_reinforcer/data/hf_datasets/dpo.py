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
from datasets import load_dataset

from nemo_reinforcer.data.interfaces import TaskDataSpec


class DPODataset:
    """Dataset class for Direct Preference Optimization (DPO) training.

    This class handles loading of preference data for DPO training.
    The input JSON files should contain examples with the following structure:
    {
        "prompt": str,           # The input prompt/context
        "chosen_response": str,  # The preferred/winning response
        "rejected_response": str # The non-preferred/losing response
        "chosen_reward": float, # (optional) The reward for the preferred response
        "rejected_reward": float # (optional) The reward for the rejected response
    }

    Args:
        train_data_path (str): Path to the JSON file containing training data
        val_data_path (str): Path to the JSON file containing validation data

    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        default_chosen_reward: float = 1.0,
        default_rejected_reward: float = 0.0,
    ):
        self.default_chosen_reward = default_chosen_reward
        self.default_rejected_reward = default_rejected_reward

        self.formatted_ds = {
            "train": load_dataset("json", data_files=train_data_path, split="train"),
            "validation": load_dataset("json", data_files=val_data_path, split="train"),
        }

        self.formatted_ds["train"] = self.formatted_ds["train"].map(
            self.add_default_rewards
        )
        self.formatted_ds["validation"] = self.formatted_ds["validation"].map(
            self.add_default_rewards
        )

        self.task_spec = TaskDataSpec(
            task_name="DPO",
        )

    def add_default_rewards(self, example):
        if "chosen_reward" not in example:
            example["chosen_reward"] = self.default_chosen_reward
        if "rejected_reward" not in example:
            example["rejected_reward"] = self.default_rejected_reward

        return example
