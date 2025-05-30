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

from nemo_rl.data.interfaces import TaskDataSpec


def format_so(data):
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["generation"],
            },
        ]
    }


class SODataset:

    def __init__(self, data_path: str, test_ratio: float = 0.1):
        original_ds = load_dataset("json", data_files=data_path)["train"]
        formatted_ds = original_ds.map(format_so)
        
        train_test_split = formatted_ds.train_test_split(test_ratio, seed=42)
        train_ds = train_test_split["train"]
        val_ds = train_test_split["test"]
        
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(
            task_name="SO_multi_choice",
        )