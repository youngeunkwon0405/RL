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


def format_custom(data):
    return {
        "messages": [
            {
                "role": "user",
                "content": data["input"],
            },
            {
                "role": "assistant",
                "content": data["output"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }

def prepare_custom_dataset(data_path):
    """Load the custom dataset from a JSONL file."""
    # Load the dataset from the provided path
    dataset = load_dataset('json', data_files=data_path)
    
    # Format the examples, removing original columns
    formatted_ds = dataset['train'].map(
        format_custom, remove_columns=dataset['train'].column_names
    )

    return formatted_ds


class CustomDataset:
    def __init__(
        self, train_path=None, val_path=None
    ):
        """Initialize the Custom dataset with train and validation data.

        Args:
            train_path: Path to the training data JSONL file
            val_path: Path to the validation data JSONL file
        """
        self.formatted_ds = {}
        
        if train_path:
            self.formatted_ds["train"] = prepare_custom_dataset(train_path)
        
        if val_path:
            self.formatted_ds["validation"] = prepare_custom_dataset(val_path)
            
        if not train_path or not val_path:
            raise ValueError("Both train_path or val_path must be provided")

        self.task_spec = TaskDataSpec(
            task_name="Custom",
        )
