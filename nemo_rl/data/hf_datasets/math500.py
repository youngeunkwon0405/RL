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


def format_math(data):
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["answer"], # Ensure solution is a string
            },
        ],
        "task_name": "math", # Consistent task name
    }


def prepare_math500_dataset(seed=42, test_size=0.1): # MATH dataset is smaller, using a slightly larger test split
    """Load and split the MATH-500 dataset into train and validation sets."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset - MATH dataset is small, load entire 'test' split
    original_ds = load_dataset("HuggingFaceH4/MATH", split="test") 

    # Split into train and validation sets
    # MATH dataset doesn't have a predefined validation split, so we create one.
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)
    train_formatted = split_ds["train"].map(
        format_math, remove_columns=split_ds["train"].column_names
    )
    val_formatted = split_ds["test"].map(
        format_math, remove_columns=split_ds["test"].column_names
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class Math500Dataset:
    def __init__(self, seed: int = 42, test_size: float = 0.1):
        """Initialize the Math500 dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0).
        """
        self.formatted_ds = prepare_math500_dataset(seed=seed, test_size=test_size)

        self.task_spec = TaskDataSpec(
            task_name="MATH500", # More specific task name
        ) 