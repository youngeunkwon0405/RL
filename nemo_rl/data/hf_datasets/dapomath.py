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


from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_dapomath(
    data: dict[str, str | float | int], output_key: str = "solution"
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"],
            },
            {
                "role": "assistant",
                "content": data[output_key],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_dapomath17k_dataset(
    output_key: str = "solution",
) -> dict[str, Dataset | None]:
    """Load and split the DAPO-Math-17k-Processed dataset into train set."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset
    original_ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "all")

    # Format the examples, removing original columns
    train_formatted = original_ds["train"].map(
        format_dapomath,
        remove_columns=original_ds["train"].column_names,
        fn_kwargs={"output_key": output_key},
    )
    val_formatted = None

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DAPOMathDataset:
    def __init__(
        self,
        output_key: str = "solution",
        prompt_file: Optional[str] = None,
    ):
        """Initialize the DAPO-Math-17k-Processed train dataset.

        Args:
            output_key: Key to use for the output solution.
        """
        self.formatted_ds = prepare_dapomath17k_dataset(output_key=output_key)

        self.task_spec = TaskDataSpec(
            task_name="DAPO-Math-17k-Processed",
            prompt_file=prompt_file,
        )
