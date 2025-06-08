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


def format_aime24(
    data: dict[str, str | float | int], output_key: str = "answer"
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data[output_key],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_aime24_dataset(
    output_key: str = "solution",
) -> dict[str, Dataset | None]:
    """Load and split the HuggingFaceH4/aime_2024 dataset into validation set."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset
    original_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

    # Format the examples, removing original columns
    val_formatted = original_ds.map(
        format_aime24,
        remove_columns=original_ds.column_names,
        fn_kwargs={"output_key": output_key},
    )

    return {
        "train": None,
        "validation": val_formatted,
    }


class AIME24Dataset:
    def __init__(
        self,
        output_key: str = "solution",
        prompt_file: Optional[str] = None,
    ):
        """Initialize the HuggingFaceH4/aime_2024 dataset.

        Args:
            output_key: Key to use for the output solution.
        """
        self.formatted_ds = prepare_aime24_dataset(output_key=output_key)

        self.task_spec = TaskDataSpec(
            task_name="HuggingFaceH4/aime_2024",
            prompt_file=prompt_file,
        )
