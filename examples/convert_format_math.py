"""Example to convert the DeepScaler dataset to the Nemo RL format."""

import argparse
import os

import jsonlines
from datasets import load_dataset

cot_prompt = """Think step-by-step to solve the following problem. Output your answer inside of \\boxed{{}} tags.:
{}

Let's think step-by-step"""


def format_math(data, dataset_name):
    return {
        "messages": [
            [
                {
                    "role": "user",
                    "content": cot_prompt.format(data["problem"]),
                    "metadata": {
                        "ground_truth": data["answer"],
                    },
                },
            ]
        ],
        "task_name": "math",
        "dataset": dataset_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to Nemo RL format")
    parser.add_argument(
        "--dataset",
        type=str,
        default="agentica-org/DeepScaleR-Preview-Dataset",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (train, validation, test)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="converted_data.jsonl",
        help="Output file name for the jsonlines file",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat each example in the output",
    )

    args = parser.parse_args()
    # Load the dataset with specified name and split
    original_ds = load_dataset(args.dataset, split=args.split)
    original_ds = original_ds.repeat(args.num_repeats)

    # Convert the data
    converted_data = [format_math(item, args.dataset) for item in original_ds]
    print("dataset length: ", len(converted_data))

    # Save the converted data as a jsonlines file
    with jsonlines.open(args.output_file, mode="w") as writer:
        writer.write_all(converted_data)

    print(f"Converted data saved to {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()
