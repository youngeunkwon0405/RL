"""Convert LiveCodeBench dataset to the Nemo RL format."""

import argparse
import os
import json

import jsonlines


FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def format_lcb(data, dataset_name):
    """Format LiveCodeBench data to NeMo RL format."""
    prompt = data['prompt']
    
    format_prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    format_prompt += f"Question: {prompt}\n\n"
    
    if data.get('starter_code', None):
        format_prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        format_prompt += f"```python\n{data['starter_code']}\n```\n\n"
    else:
        format_prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        format_prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    
    return {
        "messages": [
            [
                {
                    "role": "user",
                    "content": format_prompt,
                    "metadata": {
                        "global_id": data['global_id'],
                        "question_id": data['question_id'],
                        "tests": data['tests'],
                        "starter_code": data.get('starter_code', None),
                    },
                },
            ]
        ],
        "task_name": "lcb",
        "dataset": dataset_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert LiveCodeBench dataset to Nemo RL format")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the LiveCodeBench JSONL file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="lcb_converted.jsonl",
        help="Output file name for the jsonlines file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="livecodebench",
        help="Name of the dataset to include in metadata",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat each example in the output",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to convert (default: all)",
    )

    args = parser.parse_args()
    
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]
    
    if args.num_problems:
        data = data[:args.num_problems]
    
    print(f"Loaded {len(data)} problems from {args.input_file}")
    
    converted_data = []
    for item in data:
        formatted_item = format_lcb(item, args.dataset_name)
        for _ in range(args.num_repeats):
            converted_data.append(formatted_item)
    
    print(f"Total dataset length after {args.num_repeats}x repetition: {len(converted_data)}")
    
    with jsonlines.open(args.output_file, mode="w") as writer:
        writer.write_all(converted_data)
    
    print(f"Converted data saved to {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main() 