"""Convert synthetic BFCL dataset to the Nemo RL format."""

import argparse
import os
import jsonlines


def format_bfcl(data):
    """Convert a single synthetic BFCL example to NeMo-RL format.
    
    Args:
        data: A synthetic BFCL example:
            - id: unique identifier
            - prompt: the instruction and question with available tools
            - answers: list of expected function calls (ground truth)
            - args: contains ground_truth and metadata
    
    Returns:
        A formatted data entry for NeMo-RL training
    """
    
    if "answers" not in data:
        raise ValueError(f"No 'answers' field found in data entry {data.get('id', 'unknown')}")
    
    if "prompt" not in data:
        raise ValueError(f"No 'prompt' field found in data entry {data.get('id', 'unknown')}")
    
    return {
        "messages": [
            [
                {
                    "role": "user",
                    "content": data["prompt"],
                    "metadata": {
                        "ground_truth": data["answers"],
                    },
                },
            ]
        ],
        "task_name": "bfcl",
        "dataset": "bfcl",
    }


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic BFCL dataset to Nemo RL format")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with BFCL data",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file name for the converted jsonlines file",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat each example in the output",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert",
    )

    args = parser.parse_args()
    
    original_data = []
    with jsonlines.open(args.input_file, "r") as reader:
        for i, line in enumerate(reader):
            if args.max_samples and i >= args.max_samples:
                break
            original_data.append(line)
    
    print(f"Loaded {len(original_data)} samples from {args.input_file}")

    converted_data = []
    for item in original_data:
        try:
            formatted_item = format_bfcl(item)
            for _ in range(args.num_repeats):
                converted_data.append(formatted_item)
        except Exception as e:
            print(f"Error converting item {item.get('id', 'unknown')}: {e}")
            continue
    
    print(f"Converted {len(converted_data)} samples")

    with jsonlines.open(args.output_file, mode="w") as writer:
        writer.write_all(converted_data)

    print(f"Converted data saved to {os.path.abspath(args.output_file)}")
    
    if converted_data:
        print("\n=== Example converted data ===")
        sample = converted_data[0]
        print(f"Task name: {sample['task_name']}")
        print(f"Dataset: {sample['dataset']}")
        print(f"User content: {sample['messages'][0][0]['content']}")
        print(f"Ground truth: {sample['messages'][0][0]['metadata']['ground_truth']}")


if __name__ == "__main__":
    main() 