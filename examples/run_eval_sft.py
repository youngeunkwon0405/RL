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

import argparse
import os
import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from examples.run_sft import sft_preprocessor
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, hf_datasets
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.evals.eval_sft import MasterConfig, run_sft_eval, setup
from nemo_rl.models.generation.interfaces import configure_generation_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config: MathDataConfig, env_configs):
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # load dataset
    base_dataset = load_dataset(data_config["dataset_name"])
    if data_config["dataset_key"] is not None:
        base_dataset = base_dataset[data_config["dataset_key"]]
    dataset = hf_datasets.PromptResponseDataset(
        data_config["train_data_path"],
        data_config["val_data_path"],
        data_config["input_key"],
        data_config["output_key"],
    ).formatted_ds["validation"]

    dataset = AllTaskProcessedDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=sft_preprocessor,  # math_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "eval.yaml")

    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    # Setup data
    (
        dataset,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    (
        policy,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    # Run evaluation
    run_sft_eval(
        policy,
        dataloader,
        master_config,
        tokenizer,
    )


if __name__ == "__main__":
    main()
