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
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from examples.run_grpo_math import gpqa_data_processor, math_data_processor
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import remap_dataset_keys
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
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


def map_answer_to_shuffled_options(sample):
    problem = sample["Question"]
    choices = "ABCD"
    options = [
        sample["Correct Answer"],
        sample["Incorrect Answer 1"],
        sample["Incorrect Answer 2"],
        sample["Incorrect Answer 3"],
    ]
    truth = sample["Correct Answer"]
    random.shuffle(options)
    answer = choices[options.index(truth)]

    return {
        "problem": problem,
        "options": options,
        "expected_answer": r"\boxed{" + answer + "}",
    }


def setup_data(tokenizer: AutoTokenizer, data_config: MathDataConfig, env_configs):
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # load dataset
    base_dataset = load_dataset(
        data_config["dataset_name"], data_config["dataset_subset"]
    )
    if data_config["dataset_split"] is not None:
        base_dataset = base_dataset[data_config["dataset_split"]]

    # remap problem and solution keys
    if os.path.basename(data_config["dataset_name"]) == "gpqa":
        seed = data_config.get("seed", 42)
        print(f"Using seed {seed} for shuffle choices in GPQA")
        random.seed(seed)
        remapped_dataset = base_dataset.map(map_answer_to_shuffled_options)
        task_data_processor = gpqa_data_processor
    else:
        remapped_dataset = remap_dataset_keys(
            base_dataset,
            mapping_dict={
                data_config["problem_key"]: "problem",
                data_config["solution_key"]: "expected_answer",
            },
        )
        task_data_processor = math_data_processor

    math_env = MathEnvironment.options(
        runtime_env={"py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE}
    ).remote(env_configs["math"])

    dataset = AllTaskProcessedDataset(
        dataset=remapped_dataset,
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=task_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, math_env, tokenizer


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
        math_env,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    (
        vllm_generation,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    # Run evaluation
    run_env_eval(
        vllm_generation,
        dataloader,
        math_env,
        master_config,
    )


if __name__ == "__main__":
    main()
