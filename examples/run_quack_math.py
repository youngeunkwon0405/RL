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
from collections import defaultdict
from typing import Any, Dict

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.quack import MasterConfig, quack_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.hf_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.hf_datasets.math500 import Math500Dataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.environments.llm_judge_environment import LLMJudgeEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run QUACK training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def get_datasets(data_config: DataConfig):
    print("\n▶ Setting up data...")
    if data_config["dataset_name"] == "OpenMathInstruct-2":
        print("Loading nvidia/OpenMathInstruct2Dataset for training and validation")
        data = OpenMathInstruct2Dataset()
    elif data_config["dataset_name"] == "MATH500":
        print("Loading HuggingFaceH4/MATH-500 for training and validation (mostly for debugging, faster to run)")
        data = Math500Dataset()
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")
    
    return data.formatted_ds["train"], data.formatted_ds["validation"]


def get_task_to_env(env_configs: Dict[str, Any]):
    # actor env reports the accuracy of the actor's response against the ground truth
    math_env = MathEnvironment.options(
        runtime_env={
            "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"])

    # critic env reports the correctness of the actor's response against the critic judgement
    critic_env = LLMJudgeEnvironment.options(
        runtime_env={
            "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["critic"])

    task_to_env = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    task_to_env["critic"] = critic_env

    return task_to_env

def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "quack_math_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["actor"]["tokenizer"])
    config["actor"]["generation"] = configure_generation_config(
        config["actor"]["generation"], tokenizer
    )
    config["critic"]["generation"] = configure_generation_config(
        config["critic"]["generation"], tokenizer
    )

    dataset, val_dataset = get_datasets(config["data"])   # different from setup_data, these are raw
    task_to_env = get_task_to_env(config["env"])    # math and critic envs

    (
        actor,
        actor_generation,
        critic_generation,
        _, _,   # actor_cluster, critic_cluster
        loss_fn,
        logger,
        checkpointer,
        quack_state,
        master_config,
    ) = setup(config, tokenizer)  # TODO: process dataset

    quack_train(
        actor,
        actor_generation,
        critic_generation,
        dataset,
        val_dataset,
        tokenizer,
        loss_fn,
        task_to_env,
        logger,
        checkpointer,
        quack_state,
        master_config,
    )


if __name__ == "__main__":
    main()
