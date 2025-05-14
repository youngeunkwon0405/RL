import argparse
import json
import os
import sys

current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pprint

from data_processors import (
    code_processor,
    genrm_processor,
    llm_judge_scp_116k_processor,
    math_processor,
)
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.llm_judge_environment import MockLLMJudgeEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, env_configs):
    print("\n▶ Setting up data...")

    # TODO: ykarnati - Handle multiple tasks within and corresponding environments and processors
    # for now we only have one task - llm_judge_scp116k

    math_task_spec = TaskDataSpec(
        task_name="llm_judge_scp116k",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )
    # TODO: ykarnati - create spec for each task type
    dataset_name = data_config["dataset_name"]
    assert dataset_name == "llama_nemotron", "Only llama_nemotron is supported for now"
    dataset_path = data_config["dataset_path"]
    assert os.path.exists(dataset_path), f"{dataset_path} must exist"

    task_data_processors = {
        "llm_judge_scp116k": (math_task_spec, llm_judge_scp_116k_processor),
        "code": (math_task_spec, code_processor),  # reuse default spec for now
        "genrm": (math_task_spec, genrm_processor),
        "math": (math_task_spec, math_processor),
    }

    # Instantiate the mock judge environment
    judge_env = MockLLMJudgeEnvironment.options(
        runtime_env={
            "py_executable": MockLLMJudgeEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"])

    # we need to load jsonl file to Dataset

    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue

    train_dataset = AllTaskProcessedDataset(
        data,
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    # use same dataset for validation
    val_dataset = train_dataset

    # Alias all tasks to the same mock environment for now
    # TODO: ykarnati - create env for each task type
    task_to_env = {k: judge_env for k in task_data_processors.keys()}

    return train_dataset, val_dataset, task_to_env, task_to_env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args, overrides = parser.parse_known_args()

    return args, overrides


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_llama_nemotron_mock_1B.yaml")

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
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    # setup data
    (
        train_dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"])

    (
        policy,
        policy_generation,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
