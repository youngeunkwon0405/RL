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
from nemo_rl.environments.llm_judge_environment import LLMJudgeEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def create_envs(env_configs, task_keys, judge_handle=None):
    """Create Ray environment actors.

    Args:
        env_configs: Block from YAML under `env`
        task_keys: Iterable of task names
        judge_handle: ActorHandle or None
    Returns:
        task_to_env, val_task_to_env maps
    """
    judge_env = LLMJudgeEnvironment.options(
        runtime_env={
            "py_executable": LLMJudgeEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["llm_judge"], judge_llm_handle=judge_handle)

    task_to_env = {k: judge_env for k in task_keys}
    return task_to_env, task_to_env

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    args, overrides = parser.parse_known_args()

    return args, overrides


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_llama_nemotron_mock_1B.yaml"
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
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )