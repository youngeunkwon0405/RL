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
import itertools  # For infinite counter
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Iterator  # Added Iterator

import torch  # Added torch import
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# === MODIFIED: Use IterableDataset ===
from torch.utils.data import Dataset, IterableDataset  # Import IterableDataset

# === Core Imports (Keep from math example) ===
from nemo_reinforcer.algorithms.grpo import (
    MasterConfig,
    grpo_train,
    setup,
)  # CRITICAL: Keep imported setup
from nemo_reinforcer.algorithms.utils import get_tokenizer

# from nemo_reinforcer.data import DataConfig # Keep if setup needs it, maybe remove later
# from nemo_reinforcer.data.interfaces import TaskDataSpec # Remove later if not needed by setup_puzzle_data
from nemo_reinforcer.distributed.virtual_cluster import init_ray
from nemo_reinforcer.models.generation.interfaces import configure_generation_config
from nemo_reinforcer.utils.config import load_config, parse_hydra_overrides
from nemo_reinforcer.utils.logger import get_next_experiment_dir

from tests.unit.environments.sliding_puzzle_game import SlidingPuzzleGame
from tests.unit.test_envs import SlidingPuzzleEnv, SlidingPuzzleMetadata
from nemo_reinforcer.data.interfaces import LLMMessageLogType, DatumSpec


def generate_puzzle_datum(
    tokenizer,
    game_config: Dict,
    max_moves: int,
    task_name: str,
    idx: int,
    policy_model_name: str,
) -> DatumSpec:
    """Generates a single sliding puzzle datum (prompt and metadata)."""
    # (Content copied from previous correct version)
    initial_game_state = SlidingPuzzleGame.generate(game_config)
    initial_render = SlidingPuzzleGame.render(initial_game_state)
    welcome_message = SlidingPuzzleGame.init(initial_game_state)
    puzzle_size = game_config.get("size", 3)
    prompt_instructions = (
        f"{welcome_message}\n\n"
        f"Current Board State:\n{initial_render}\n\n"
        f"Reach the goal state where numbers are ordered 1 through {puzzle_size**2 - 1} "
        f"with the empty space (0) at the bottom right.\n"
        f"Valid actions: 'up', 'down', 'left', 'right', or 'slide row col' (e.g., 'slide 1 2').\n"
        f"After thinking, output your chosen action on a new line starting with 'Action:' like this:\nAction: your_action"
        f"\nThink step-by-step before acting.\n"
    )
    add_system_prompt = "chat" in policy_model_name.lower()
    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_instructions}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    tokenized_prompt = tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]
    metadata = SlidingPuzzleMetadata(
        game_state=initial_game_state, num_moves=0, max_moves=max_moves
    )
    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
    }
    return datum


# === MODIFIED: Replace PreGeneratedPuzzleDataset with IterablePuzzleDataset ===
class IterablePuzzleDataset(IterableDataset):
    """An IterableDataset that generates sliding puzzle data indefinitely."""

    # === MODIFIED: Removed dataset_size, generates indefinitely ===
    def __init__(
        self, tokenizer, game_config, max_moves, task_name, policy_model_name, length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.game_config = game_config
        self.max_moves = max_moves
        self.task_name = task_name
        self.policy_model_name = policy_model_name
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        print(
            f"Starting new iteration of IterablePuzzleDataset (indefinite generation)."
        )
        # Use itertools.count for an infinite index generator
        for i in itertools.count():
            yield generate_puzzle_datum(
                tokenizer=self.tokenizer,
                game_config=self.game_config,
                max_moves=self.max_moves,
                task_name=self.task_name,
                idx=i,
                policy_model_name=self.policy_model_name,
            )
        # This print message will never be reached in normal operation
        # print(f"Finished iteration of IterablePuzzleDataset.")

    def __len__(self):
        return self.length


# === MODIFIED: setup_puzzle_data now returns IterablePuzzleDataset ===
def setup_puzzle_data(
    tokenizer: AutoTokenizer,
    # === MODIFIED: Accept `env_cfg` instead of `env_configs` ===
    env_cfg: Dict[str, Any],
    policy_cfg: Dict[str, Any],
    task_name: str,
    length: int,
) -> Tuple[IterableDataset, IterableDataset | None, Dict, Dict]:
    """Sets up the iterable data generator and env map for the sliding puzzle task."""
    print("Setting up Sliding Puzzle iterable data and environment...")
    # === MODIFIED: Access env config directly via task_name ===
    env_config = env_cfg[task_name]

    # --- Instantiate Environment Actor --- #
    print(f"Instantiating environment actor for task '{task_name}'...")
    module_path, class_name = env_config["env_class"].rsplit(".", 1)
    try:
        EnvClass = getattr(__import__(module_path, fromlist=[class_name]), class_name)
    except ImportError as e:
        print(
            f"ERROR: Could not import environment class {env_config['env_class']}. Ensure it's in PYTHONPATH."
        )
        raise e
    env_actor = EnvClass.options(num_gpus=0).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env_actor}
    print(f"Environment actor '{task_name}' created.")

    # --- Instantiate Iterable Dataset --- #
    print(f"Creating IterablePuzzleDataset...")
    training_dataset = IterablePuzzleDataset(
        tokenizer=tokenizer,
        game_config=dict(env_config["cfg"]["game_config"]),
        max_moves=env_config["cfg"]["max_moves"],
        task_name=task_name,
        policy_model_name=policy_cfg.get("model_name", ""),
        length=length,
    )
    print("Iterable training dataset created.")

    validation_dataset = IterablePuzzleDataset(
        tokenizer=tokenizer,
        game_config=dict(env_config["cfg"]["game_config"]),
        max_moves=env_config["cfg"]["max_moves"],
        task_name=task_name,
        policy_model_name=policy_cfg.get("model_name", ""),
        length=256,
    )
    val_task_to_env = task_to_env

    return training_dataset, validation_dataset, task_to_env, val_task_to_env


# === Argparse function (Keep as is) ===
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


# === Main function (Follow math structure exactly) ===
def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    # Default config path
    if not args.config:
        # --- MODIFIED: Default config path ---
        default_config_path = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_sliding_puzzle.yaml"
        )
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(
                f"Default config file not found at {default_config_path}."
            )
        args.config = default_config_path
        print(f"No config provided, using default: {args.config}")

    # Load base config
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    # Apply overrides
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)  # Returns OmegaConf object
        print("Applied CLI overrides.")
    else:
        # Ensure config is OmegaConf object even without overrides for consistency
        config = OmegaConf.create(config)

    # Convert final config to dictionary for local use AFTER overrides
    # Use resolve=True to handle interpolations if any remain
    final_config_obj = config  # Keep as OmegaConf object for setup/utils
    final_config_dict = OmegaConf.to_container(config, resolve=True)
    print("----- Final Configuration ----- ")
    pprint.pprint(final_config_dict)
    print("--------------------------------- ")

    # Configure logging directory
    # Use dictionary access here
    logger_cfg = final_config_dict.get("logger", {})
    if "log_dir" in logger_cfg:
        try:
            log_dir = get_next_experiment_dir(logger_cfg["log_dir"])
            # Update dictionary for consistency, though setup might use OmegaConf obj
            final_config_dict["logger"]["log_dir"] = log_dir
            # Also update OmegaConf object if setup relies on it
            if isinstance(final_config_obj, OmegaConf):
                OmegaConf.update(
                    final_config_obj, "logger.log_dir", log_dir, merge=True
                )
            print(f"Logging directory set to: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"WARNING: Could not configure logging directory: {e}")
    else:
        print(
            "WARNING: 'logger.log_dir' not found in config, using default logging behavior."
        )

    # Configure checkpointing directory
    # Use dictionary access here
    checkpoint_cfg = final_config_dict.get("checkpointing", {})
    if checkpoint_cfg.get("enabled"):
        if "checkpoint_dir" in checkpoint_cfg:
            print(
                f"Checkpointing enabled. Directory: {checkpoint_cfg['checkpoint_dir']}"
            )
            os.makedirs(checkpoint_cfg["checkpoint_dir"], exist_ok=True)
        else:
            print(
                "WARNING: Checkpointing enabled but 'checkpointing.checkpoint_dir' not specified."
            )

    # Initialize Ray first
    # Pass the dictionary config to init_ray
    init_ray()

    # Setup tokenizer
    # === MODIFIED: Access tokenizer config from new structure ===
    policy_cfg = final_config_dict["policy"]
    tokenizer_cfg = policy_cfg.get(
        "tokenizer", policy_cfg
    )  # Use policy dict if 'tokenizer' key absent
    tokenizer = get_tokenizer(tokenizer_cfg)
    print("Tokenizer loaded.")

    # Configure generation config
    # === MODIFIED: Access generation config from new structure ===
    if "generation" in policy_cfg:
        policy_cfg["generation"] = configure_generation_config(
            policy_cfg["generation"], tokenizer
        )
        # Update the main config dict/obj if needed by setup
        final_config_dict["policy"]["generation"] = policy_cfg["generation"]
        if isinstance(final_config_obj, OmegaConf):
            OmegaConf.update(
                final_config_obj,
                "policy.generation",
                policy_cfg["generation"],
                merge=True,
            )
        print("Generation config configured.")
    else:
        print("WARNING: Policy generation config not found.")

    # Setup data & env map
    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_puzzle_data(
        tokenizer=tokenizer,
        env_cfg=final_config_dict["env"],  # Pass 'env' section
        policy_cfg=policy_cfg,
        task_name="sliding_puzzle_game",
        length=ds_length,
    )

    # Call the IMPORTED setup function
    print("Running main setup...")
    # Pass the dictionary config
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,  # Instantiated logger object
        checkpointer,  # Instantiated checkpointer object
        grpo_state,  # Initial state for training
        master_config,  # Processed MasterConfig object
        # Pass final_config_dict (plain dict) to setup
    ) = setup(final_config_dict, tokenizer, dataset, val_dataset)
    print("Main setup complete.")

    # Call grpo_train with the components returned by setup
    print("Starting GRPO training...")
    grpo_train(
        policy,
        policy_generation,
        dataloader,
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
    print("GRPO training finished.")

    # Final logging message
    output_dir = None
    if logger is not None and hasattr(logger, "log_dir") and logger.log_dir:
        output_dir = logger.log_dir
    elif "logger" in final_config_dict and "log_dir" in final_config_dict["logger"]:
        output_dir = final_config_dict["logger"]["log_dir"]
    if not output_dir:
        output_dir = final_config_dict.get(
            "output_dir", "./grpo_sliding_puzzle_outputs/unknown_run"
        )
    print(f"Checkpoints and logs should be in: {output_dir}")
    print("Script finished successfully.")


if __name__ == "__main__":
    main()
