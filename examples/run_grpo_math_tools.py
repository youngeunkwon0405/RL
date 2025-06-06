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
from typing import Any, Optional, cast

import torch
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.hf_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)

from nemo_rl.distributed.virtual_cluster import init_ray, PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_tools_environment import (
    MathToolsEnvironment,
    MathToolsMetadata,
    create_working_directory,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GRPO training with math tools environment")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    args, overrides = parser.parse_known_args()

    return args, overrides


TokenizerType = PreTrainedTokenizerBase


def math_tools_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for the Math Tools Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    ground_truth = user_message[1]["content"]
    
    working_dir = create_working_directory()
    
    extra_env_info = MathToolsMetadata(
        problem_id=datum_dict.get("problem_id", f"problem_{idx}"),
        problem_text=problem,
        ground_truth=ground_truth,
        working_dir=working_dir,
        current_turn=0,
        max_turns=10,  # Will be overridden by config
        files_created=[],
        bash_history=[],
        tool_calls_count={},
    )

    message_log: LLMMessageLogType = []
    
    # Add system prompt if specified
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(sys, return_tensors="pt")["input_ids"][0]
        message_log.append(sys_prompt)
    
    # Format user message with prompt template
    if task_data_spec.prompt:
        formatted_problem = task_data_spec.prompt.format(problem=problem)
    else:
        formatted_problem = problem
    
    user_msg = {"role": "user", "content": formatted_problem}
    message = tokenizer.apply_chat_template(
        [user_msg],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_msg["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_msg["content"] = message
    message_log.append(user_msg)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # Truncate if too long
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "math"),
                                                 "stop_strings": ["</answer>", "</tool_call>", "<|im_end|>"],
    }
    return output


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up math tools data...")
    
    # Create task specification
    math_tools_task_spec = TaskDataSpec(
        task_name="math_tools",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )

    # Load dataset
    if data_config["dataset_name"] == "OpenMathInstruct-2":
        print("Loading nvidia/OpenMathInstruct2Dataset for training and validation")
        data: Any = OpenMathInstruct2Dataset()
    elif data_config["dataset_name"] == "DeepScaler":
        print("Loading agentica-org/DeepScaleR-Preview-Dataset for training and validation")
        data: Any = DeepScalerDataset()
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    # Set up task data processors
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = {
        "math": (math_tools_task_spec, math_tools_data_processor)
    }

    # Create math tools environment
    math_tools_env = MathToolsEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": PY_EXECUTABLES.SYSTEM,
            "env_vars": dict(os.environ), 
        }
    ).remote(env_configs["math_tools"])
    
    # Create datasets
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_tools_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds.get("validation") is not None:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            math_tools_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

    task_to_env: dict[str, EnvironmentInterface] = {
        "math": math_tools_env
    }
    
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_tools.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

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


if __name__ == "__main__":
    main() 

