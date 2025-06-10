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
import itertools
import os
import pprint
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import jsonlines
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.llm_judge_async_environment import LLMJudgeAsyncEnvironment
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.environments.ifeval_environment import IFEvalEnvironment
from nemo_rl.environments.reasoning_gym_env import (
    ReasoningGymEnv,
    ReasoningGymGameLogic,
    ReasoningGymMetadata,
)
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


@dataclass
class JsonlinesDataset:
    jsonl_path: str
    seed: int
    tokenizer: AutoTokenizer
    max_seq_length: int
    filter_long_samples: bool = False

    def __post_init__(self):
        self.data = self._load_data()

        idx_to_ignore = set()
        if self.filter_long_samples:
            for i, item in enumerate(self):
                if item["length"] > self.max_seq_length:
                    idx_to_ignore.add(i)
            print(f"found {len(idx_to_ignore)} long samples to ignore on dataset init")

        self.data = [item for i, item in enumerate(self.data) if i not in idx_to_ignore]

    def _load_data(self):
        with jsonlines.open(self.jsonl_path, "r") as reader:
            data = [line for line in reader]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> DatumSpec:
        data = self.data[idx]
        # support single turn for now
        assert len(data["messages"]) == 1
        single_message = data["messages"][0]

        message_log = []

        # this will also contain system prompt
        user_message = {"role": "user"}

        for m in single_message:
            # it's actually taking only the last user message's metadata
            if m["role"] == "user":
                # need to be deepcopy to avoid overwriting the original metadata
                extra_env_info = deepcopy(m["metadata"])

        message = self.tokenizer.apply_chat_template(
            single_message,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        user_message["token_ids"] = self.tokenizer.apply_chat_template(
            single_message,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        user_message["content"] = message
        message_log.append(user_message)

        length = sum(len(m["token_ids"]) for m in message_log)

        output = {
            "message_log": message_log,
            "length": length,
            "extra_env_info": extra_env_info,
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": data["task_name"],
            "dataset": data["dataset"],
        }

        return output


def generate_reasoning_gym_datum(
    tokenizer,
    game_config: dict[str, Any],
    task_name: str,
    idx: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generate a single reasoning-gym puzzle datum (prompt and metadata)."""
    
    game_state = ReasoningGymGameLogic.generate(game_config)
    puzzle_question = ReasoningGymGameLogic.init(game_state)
    
    prompt_instructions = (
        f"{puzzle_question}\n\n"
        f"Think carefully about this problem and provide your final answer.\n"
        f"After reasoning, output your answer on a new line using this format:\n"
        f"<answer>your_final_answer</answer>\n"
    )
    
    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_instructions}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    
    # for length calculation
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
    
    metadata = ReasoningGymMetadata(
        puzzle_entry=game_state,
        dataset_name=game_state["dataset_name"]
    )
    
    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
        "stop_strings": ["</answer>"],
    }
    
    return datum


class IterableReasoningGymDataset(IterableDataset):
    """An IterableDataset that generates reasoning-gym puzzle data indefinitely."""

    def __init__(
        self, tokenizer, game_config, task_name, add_system_prompt, length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.game_config = game_config
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        if "tasks" in self.game_config:
            num_tasks = len(self.game_config['tasks'])
            dataset_info = f"{num_tasks} tasks: {', '.join([task['dataset_name'] for task in self.game_config['tasks']])}"
        else:
            dataset_info = "unknown configuration"
            
        print(f"Starting IterableReasoningGymDataset for {dataset_info} (indefinite generation).")
        for i in itertools.count():
            yield generate_reasoning_gym_datum(
                tokenizer=self.tokenizer,
                game_config=self.game_config,
                task_name=self.task_name,
                idx=i,
                add_system_prompt=self.add_system_prompt,
            )

    def __len__(self):
        return self.length


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, env_configs):
    print("\n▶ Setting up data...")

    # if reasoning-gym is enabled, use it instead of jsonlines
    if "reasoning_gym_task" in env_configs and env_configs["reasoning_gym_task"]["enable"]:
        print("Reasoning Gym enabled - using IterableReasoningGymDataset")
        
        rg_config = env_configs["reasoning_gym_task"]
        if "tasks" in rg_config:
            game_config = {"tasks": rg_config["tasks"]}
        else:
            raise ValueError("reasoning_gym_task config must specify 'tasks'")
        
        train_ds = IterableReasoningGymDataset(
            tokenizer=tokenizer,
            game_config=game_config,
            task_name="reasoning_gym_task",
            add_system_prompt=data_config.get("add_system_prompt", False),
            length=10000, # TODO: not sure how this is used, but following sliding puzzle example...
        )
        
        val_ds = IterableReasoningGymDataset(
            tokenizer=tokenizer,
            game_config=game_config,
            task_name="reasoning_gym_task", 
            add_system_prompt=data_config.get("add_system_prompt", False),
            length=1000,
        )
    else:
        train_ds = JsonlinesDataset(
            data_config["train"]["jsonl_path"],
            data_config["train"]["seed"],
            tokenizer,
            max_seq_length=data_config["max_input_seq_length"],
            filter_long_samples=data_config["train"]["filter_long_samples"],
        )
        val_ds = JsonlinesDataset(
            data_config["val"]["jsonl_path"],
            data_config["val"]["seed"],
            tokenizer,
            max_seq_length=data_config["max_input_seq_length"],
            filter_long_samples=data_config["val"]["filter_long_samples"],
        )

    task_to_env = {}

    if "math" in env_configs and env_configs["math"]["enable"]:
        math_env = MathEnvironment.options(
            runtime_env={
                "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(
                    os.environ
                ),  # Pass thru all user environment variables
            }
        ).remote(env_configs["math"])
        task_to_env["math"] = math_env
    
    if "ifeval" in env_configs and env_configs["ifeval"]["enable"]:
        ifeval_env = IFEvalEnvironment.options(
            runtime_env={
                "py_executable": IFEvalEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs["ifeval"])
        task_to_env["ifeval"] = ifeval_env
        
    if "reasoning_gym_task" in env_configs and env_configs["reasoning_gym_task"]["enable"]:
        rg_config = env_configs["reasoning_gym_task"]
        if "tasks" in rg_config:
            game_config = {"tasks": rg_config["tasks"]}
        else:
            raise ValueError("reasoning_gym_task config must specify 'tasks'")
            
        reasoning_gym_env = ReasoningGymEnv.options(num_gpus=0).remote(cfg=game_config)
        task_to_env["reasoning_gym_task"] = reasoning_gym_env
        
    if "llm_judge_async" in env_configs and env_configs["llm_judge_async"]["enable"]:
        # Extract max_concurrency from config, default to 16 if not specified
        max_concurrency = env_configs["llm_judge_async"].get("max_concurrency", 16)

        llm_judge_async_env = LLMJudgeAsyncEnvironment.options(
            max_concurrency=max_concurrency,
            runtime_env={
                "py_executable": LLMJudgeAsyncEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs["llm_judge_async"])
        task_to_env["llm_judge"] = llm_judge_async_env

    return train_ds, val_ds, task_to_env, task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_1B.yaml")

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
