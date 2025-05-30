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

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import remap_dataset_keys
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.environments.mc_environment import MCEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation.interfaces import configure_generation_config
import random 
import numpy as np
import torch
import json

from typing import Dict, Any
# import DatumSpec
from nemo_rl.data.interfaces import DatumSpec
# import LLMMessageLogType
from nemo_rl.data.llm_message_utils import LLMMessageLogType

def llama_nemotron_math_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["expected_answer"])
    extra_env_info = {"ground_truth": solution}

    message_log: LLMMessageLogType = []

    sys_message = {"role": "system", "content": task_data_spec.system_prompt}
    problem = task_data_spec.prompt.format(problem)
    user_message = {"role": "user", "content": problem}
    # user prompt
    assert task_data_spec.prompt is not None
    
    message_list = [sys_message,user_message]
    message = tokenizer.apply_chat_template(
        message_list,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

    
class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of differences from the mean

    def update(self, x):
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self):
        return np.sqrt(self.M2 / self.n) if self.n > 0 else float('nan')

def set_seed(seed: int):
    """Sets the seed for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def setup_data(tokenizer: AutoTokenizer, data_config, env_configs, skipped_problems = []):
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # load dataset

    base_dataset = load_dataset(data_config["dataset_name"])
    print (f"base_dataset: {base_dataset}")
    if data_config["dataset_key"] is not None:
        base_dataset = base_dataset[data_config["dataset_key"]]
    
    
    # remap problem and solution keys

    remapped_dataset = remap_dataset_keys(
        base_dataset,
        mapping_dict={
            data_config["problem_key"]: "problem",
            data_config["solution_key"]: "expected_answer",
        },
    )
    

    if data_config.get("shuffle", False):
        remapped_dataset = remapped_dataset.shuffle(seed=data_config.get("shuffle_seed", 42))

    task_type = data_config.get("task_type", "math")
    if task_type == "math":
        env = MathEnvironment.options(
            runtime_env={"py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE}
        ).remote(env_configs["math"])
    elif task_type == "mc":
        env = MCEnvironment.options(
            runtime_env={"py_executable": MCEnvironment.DEFAULT_PY_EXECUTABLE}
        ).remote(env_configs["mc"])
    else:
        raise ValueError(f"Invalid task type: {task_type}")

        
    data_processor = llama_nemotron_math_data_processor
    if data_config.get("planted_thinking_prompt", "none") != "none":
        data_processor = llama_nemotron_math_data_processor_with_planted_thinking(data_config["planted_thinking_prompt"], tokenizer)
        
    dataset = AllTaskProcessedDataset(
        dataset=remapped_dataset,
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    if len(skipped_problems) > 0:
        print (f"Skipping {len(skipped_problems)} problems")
        print (f"original length of dataset: {len(dataset)}")
        skipped_problems_list = list(skipped_problems)
        skipped_problems_list.sort()
        num_duplicates = 0
        for i in range(1, len(skipped_problems_list)):
            if skipped_problems_list[i] == skipped_problems_list[i-1]:
                num_duplicates += 1
        print (f"num_duplicates: {num_duplicates}")
        
        skipped_dataset = []
        for item in dataset:
            if item["message_log"][0]["content"] not in skipped_problems:
                skipped_dataset.append(item)
        dataset = skipped_dataset
        print (f"new length of dataset: {len(dataset)}")
    else:
        print ("No problems to skip")
    return dataset, env, tokenizer

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

    # problems that will be skipped in dataset
    original_problems = []
    if "debug" in config and "outfile" in config["debug"] and config["debug"]["outfile"] != "none":
        # check if file config["debug"]["outfile"] exists
        if os.path.exists(config["debug"]["outfile"]):
            # read all the lines in the file, convert them to json, and extract "original_problem" key
            with open(config["debug"]["outfile"], "r") as f:
                for line in f:
                    record = json.loads(line)
                    original_problems.append(record["original_problem"])
            print (f"Found {len(original_problems)} unique original problems, will skip them")
        outfile = open(config["debug"]["outfile"], "a")
    else:
        outfile = None

    # Setup data
    print ("Setting up data")
    (
        dataset,
        env,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"], skipped_problems=original_problems)

        
    # Setup
    print ("Setting up vllm generation")
    (
        vllm_generation,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    print (f"config['data']['planted_thinking_prompt']: {config['data']['planted_thinking_prompt']}")
    if config["data"].get("planted_thinking_prompt", "none") != "none":
        print (f"Using planted thinking prompt {config['data']['planted_thinking_prompt']}")

    online_stats_len = OnlineStats()
    
    def generation_callback(batch, env_return):
        for i in range(len(batch["message_log"])):
            message_log = batch["message_log"][i]
            correctness = env_return[0][i]
            ground_truth = env_return[1][i]
            idx = batch["idx"][i]
            content = ""
            assistant_tokens = []
            original_problem = ""
            for message in message_log:
                if message["role"] == "assistant":
                    content = message["content"]
                    assistant_tokens = tokenizer.encode(content)
                    online_stats_len.update(len(assistant_tokens))
                if message["role"] == "user":
                    original_problem = message["content"]
            if outfile is not None:
                print(f"Writing results of batch to file {config['debug']['outfile']}")
                record_to_write = {
                    "correctness": correctness,
                    "ground_truth": ground_truth,
                    "response_length": len(assistant_tokens),
                    "original_problem": original_problem,
                }
                if config["debug"]["output_generation"]:
                    record_to_write["generation"] = content
                outfile.write(json.dumps(record_to_write) + "\n") 
                outfile.flush()
                

    # Run evaluation
    print ("Starting evaluation")
    run_env_eval(
        vllm_generation,
        dataloader,
        env,
        master_config,
        generation_callback=generation_callback
    )

    print (f"Mean length: {online_stats_len.get_mean()}")
    print (f"Std length: {online_stats_len.get_std()}")


if __name__ == "__main__":
    main()
