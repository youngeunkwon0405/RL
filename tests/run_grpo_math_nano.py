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

# Example:
# uv run python -m pdb tests/run_grpo_math_nano.py --config tests/configs/grpo_math_8B_nano.yaml 
import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Dict

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment_for_L1 import MathEnvironmentForL1
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
import torch
import json 
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from nemo_rl.data.llm_message_utils import remap_dataset_keys
import pdb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
# A Dataset that uses information from L1 expected response length
# -------------------------------------------------------------------------------
def strip_templating_from_problem_50(problem):
    assert problem.find("detailed thinking off") == -1, "Problem contains detailed thinking off"
    prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\ndetailed thinking on<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if problem.startswith(prefix):
        problem = problem[len(prefix):]
    else:
        assert False, "Problem does not start with prefix"
    if problem.endswith(suffix):
        problem = problem[:-len(suffix)]
    else:
        assert False, "Problem does not end with suffix"
    #import pdb; p = pdb.Pdb(); p.prompt="breakpoint-strip_templating_from_problem:"; p.set_trace()
    problem = problem.replace(' ', '')
    problem = problem.replace('\n', '')
    cut = min(50, len(problem))
    return problem[:cut]

class L1ExpectedAnswerLengthDataset(Dataset):
    def __init__(self, dataset, metadata):
        self.dataset = dataset
        #self.sorted_keys = sorted(self.metadata.keys())
        # the metadata keys are the original problems.  They may contain some templating.  We need to remove the temoplating
        # to get the original problem
        self.metadata = dict([(strip_templating_from_problem_50(k), v) for k, v in metadata.items()])
        count = 0
        self.found_idxs = []
        for idx,problem in enumerate(dataset['problem']):
            p = problem.replace(' ', '').replace('\n', '')
            cut = min(50, len(p))
            p = p[:cut]
            if p not in self.metadata:
                count += 1
            else:
                self.found_idxs.append(idx)
            #assert p in self.metadata, f"Problem {p} not in metadata"
        print (f"problems not found in metadata: {count}")
        #import pdb; pdb.set_trace()
    def __len__(self):
        return len(self.found_idxs)
    
    def __getitem__(self, idx):
        #datum = self.dataset[self.sorted_keys[idx]]
        idx = self.found_idxs[idx]
        datum = self.dataset[idx]
        problem_key = datum['problem'].replace(' ', '').replace('\n', '')
        problem_key = problem_key[:min(50, len(problem_key))]
        datum['L1_metadata'] = self.metadata[problem_key]
        #import pdb; pdb.set_trace()
        # add the expected answer length to the datum
        #datum['L1_metadata'] = self.metadata[self.sorted_keys[idx]]
        return datum

    # need an iterator?
    def __iter__(self):
        assert False, "Not implemented"

# ===============================================================================
#                             Llama nemotron Math Data Processor
# ===============================================================================

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
    if 'L1_metadata' in datum_dict:
        extra_env_info['L1_metadata'] = datum_dict['L1_metadata']

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

def openinstructmath2_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Math Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}


    message_log: LLMMessageLogType = []
    system_message = {
        "role": "system",
        "content": task_data_spec.system_prompt,
    }
    user_message = {
        "role": "user",
        "content": task_data_spec.prompt.format(problem),
    }
    message = tokenizer.apply_chat_template(
        [system_message,user_message],
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
        "task_name": datum_dict["task_name"],
    }
    return output

def add_task_name_to_dict(datum_dict):
    datum_dict["task_name"] = "math"
    return datum_dict

def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, env_configs):
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    data = load_dataset(data_config["dataset_name"])
    print (f"data: {data}")
    if data_config["dataset_key"] is not None:
        data = data[data_config["dataset_key"]]
    data = remap_dataset_keys(
        data,
        mapping_dict={
            data_config["problem_key"]: "problem",
            data_config["solution_key"]: "expected_answer",
        },
    )
    data = data.map(add_task_name_to_dict)

    val_data = load_dataset(data_config["val_dataset_name"])
    print (f"val_data: {val_data}")
    if data_config["val_dataset_key"] is not None:
        val_data = val_data[data_config["val_dataset_key"]]
    val_data = remap_dataset_keys(
        val_data,
        mapping_dict={
            data_config["val_problem_key"]: "problem",
            data_config["val_solution_key"]: "expected_answer",
        },
    )
    val_data = val_data.map(add_task_name_to_dict)

    if "L1_loss" in env_configs['math'] and env_configs['math']['L1_loss']['enabled']:
        baseline_answer_length_path = env_configs['math']["L1_loss"]["baseline_answer_length_path"]
        shrink_factor_for_incorrect_answer = env_configs['math']["L1_loss"]["shrink_factor_for_incorrect_answer"]    
        lower_bound_factor = env_configs['math']["L1_loss"]["lower_bound_factor"]
        upper_bound_factor = env_configs['math']["L1_loss"]["upper_bound_factor"]
        penalty_factor = env_configs['math']["L1_loss"]["penalty_factor"]
        L1_metadata = {}
        with open(baseline_answer_length_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record["correctness"]["content"] == "Environment: correct":
                    baseline_answer_length = record['response_length']
                else:
                    assert record["correctness"]["content"] == "Environment: incorrect"
                    baseline_answer_length = record['response_length'] / shrink_factor_for_incorrect_answer
                L1_metadata[record['original_problem']] = {
                    "baseline_answer_length": baseline_answer_length,
                    "penalty_factor": penalty_factor,
                    "lower_bound_factor": lower_bound_factor,
                    "upper_bound_factor": upper_bound_factor,
                }
                
        # this will create a new daset resttricted to only those indices that have an expected answer length
        # and will augment the dataset with the expected answer length
        #import pdb; pdb.set_trace()
        data = L1ExpectedAnswerLengthDataset(data, L1_metadata)
        #print (dataset[0])
        #import pdb; pdb.set_trace()


    # Load OpenMathInstruct2Dataset using reinforcer datasets
    #if data_config["dataset_name"] == "OpenMathInstruct-2":
    #    print(f"Loading nvidia/OpenMathInstruct2Dataset for training and validation")
    #    data = OpenMathInstruct2Dataset()
    #else:
    #    raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    task_data_processors = defaultdict(
        lambda: (math_task_spec, llama_nemotron_math_data_processor)
    )
    task_data_processors["math"] = (math_task_spec, llama_nemotron_math_data_processor)
    math_env = MathEnvironmentForL1.options(
        runtime_env={
            "py_executable": MathEnvironmentForL1.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"], tokenizer)
    val_math_env = MathEnvironment.options(
        runtime_env={
            "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"])

    dataset = AllTaskProcessedDataset(
        data, #.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_data,
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    #import pdb; p = pdb.Pdb(); p.prompt='breakpoint-run_grpo_dataset creation:)';  p.set_trace()
        
    task_to_env = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    
    val_task_to_env = defaultdict(lambda: val_math_env)
    val_task_to_env["math"] = val_math_env
    
    return dataset, val_dataset, task_to_env, val_task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_8B.yaml"
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

    c=0
    for batch in dataloader:
       c+=1
    print (f"c: {c}")
    #import pdb; p=pdb.Pdb(); p.prompt='breakpoint-main :)'; p.set_trace()
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
