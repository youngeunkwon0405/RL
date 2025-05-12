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

import os
from typing import Tuple, TypedDict

import ray
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_reinforcer.data import MathDataConfig
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_reinforcer.data.llm_message_utils import get_keys_from_message_log
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_reinforcer.environments.math_environment import MathEnvConfig
from nemo_reinforcer.models.generation.interfaces import GenerationConfig
from nemo_reinforcer.models.generation.vllm import VllmGeneration

from nemo_reinforcer.data.llm_message_utils import (
    get_keys_from_message_log,
    batched_message_log_to_flat_message,
)
from nemo_reinforcer.models.generation.interfaces import (
    GenerationDatumSpec,
)

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    FlatMessagesType,
)
# ===============================================================================
# Configuration
# ===============================================================================


class MasterConfig(TypedDict):
    generate: GenerationConfig
    data: MathDataConfig
    env: MathEnvConfig
    cluster: ClusterConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> Tuple[
    VllmGeneration,
    DataLoader,
    MasterConfig,
    VllmGeneration,
]:
    """Set up components for model evaluation.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, and config.
    """
    # Extract individual configs for easier access
    generation_config = master_config["generation"]
    target_generation_config = master_config["target_generation"]
    cluster_config = master_config["cluster"]

    # ==========================
    #           Data
    # ==========================
    if generation_config["num_prompts_per_step"] == -1:
        generation_config["num_prompts_per_step"] = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=generation_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    print(f"  ✓ Evaluation dataset loaded with {len(dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]//2]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"]//2,
        max_colocated_worker_groups=1,
    )

    cluster2 = RayVirtualCluster(
        name="eval_cluster_target",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]//2]
                                * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"]//2,
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    # check backend
    backend = generation_config["backend"]
    assert backend == "vllm", "Only vLLM backend is supported for evaluation"

    # initialize vllm generation
    vllm_generation = VllmGeneration(cluster=cluster, config=generation_config)


    # target_generation_config["max_new_tokens"] = 2 * target_generation_config["max_new_tokens"]
    # target_generation_config["vllm_cfg"]["max_model_len"] = 2 * target_generation_config["vllm_cfg"]["max_model_len"]
    target_vllm_generation = VllmGeneration(cluster=cluster2, config=target_generation_config,name_prefix='vllm_target')
    print(
        f"  ✓ Using vLLM backend for generation with {generation_config['model_name']}"
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        vllm_generation,
        dataloader,
        master_config,
        target_vllm_generation,
    )


# ===============================================================================
# Evaluation
# ===============================================================================


def run_env_eval(vllm_generation, dataloader, env, master_config,target_vllm_generation,target_tokenizer):
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        vllm_generation: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        env: Environment that scores responses.
        master_config: Configuration settings.
    """
    # Run evaluation loop
    score, count = 0.0, 0
    for batch in dataloader:
        # get input prompt from message_log
        prompts = []
        for message_log in batch["message_log"]:
            content = [message["content"] for message in message_log]
            content = "\n".join(content)
            prompts.append(content)
        # generate by vllm
        inputs = BatchedDataDict({"prompts": prompts})
        outputs = vllm_generation.generate_text(inputs)["texts"]

        # import pdb;
        # pdb.set_trace()



        user_messages = []
        target_model_inputs_len = []
        for input_sample, generated_text_sample in zip(inputs['prompts'], outputs):
            # TODO this support only the first turn
            problem = input_sample
            thought = generated_text_sample
            thought = problem + ' <think> '+thought+' </think>'
            # import pdb;
            # pdb.set_trace()

            user_message = target_tokenizer.process_data(problem, thought)
            user_messages.append(user_message)
        #
        active_flat_messages: FlatMessagesType
        active_flat_messages, active_input_lengths = (
            batched_message_log_to_flat_message(
                user_messages,
                pad_value_dict={"token_ids": target_tokenizer.pad_token_id},
            )
        )

        # Extract input_ids and lengths from the flat messages
        active_input_ids = active_flat_messages["token_ids"]
        active_stop_strings = [None] * len(active_input_lengths)

        target_generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": active_input_ids,
                "input_lengths": active_input_lengths,
                "stop_strings": active_stop_strings,
            }
        )

        # import pdb;
        # pdb.set_trace()
        # target_generation_outputs = policy_generation.generate(generation_input_data, greedy=greedy)
        target_generation_outputs = target_vllm_generation.generate(target_generation_input_data, greedy=True)



        # Extract generated tokens
        generated_ids = []
        unpadded_sequence_lengths = target_generation_outputs["unpadded_sequence_lengths"]
        for output_ids, input_length, total_length in zip(
                target_generation_outputs["output_ids"], active_input_lengths, unpadded_sequence_lengths
        ):
            generated_ids.append(output_ids[input_length:total_length])

        # import pdb;
        # pdb.set_trace()

        # import pdb;
        # pdb.set_trace()

        outputs = target_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # import pdb;
        # pdb.set_trace()

        # append to message_log
        for idx, output in enumerate(outputs):
            batch["message_log"][idx].append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

        # evaluate generations with the environment
        to_env = [
            get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
            for i in range(len(batch["message_log"]))
        ]
        env_return = ray.get(env.step.remote(to_env, batch["extra_env_info"]))

        score += env_return.rewards.sum().item()
        count += len(env_return.rewards)

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_generation.shutdown()

    # Print results
    dataset_name = os.path.basename(master_config["data"]["dataset_name"])
    model_name = os.path.basename(master_config["generation"]["model_name"])
    average_score = score / count

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"score={average_score:.2f} ({score}/{count})")
    print("=" * 60 + "\n")
