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

from typing import Tuple, TypedDict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.hf_policy import HfPolicy

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
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    policy_config = master_config["policy"]

    print("\n▶ Setting up model...")
    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=False,
        init_reference_model=False,
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        dataloader,
        master_config,
    )


# ===============================================================================
# Evaluation
# ===============================================================================


def run_sft_eval(policy, dataloader, master_config, tokenizer):
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        policy: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        master_config: Configuration settings.
    """
    for batch in dataloader:
        ## add loss mask based on role to every message
        add_loss_mask_to_message_log(
            batch["message_log"],
            roles_to_train_on=["assistant"],
        )

        cat_and_padded, input_lengths = batched_message_log_to_flat_message(
            batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=master_config["policy"][
                "make_sequence_length_divisible_by"
            ],
        )

        data: BatchedDataDict = BatchedDataDict(
            {
                "input_ids": cat_and_padded["token_ids"],
                "input_lengths": input_lengths,
                "token_mask": cat_and_padded["token_loss_mask"],
                "sample_mask": batch["loss_multiplier"],
            }
        )

        ## get model predictions
        logprobs = policy.get_logprobs(
            data,
            eval_mode=True,
            gbs=len(data["input_ids"]),  ## TODO: make configrable
            mbs=4,  ## TODO: make configurable
        )

        ## score predictions
        # Get the predicted tokens (argmax of logprobs)
        predicted_tokens = torch.argmax(logprobs["logprobs"], dim=-1)

        # Get the true tokens from input_ids
        true_tokens = data["input_ids"][:, 1:]

        # Get the token mask to identify which positions to evaluate
        token_mask = data["token_mask"][:, 1:]

        # Calculate accuracy for each sequence
        correct_predictions = (predicted_tokens == true_tokens) * token_mask
        sequence_lengths = token_mask.sum(dim=-1)
        sequence_accuracies = (
            correct_predictions.sum(dim=-1) / sequence_lengths
        ) == 1.0

        # Calculate overall accuracy
        overall_accuracy = sequence_accuracies.mean().item()

        # Print results
        print(f"Batch accuracy: {overall_accuracy:.4f}")
        print(f"Per-sequence accuracies: {sequence_accuracies.tolist()}")
