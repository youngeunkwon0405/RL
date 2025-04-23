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
import warnings
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from nemo_reinforcer.algorithms.loss_functions import NLLLoss
from nemo_reinforcer.algorithms.utils import (
    configure_logger,
    extract_individual_configs,
    log_metrics,
    reduce_microbatch_metrics,
    save_checkpoint,
    setup_checkpointer,
    setup_dataloaders,
    setup_policy,
    should_checkpoint,
    should_validate,
    validate_checkpointing_config,
    set_seed,
)
from nemo_reinforcer.data import DataConfig
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_reinforcer.data.interfaces import TaskDataSpec
from nemo_reinforcer.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.utils.checkpoint import CheckpointManager, CheckpointingConfig
from nemo_reinforcer.utils.logger import Logger, LoggerConfig
from nemo_reinforcer.utils.timer import Timer


class SFTConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int


class SFTSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float


def _default_sft_save_state() -> SFTSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> Tuple[
    HfPolicy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    NLLLoss,
    Logger,
    CheckpointManager,
    SFTSaveState,
    MasterConfig,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, sft_save_state, master_config
    """
    set_seed(master_config["sft"]["seed"])

    # Extract individual configs for easier access
    (
        policy_config,
        data_config,
        logger_config,
        cluster_config,
        checkpointing_config,
    ) = extract_individual_configs(master_config)
    sft_config = master_config["sft"]

    # ==========================
    #         Logger
    # ==========================
    logger = configure_logger(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer, last_checkpoint_path, sft_save_state = setup_checkpointer(
        checkpointing_config, _default_sft_save_state()
    )

    # verify that checkpoint period is a multiple of validation period
    validate_checkpointing_config(checkpointing_config, sft_config)

    # ==========================
    #           Data
    # ==========================
    train_dataloader, val_dataloader, train_dataloader_kwargs = setup_dataloaders(
        train_dataset,
        val_dataset,
        rl_collate_fn,
        sft_config,
        policy_config,
        last_checkpoint_path,
        return_train_dl_kwargs=True,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="sft_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training
    # ==========================
    policy = setup_policy(
        cluster,
        policy_config,
        tokenizer,
        last_checkpoint_path,
        init_reference_model=False,
    )

    # initialize loss function
    loss_fn = NLLLoss()

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    )


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    sft_task_spec: TaskDataSpec,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    logger: Logger,
):
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        val_metrics = {"val_loss": 0.0}
        num_valid_batches = 0

        policy.prepare_for_training()
        for batch_idx, val_batch in enumerate(val_dataloader):
            ## add loss mask based on role to every message
            add_loss_mask_to_message_log(
                val_batch["message_log"],
                roles_to_train_on=["assistant"],
            )

            cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                val_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=master_config["policy"][
                    "make_sequence_length_divisible_by"
                ],
            )

            val_data: BatchedDataDict = BatchedDataDict(
                {
                    "input_ids": cat_and_padded["token_ids"],
                    "input_lengths": input_lengths,
                    "token_mask": cat_and_padded["token_loss_mask"],
                    "sample_mask": val_batch["loss_multiplier"],
                }
            )

            ## just run model fwd
            val_results = policy.train(
                val_data,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size,
                mbs=val_mbs,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                val_metrics["val_loss"] += float(val_results["loss"])
                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if num_valid_batches > 0:
            val_metrics["val_loss"] /= num_valid_batches
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    if num_valid_batches > 0:
        log_to_console = {
            "loss": val_metrics["val_loss"],
        }
        log_metrics(log_to_console, val_metrics, timer, step, logger, is_val=True)

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics


def sft_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    sft_task_spec,
    checkpointer,
    sft_save_state,
):
    # Run basic sft training
    timer = Timer()

    current_epoch = sft_save_state["epoch"]
    current_step = sft_save_state["step"]
    total_steps = sft_save_state["total_steps"]

    sft_config = master_config["sft"]
    # Validation configuration
    val_period = sft_config["val_period"]
    val_at_start = sft_config["val_at_start"]
    max_num_epochs = sft_config["max_num_epochs"]

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        val_metrics = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            sft_task_spec=sft_task_spec,
            val_batches=sft_config["val_batches"],
            val_batch_size=sft_config["val_global_batch_size"],
            val_mbs=sft_config["val_micro_batch_size"],
            logger=logger,
        )

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["sft"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['sft']['max_num_steps'])} {'=' * 25}"
            )

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
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

                    train_data: BatchedDataDict = BatchedDataDict(
                        {
                            "input_ids": cat_and_padded["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": cat_and_padded["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )

                print("▶ Taking a training step...")
                train_results = policy.train(train_data, loss_fn)

                # Run validation if it's a validation step
                if should_validate(val_period, total_steps):
                    val_metrics = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        sft_task_spec=sft_task_spec,
                        val_batches=sft_config["val_batches"],
                        val_batch_size=sft_config["val_global_batch_size"],
                        val_mbs=sft_config["val_micro_batch_size"],
                        logger=logger,
                    )

                ## Checkpointing
                if should_checkpoint(master_config["checkpointing"], total_steps):
                    sft_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    sft_save_state["total_steps"] = total_steps + 1
                    sft_save_state["epoch"] = current_epoch
                    sft_save_state["val_loss"] = val_metrics["val_loss"]
                    save_checkpoint(
                        checkpointer,
                        master_config,
                        sft_save_state,
                        total_steps,
                        train_dataloader,
                        policy,
                        timer,
                    )

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
            }
            metrics.update(reduce_microbatch_metrics(train_results["all_mb_metrics"]))
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            log_to_console = {
                "loss": float(metrics["loss"]),
            }
            log_metrics(
                log_to_console, metrics, timer, total_steps, logger, is_val=False
            )

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps >= master_config["sft"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
