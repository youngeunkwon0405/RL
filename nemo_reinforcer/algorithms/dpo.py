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
from collections import defaultdict
from functools import partial
from pathlib import Path
from transformers import AutoTokenizer
from typing import Optional, Tuple, TypedDict
from tqdm import tqdm

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from nemo_reinforcer.algorithms.loss_functions import DPOLossFn
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
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, dpo_collate_fn
from nemo_reinforcer.data.interfaces import TaskDataSpec
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.utils.checkpoint import CheckpointManager, CheckpointingConfig
from nemo_reinforcer.utils.logger import Logger, LoggerConfig
from nemo_reinforcer.utils.timer import Timer


class DPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    ## TODO(@ashors) support other loss functions
    ## https://github.com/NVIDIA/reinforcer/issues/193
    # preference_loss: str
    # gt_reward_scale: float
    preference_loss_weight: float
    sft_loss_weight: float


class DPOSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float


def _default_dpo_save_state() -> DPOSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    dpo: DPOConfig
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
    DPOLossFn,
    Logger,
    CheckpointManager,
    DPOSaveState,
    MasterConfig,
]:
    """Main entry point for running DPO algorithm.

    Returns:
        Tuple of policy, cluster, train_dataloader, train_dataloader_kwargs, val_dataloader, loss_fn, logger, checkpointer, dpo_save_state, master_config
    """
    set_seed(master_config["dpo"]["seed"])

    # Extract individual configs for easier access
    (
        policy_config,
        data_config,
        logger_config,
        cluster_config,
        checkpointing_config,
    ) = extract_individual_configs(master_config)
    dpo_config = master_config["dpo"]

    logger = configure_logger(master_config)

    checkpointer, last_checkpoint_path, dpo_save_state = setup_checkpointer(
        checkpointing_config, _default_dpo_save_state()
    )

    # verify that checkpoint period is a multiple of validation period
    validate_checkpointing_config(checkpointing_config, dpo_config)

    # ==========================
    #           Data
    # ==========================
    ## TODO(@ashors) reduce boilerplate and move reused code into utils
    dpo_collate = partial(
        dpo_collate_fn,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=policy_config[
            "make_sequence_length_divisible_by"
        ],
    )
    train_dataloader, val_dataloader, train_dataloader_kwargs = setup_dataloaders(
        train_dataset,
        val_dataset,
        dpo_collate,
        dpo_config,
        policy_config,
        last_checkpoint_path,
        return_train_dl_kwargs=True,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="dpo_cluster",
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
    policy = setup_policy(cluster, policy_config, tokenizer, last_checkpoint_path)

    loss_fn = DPOLossFn(dpo_config)

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
        dpo_save_state,
        master_config,
    )


def add_ref_logprobs_to_data(dataloader, policy, master_config):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)

            ## append ref policy logprobs to batch
            logprobs = policy.get_reference_policy_logprobs(
                batch,
                micro_batch_size=master_config["policy"]["train_micro_batch_size"] * 2,
            )["reference_logprobs"]
            ## want logprobs for batch to correspond to the log probabilities of the next tokens
            ## so we roll the logprobs to the left by one
            batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

            yield batch

        except StopIteration:
            break


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

        val_metrics = defaultdict(lambda: 0.0)
        num_valid_batches = 0
        for batch_idx, val_batch in enumerate(
            add_ref_logprobs_to_data(val_dataloader, policy, master_config)
        ):
            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size * 2,
                mbs=val_mbs * 2,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )

            else:
                for k, v in val_results["all_mb_metrics"].items():
                    val_metrics[k] += np.mean(v).item()
                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        for k, v in val_metrics.items():
            if k == "num_valid_samples":
                continue
            val_metrics[k] /= num_valid_batches

        # Calculate validation metrics
        policy.prepare_for_training()

    if len(val_metrics) == 0:
        warnings.warn(
            "No validation metrics were collected."
            " This is likely because there were no valid samples in the validation set."
        )

    else:
        log_to_console = {
            "loss": val_metrics["loss"],
        }
        log_metrics(log_to_console, val_metrics, timer, step, logger, is_val=True)

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics


def dpo_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    dpo_save_state,
):
    # Run dpo training
    timer = Timer()

    if dpo_save_state is None:
        dpo_save_state = _default_dpo_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = dpo_save_state["epoch"]
        current_step = dpo_save_state["step"]
        total_steps = dpo_save_state["total_steps"]

    dpo_config = master_config["dpo"]
    # Validation configuration
    val_period = dpo_config["val_period"]
    val_at_start = dpo_config["val_at_start"]
    max_num_epochs = dpo_config["max_num_epochs"]

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
            val_batches=dpo_config["val_batches"],
            val_batch_size=dpo_config["val_global_batch_size"],
            val_mbs=dpo_config["val_micro_batch_size"],
            logger=logger,
        )

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["dpo"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in add_ref_logprobs_to_data(train_dataloader, policy, master_config):
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['dpo']['max_num_steps'])} {'=' * 25}"
            )

            with timer.time("total_step_time"):
                print("▶ Taking a training step...")
                train_results = policy.train(
                    batch,
                    loss_fn,
                    eval_mode=False,
                    ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                    ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                    gbs=master_config["policy"]["train_global_batch_size"] * 2,
                    mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                )

                # Run validation if it's a validation step
                if should_validate(val_period, total_steps):
                    val_metrics = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=dpo_config["val_batches"],
                        val_batch_size=dpo_config["val_global_batch_size"],
                        val_mbs=dpo_config["val_micro_batch_size"],
                        logger=logger,
                    )

                ## Checkpointing
                if should_checkpoint(master_config["checkpointing"], total_steps):
                    dpo_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    dpo_save_state["total_steps"] = total_steps + 1
                    dpo_save_state["epoch"] = current_epoch
                    dpo_save_state["val_loss"] = val_metrics["loss"]
                    save_checkpoint(
                        checkpointer,
                        master_config,
                        dpo_save_state,
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

            log_to_console = {
                "loss": metrics["loss"],
            }
            log_metrics(log_to_console, metrics, timer, total_steps, logger)

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps >= master_config["dpo"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
