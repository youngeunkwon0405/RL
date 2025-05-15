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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import NLLLoss
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.data.quack_data_utils import (
    setup_data,
    convert_actor_rollouts_to_buffer_items,
    ReplayBuffer,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.timer import Timer
# from nemo_rl.metrics.metrics_utils import combine_metrics

# ===============================================================================
# Configuration
# ===============================================================================


class QUACKConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    checkpoint_dir: str


class QUACKSaveState(TypedDict):
    step: int
    val_reward: float
    consumed_samples: int


def _default_quack_save_state() -> QUACKSaveState:
    return {
        "step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


class MasterConfig(TypedDict):
    actor: PolicyConfig
    critic: PolicyConfig
    env_configs: Dict[str, Any]
    data: DataConfig
    quack: QUACKConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[
    PolicyInterface,
    GenerationInterface,
    GenerationInterface,
    RayVirtualCluster,
    RayVirtualCluster,
    NLLLoss,
    Logger,
    CheckpointManager,
    QUACKSaveState,
    MasterConfig,
]:
    """Main entry point for running QUACK algorithm.

    Returns:
        Tuple of actor, actor_generation, critic_generation, 
        actor_cluster, critic_cluster, loss_fn, logger, checkpointer, quack_save_state, master_config
    """
    # Extract individual configs for easier access
    actor_config = master_config["actor"]
    actor_generation_config = master_config["actor"]["generation"]
    critic_generation_config = master_config["critic"]["generation"]
    quack_config = master_config["quack"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    quack_save_state: Optional[QUACKSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    if quack_save_state is None:
        quack_save_state = _default_quack_save_state()

    # config validation checks
    if master_config["checkpointing"]["enabled"]:
        assert master_config["checkpointing"]["save_period"] > 0
        assert (
            master_config["checkpointing"]["save_period"]
            % master_config["quack"]["val_period"]
            == 0
        ), (
            f"Checkpointing save period {master_config['checkpointing']['save_period']} "
            f"must be a multiple of validation period {master_config['quack']['val_period']}"
            f", or we won't know what metric to save!"
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    assert actor_generation_config["backend"] == 'vllm', "Quack is pretty generation heavy, rerun with vllm backend."
    critic_generation_config["backend"] = 'vllm'

    total_nodes = cluster_config["num_nodes"]
    critic_nodes = master_config["critic"]["num_nodes"] or 0
    actor_nodes = total_nodes - critic_nodes
    
    # Ensure total nodes used doesn't exceed available, adjust if necessary
    if actor_nodes + critic_nodes > total_nodes and total_nodes > 0:
        # This can happen if total_nodes is 1, actor_nodes becomes 1, critic_nodes becomes 1.
        # Prioritize actor if only 1 node is available.
        if total_nodes == 1:
            actor_nodes = 1
            critic_nodes = 0 # Critic will run on CPU or fail if GPU is required by its config
        else: # if total_nodes > 1, and sum exceeds, reduce critic first
            critic_nodes = total_nodes - actor_nodes


    print(f"  Allocating {actor_nodes} nodes for actor cluster and {critic_nodes} nodes for critic cluster.")

    actor_max_colocated_worker_groups = 1 + int(actor_generation_config["backend"] == 'vllm')
    actor_cluster = RayVirtualCluster(
        name="actor_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * actor_nodes,
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=actor_max_colocated_worker_groups,
    )
    print(f"  ✓ Actor Ray cluster initialized with {actor_nodes} nodes")

    critic_max_colocated_worker_groups = 1 # For the critic_generation vLLM
    # Only create critic cluster if critic_nodes > 0
    critic_cluster = None
    if critic_nodes > 0:
        critic_cluster = RayVirtualCluster(
            name="critic_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * critic_nodes,
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=critic_max_colocated_worker_groups,
        )
        print(f"  ✓ Critic Ray cluster initialized with {critic_nodes} nodes")
    else:
        print(f"  ℹ️ Critic cluster not initialized as 0 nodes are allocated. Critic might run on CPU or actor's resources if not vLLM, or fail.")


    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")
    # vllm model loading prefers clean environment, initialize actor_generation before actor (#52 will fix this)
    # model_name is needed for vLLM
    actor_generation_config["model_name"] = actor_config["model_name"]
    critic_generation_config["model_name"] = actor_config["model_name"] # Assuming critic uses the same base model
    
    # vllm generation
    actor_generation = VllmGeneration(cluster=actor_cluster, config=actor_generation_config, name_prefix="vllm_actor")

    # Worker groups are not initialized until the first call to run something on workergroups.
    # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).

    # important to call this before initializing critic_generation, this will ensure offload
    actor_generation.finish_generation()
    
    critic_generation = None
    if critic_cluster: # only initialize if critic cluster exists
        critic_generation = VllmGeneration(cluster=critic_cluster, config=critic_generation_config, name_prefix="vllm_critic")
    elif critic_generation_config["backend"] == 'vllm':
        # If critic is vLLM but has no cluster, this is problematic.
        # For now, let's try to put it on actor_cluster, though this defeats the purpose of separation.
        # A better solution would be to require critic_nodes > 0 for vLLM critic.
        print(f"  ⚠️ Warning: Critic backend is vLLM but no dedicated nodes allocated. Attempting to use actor_cluster for critic_generation.")
        critic_generation = VllmGeneration(cluster=actor_cluster, config=critic_generation_config, name_prefix="vllm_critic_on_actor_cluster")
    else:
        # If critic is not vLLM, it might not need a Ray cluster or can use a default/local setup.
        # This part of the logic might need to be adapted based on how non-vLLM critics are handled.
        print(f"  ℹ️ Critic backend is not vLLM ('{critic_generation_config['backend']}'). critic_generation will not be a VllmGeneration instance on a dedicated cluster.")
        # Placeholder: Initialize critic_generation appropriately if it's not vLLM and needs specific setup.
        # For now, assuming if not vLLM, it might be run differently or this will lead to an error later if not handled.
        pass


    if critic_generation and hasattr(critic_generation, 'finish_generation'): # Check if critic_generation is not None and has the method
        critic_generation.finish_generation()
    print(
        f"  ✓ Using vLLM backend for generation with {actor_config['model_name']}"
    )

    actor = HfPolicy(
        cluster=actor_cluster, # Actor policy uses the actor_cluster
        config=actor_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "actor" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "actor" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
    )

    loss_fn = NLLLoss()

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        actor,
        actor_generation,
        critic_generation,
        actor_cluster, # Return actor_cluster
        critic_cluster, # Return critic_cluster
        loss_fn,
        logger,
        checkpointer,
        quack_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================


def refit(
    actor: PolicyInterface,
    generation: GenerationInterface,
    refit_buffer_size_gb: int,  # GB
):
    """Refit the actor generation interface with the latest actor weights."""
    actor.offload_before_refit()
    generation.prepare_for_generation(tags=["weights"])
    # Streaming update weights to save memory
    state_dict_info = actor.prepare_weights_for_ipc()
    # group keys to save time
    available_bytes = refit_buffer_size_gb * (1024**3)
    split_keys, keys = [], []
    for key, size_in_bytes in state_dict_info:
        if size_in_bytes > available_bytes:
            if keys:
                split_keys.append(keys)
                keys = []
            available_bytes = refit_buffer_size_gb * (1024**3)

        keys.append(key)
        available_bytes -= size_in_bytes

    if len(keys) > 0:
        split_keys.append(keys)
    # do update
    for keys in split_keys:
        ipc_handles = actor.get_weights_ipc_handles(keys)
        generation.update_weights(ipc_handles)
    actor.offload_after_refit()
    generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================


def quack_train(
    actor: PolicyInterface,
    actor_generation: Optional[GenerationInterface],
    critic_generation: Optional[GenerationInterface],
    dataset: Dataset,
    val_dataset: Optional[Dataset],
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: Dict[str, EnvironmentInterface],
    logger: Logger,
    checkpointer: CheckpointManager,
    quack_save_state: Optional[QUACKSaveState],
    master_config: MasterConfig,
):
    """Run QUACK training algorithm."""

    quack_config = master_config["quack"]

    # ==========================
    #           Data
    # ==========================
    print("\n▶ Setting up dataloaders...")
    prompt_dataset = setup_data(dataset, tokenizer, master_config["data"], "math")
    prompt_dataloader = StatefulDataLoader(
        prompt_dataset,
        batch_size=quack_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=rl_collate_fn,
    )
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        prompt_dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

    # Load validation dataset if provided
    val_prompt_dataloader = None
    # If validation is enabled, load the validation dataloader
    if quack_config["val_period"] > 0 or quack_config["val_at_start"]:
        val_prompt_dataset = setup_data(val_dataset, tokenizer, master_config["data"], "math")
        val_prompt_dataloader = StatefulDataLoader(
            val_prompt_dataset,
            batch_size=quack_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ==========================
    #       Replay Buffer
    # ==========================

    replay_buffer = ReplayBuffer(buffer_size=master_config["replay_buffer"]["buffer_size"])

    # ==========================
    #           Training
    # ==========================
    timer = Timer()
    NEED_REFIT_ACTOR = True
    NEED_REFIT_CRITIC = True

    # common config/state itmes
    step = quack_save_state["step"]
    consumed_samples = quack_save_state["consumed_samples"]
    critic_refit_period = master_config["quack"]["critic_refit_period"]
    val_period = master_config["quack"]["val_period"]
    val_at_start = master_config["quack"]["val_at_start"]
    refit_buffer_size_gb = master_config["actor"]["refit_buffer_size_gb"]

    # track if generation needs a refit before running
    ACTOR_GENERATION_STALE = True
    CRITIC_GENERATION_STALE = lambda _: critic_generation is not None

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT_ACTOR and ACTOR_GENERATION_STALE:
            refit(actor, actor_generation, refit_buffer_size_gb)
            ACTOR_GENERATION_STALE = False
        else:
            actor_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            actor_generation,
            val_prompt_dataloader,
            tokenizer,
            task_to_env,
            step=0,
            master_config=master_config,
        )
        actor_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run quack training (single-turn)
    batch: BatchedDataDict[DatumSpec]
    for batch in prompt_dataloader:
        print(
            f"\n{'=' * 25} Step {step + 1}/{min(len(prompt_dataloader), master_config['quack']['max_num_steps'])} {'=' * 25}"
        )

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["quack"]["num_generations_per_prompt"]
                )

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating *ANSWERS* for batch of size {repeated_batch.size}...")
            with timer.time("prepare_actor_for_generation"):
                if NEED_REFIT_ACTOR and ACTOR_GENERATION_STALE:
                    refit(
                        actor,
                        actor_generation,
                        refit_buffer_size_gb,
                    )
                    ACTOR_GENERATION_STALE = False
                else:
                    actor_generation.prepare_for_generation()

            with timer.time("actor_generation"):
                repeated_batch, rollout_metrics_actor = run_multi_turn_rollout(
                    policy_generation=actor_generation,
                    input_batch=repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["actor"]["generation"]["max_new_tokens"],
                    max_rollout_turns=master_config["quack"]["max_rollout_turns"],
                    greedy=False,
                )
                actor_generation.finish_generation()
                buffer_items = convert_actor_rollouts_to_buffer_items(repeated_batch)
                replay_buffer.push_batch(buffer_items)
            
            # check if we have enough samples in the replay buffer
            # we also make sure critic is refitted on the first step
            if not replay_buffer.can_sample(master_config["quack"]["train_dataset_size"]):
                print(f"  ⚠️ Not enough samples in replay buffer ({len(replay_buffer)}/{master_config['quack']['train_dataset_size']}), skipping training for now...")
                # still need to advance step and check max_num_steps
                step += 1
                if step >= master_config["quack"]["max_num_steps"]:
                    break
                continue
            
            if not critic_generation:
                print(f"  ⚠️ Critic generation service is not available. Skipping training step as critiques cannot be generated.")
                logger.log_metrics({f"error_step_{step+1}": "critic_generation_unavailable"}, step + 1, prefix="error")
                step += 1
                if step >= master_config["quack"]["max_num_steps"]:
                    break
                continue

            # get train dataset batch from replay buffer
            with timer.time("sampling_from_buffer"):
                critic_dataset = replay_buffer.sample(master_config["quack"]["train_dataset_size"])
                critic_dataset = setup_data(critic_dataset, tokenizer, master_config["critic_data"], "critic")
                critic_iterator = DataLoader(
                    critic_dataset, 
                    batch_size=master_config["quack"]["critic_inference_batch_size"], 
                    shuffle=False, 
                    collate_fn=rl_collate_fn
                )
                # critic_batch = rl_collate_fn(critic_dataset)    # NOTE: we assuem critic dataset is small enough to fit into memory

            print(f"▶ Generating *CRITIQUES* for dataset of size {len(critic_dataset)}...")
            with timer.time("prepare_critic_for_generation"):
                if NEED_REFIT_CRITIC and CRITIC_GENERATION_STALE(step):
                    print(f"  Refitting critic generation for step {step +1}...")
                    refit(
                        actor, # Assuming actor weights are used for critic model, or critic has its own policy object
                        critic_generation,
                        refit_buffer_size_gb,
                    )
                    CRITIC_GENERATION_STALE = lambda current_step: current_step % critic_refit_period == 0
                else:
                    print(f"  Preparing critic generation (no refit) for step {step+1}...")
                    critic_generation.prepare_for_generation()

            with timer.time("critic_generation"):
                critic_micro_batch_list = []
                for critic_micro_batch in critic_iterator:
                    critic_micro_batch, rollout_metrics_critic = run_multi_turn_rollout(
                        policy_generation=critic_generation,
                        input_batch=critic_micro_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["critic"]["generation"]["max_new_tokens"],
                        max_rollout_turns=master_config["quack"]["max_rollout_turns"],
                        greedy=False,
                    )
                    critic_micro_batch_list.append(critic_micro_batch)
                critic_batch = BatchedDataDict.from_batches(critic_micro_batch_list)
                critic_generation.finish_generation()
                
            with timer.time("data_processing"):
                print("▶ Preparing batch...")

                ## add loss mask based on role to every message
                add_loss_mask_to_message_log(
                    critic_batch["message_log"],
                    roles_to_train_on=["assistant"],
                )

                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    critic_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["actor"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                train_data: BatchedDataDict = BatchedDataDict(
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": critic_batch["loss_multiplier"],
                    }
                )

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                actor.prepare_for_training()  # set model train and reload optim to GPU
                ACTOR_GENERATION_STALE = True

            print("▶ Training actor...")
            with timer.time("actor_training"):
                train_results = actor.train(train_data, loss_fn)
                # train_results_list = []
                # for train_micro_batch in train_data.make_microbatch_iterator(master_config["quack"]["train_dataset_batch_size"]):
                #     train_results = actor.train(train_micro_batch, loss_fn)
                #     train_results_list.append(train_results)

                # # Combine results from all microbatches
                # train_results = combine_metrics(train_results_list)

            # Run validation if it's a validation step
            if val_period > 0 and (step + 1) % val_period == 0:
                if NEED_REFIT_ACTOR and ACTOR_GENERATION_STALE:
                    refit(
                        actor,
                        actor_generation,
                        refit_buffer_size_gb,
                    )
                    ACTOR_GENERATION_STALE = False
                else:
                    actor_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    actor_generation,
                    val_prompt_dataloader,
                    tokenizer,
                    task_to_env,
                    step=step + 1,
                    master_config=master_config,
                )
                actor_generation.finish_generation()
                logger.log_metrics(
                    validation_timings, step + 1, prefix="timing/validation"
                )
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            consumed_samples += master_config["quack"]["num_prompts_per_step"]
            if (
                master_config["checkpointing"]["enabled"]
                and (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):  # +1 because step is 0-indexed
                actor.prepare_for_training()

                quack_save_state["step"] = step + 1
                quack_save_state["val_reward"] = val_metrics["accuracy"]
                quack_save_state["consumed_samples"] = consumed_samples
                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, quack_save_state, master_config
                    )
                    actor.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "actor", "weights"),
                        optimizer_path=os.path.join(
                            checkpoint_path, "actor", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "actor", "tokenizer"
                        ),
                    )
                    torch.save(
                        prompt_dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)
                actor.offload_after_refit()

        # Logging
        # Log training data
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = [item['reward'] for item in critic_batch["extra_env_info"]]
        log_data["critic_reward"] = critic_batch["total_reward"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        print("\n📊 Training Results:")
        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": repeated_batch["total_reward"].numpy(),   # read from repeated_batch to reflect the reward of the latest generated answers
            "critic_reward": critic_batch["total_reward"].numpy(),
            "grad_norm": train_results["grad_norm"].numpy(),
        }
        metrics.update(train_results["all_mb_metrics"])
        for k, v in metrics.items():
            metrics[k] = np.mean(v).item()
        metrics.update(rollout_metrics_actor)

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        print(f"  • Loss: {metrics['loss']:.4f}")
        print(f"  • Avg Reward: {np.mean(repeated_batch['total_reward'].numpy()):.4f}")
        print(f"  • Critic Reward: {np.mean(critic_batch['total_reward'].numpy()):.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics_actor['mean_gen_tokens_per_sample']:.4f}"
        )

        print("\n⏱️  Timing:")
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)
        print(f"  • Total step time: {total_time:.2f}s")

        # Display all other timing metrics
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        logger.log_metrics(metrics, step + 1, prefix="train")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        timer.reset()
        step += 1
        if step >= master_config["quack"]["max_num_steps"]:
            print(f"Reached max_num_steps ({master_config['quack']['max_num_steps']}). Stopping training.")
            break


def validate(
    actor_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    val_task_to_env: Dict[str, EnvironmentInterface],
    step: int,
    master_config: MasterConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["quack"]["max_val_samples"]
            // master_config["quack"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            val_batch, gen_metrics = run_multi_turn_rollout(
                actor_generation,
                val_batch,
                tokenizer,
                val_task_to_env,
                max_seq_len=master_config["actor"]["max_total_sequence_length"],
                max_rollout_turns=master_config["quack"]["max_rollout_turns"],
                greedy=False,
            )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = get_keys_from_message_log(
                val_batch["message_log"], ["role", "content"]
            )
            all_message_logs.extend(to_env)

        # Calculate validation metrics
        accuracy = sum(total_rewards) / len(total_rewards)
        avg_length = sum(total_lengths) / len(total_lengths)

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
