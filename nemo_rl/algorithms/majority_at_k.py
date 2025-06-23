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
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.grpo import (
    setup,
    refit_policy_generation,
    validate,
    MasterConfig,
    GRPOSaveState,
    _default_grpo_save_state,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    print_message_log_samples,
)
from nemo_rl.utils.timer import Timer
from nemo_rl.algorithms.utils import _calculate_single_majority_at_k, calculate_math_majority_at_k, calculate_majority_at_k_advantages


def majority_at_k_train(
    policy: PolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: Dict[str, EnvironmentInterface],
    val_task_to_env: Optional[Dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: Optional[GRPOSaveState],
    master_config: MasterConfig,
):
    """Run Majority@K training algorithm."""
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (hf framework backend)
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running

    # common config/state items
    step = grpo_save_state["step"]
    optim_step = grpo_save_state["optim_step"]

    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    refit_buffer_size_gb = master_config["policy"]["refit_buffer_size_gb"]

    num_epochs = master_config["grpo"]["num_epochs"]
    max_num_steps = num_epochs * len(dataloader)

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, refit_buffer_size_gb)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run majority@K training
    iter_dataloader = iter(dataloader)

    while step < max_num_steps:
        try:
            batch = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            batch = next(iter_dataloader)

        print(f"\n{'=' * 25} Step {step + 1}/{max_num_steps} {'=' * 25}")
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating responses for batch of size {repeated_batch.size}...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                repeated_batch, rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
                policy_generation.finish_generation()

            # Calculate majority@K advantages
            print("▶ Computing majority@K advantages...")
            with timer.time("advantage_calculation"):
                # Extract rewards from final_batch
                rewards = repeated_batch["total_reward"]

                print("▶ Computing majority@K advantages...")
                advantages = calculate_majority_at_k_advantages(
                    repeated_batch["message_log"],
                    input_ids,
                    rewards,
                    torch.ones_like(rewards),
                    bias=master_config["grpo"]["use_majority_at_k_bias_reduction"],
                ).unsqueeze(-1)

                # Calculate statistics for logging
                advantages_min, advantages_mean, advantages_max = (
                    advantages.min(),
                    advantages.mean(),
                    advantages.max(),
                )
                reward_min, reward_mean, reward_max = (
                    rewards.min(),
                    rewards.mean(),
                    rewards.max(),
                )

                rollout_metrics.update(
                    {
                        "advantages_min": advantages_min,
                        "advantages_mean": advantages_mean,
                        "advantages_max": advantages_max,
                        "reward_min": reward_min,
                        "reward_mean": reward_mean,
                        "reward_max": reward_max,
                        "math_majority_at_k": calculate_math_majority_at_k(
                            repeated_batch["message_log"], input_ids, rewards, torch.ones_like(rewards)
                        ),
                    }
                )

            with timer.time("data_processing"):
                # Add loss mask and advantages to each message in LLMMessageLogType
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"]
                            )
                        else:
                            message["token_loss_mask"] = torch.zeros_like(
                                message["token_ids"]
                            )
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = advantages[i].expand(
                            message["token_ids"].shape
                        )

                # Convert updated LLMMessageLogType to FlatMessagesType for training
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                # Create training data from flattened messages
                train_data = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "advantages": flat_messages["advantages"],
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )
                train_data.to("cpu")

            print("▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...")
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...")
            with timer.time("policy_training"):
                list_of_train_metrics = policy.train(train_data, loss_fn)

            is_last_step = step + 1 == min(max_num_steps, len(dataloader))

            # Run validation if it's a validation step
            if is_last_step or (val_period > 0 and (step + 1) % val_period == 0):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step + 1,
                    master_config=master_config,
                    logger=logger,
                )
                policy_generation.finish_generation()
                logger.log_metrics(
                    validation_timings, step + 1, prefix="timing/validation"
                )
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            consumed_samples += master_config["grpo"]["num_prompts_per_step"]
            if master_config["checkpointing"]["enabled"] and (
                is_last_step
                or (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):  # +1 because step is 0-indexed
                policy.prepare_for_training()

                grpo_save_state["step"] = step + 1
                grpo_save_state["val_reward"] = val_metrics["accuracy"]
                grpo_save_state["consumed_samples"] = consumed_samples
                grpo_save_state["optim_step"] = optim_step + len(list_of_train_metrics)
                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)
                policy.offload_after_refit()

        # Logging
        # Log training data
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = rewards.tolist()
        log_data["advantages"] = advantages.squeeze(-1).tolist()
        log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")
        table = logger.log_batched_dict_as_table(log_data, prefix="train", step=step)

        print("\n📊 Training Results:")

        rollout_metrics["table"] = table
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
        print(f"  • Avg Advantage: {np.mean(advantages.numpy()):.4f}")
        print(f"  • Majority@K Score: {rollout_metrics['math_majority_at_k']:.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
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

        for i, train_step_metric in enumerate(list_of_train_metrics):
            train_step_metric["optim_step"] = optim_step + i + 1
            train_step_metric["outer_loop_step"] = step + 1
            logger.log_metrics(
                train_step_metric,
                train_step_metric["optim_step"],
                prefix="train",
            )

        logger.log_metrics(rollout_metrics, step + 1, prefix="train_rollout")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        timer.reset()
        step += 1
        optim_step += len(list_of_train_metrics)

        if step >= max_num_steps:
            break


def _calculate_math_majority_at_k(message_logs, prompts, rewards, valid_mask):
    """Calculate the overall majority@K score across all prompts for logging."""
    import re
    from collections import Counter
    
    unique_prompts = torch.unique(prompts, dim=0)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = torch.device("cpu")
    
    total_score = 0.0
    num_prompts = 0
    
    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]
        
        if valid_mask[prompt_idx].sum() <= 1:
            continue
            
        # Extract answers for this prompt
        answers = []
        prompt_rewards = []
        
        for idx in prompt_idx:
            if valid_mask[idx] == 0:
                continue
                
            # Extract answer from message log
            message_log = message_logs[idx.item()]
            extracted_answer = ""
            
            # Find last assistant response and extract answer
            for message in reversed(message_log):
                if message["role"] == "assistant":
                    response = message["content"]
                    # Try to extract from \boxed{}
                    boxed_match = re.search(r'\\boxed\{([^}]*)\}', response)
                    if boxed_match:
                        extracted_answer = boxed_match.group(1).strip()
                    break
            
            answers.append(extracted_answer)
            prompt_rewards.append(rewards[idx].item())
        
        if len(answers) == 0:
            continue
            
        num_prompts += 1
        majority_score = _calculate_single_majority_at_k(answers, prompt_rewards)
        total_score += majority_score
    
    return total_score / num_prompts if num_prompts > 0 else 0.0 