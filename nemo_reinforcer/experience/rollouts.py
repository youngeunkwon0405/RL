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

# Generate rollouts

import torch
from typing import List, Tuple, Dict, Optional, Any, NamedTuple
from transformers import AutoTokenizer
import ray

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_reinforcer.data.llm_message_utils import (
    get_keys_from_message_log,
    batched_message_log_to_flat_message,
)
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
)
from nemo_reinforcer.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


# Return type for calculate_rewards
class RewardsOutput(NamedTuple):
    rewards: torch.Tensor
    env_observations: List[Dict[str, str]]
    terminateds: torch.Tensor
    next_stop_strings: List[Optional[List[str]]]
    metadata: List[Optional[Dict[str, Any]]]


def generate_responses(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: AutoTokenizer,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
) -> Tuple[BatchedDataDict[DatumSpec], List[torch.Tensor], dict]:
    """Generate responses from policy."""
    # Add stop_strings to generation_input_data if present in the batch
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        # Ensure the key exists even if it's None, matching GenerationDatumSpec
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    # Generate responses
    generation_outputs = policy_generation.generate(
        generation_input_data, greedy=greedy
    )

    # Extract generated tokens
    generated_ids = []
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]
    for output_ids, input_length, total_length in zip(
        generation_outputs["output_ids"], input_lengths, unpadded_sequence_lengths
    ):
        generated_ids.append(output_ids[input_length:total_length])

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        message = {
            "role": "assistant",
            "content": text,
            "token_ids": generation_outputs["output_ids"][i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(message)

    metrics = {
        "mean_generation_length": (
            torch.sum(unpadded_sequence_lengths) - torch.sum(input_lengths)
        ).item()
        / len(unpadded_sequence_lengths),
        "max_seqlen": torch.max(unpadded_sequence_lengths).item(),
    }

    return batch, generated_ids, metrics


def calculate_rewards(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: Dict[str, EnvironmentInterface],
) -> RewardsOutput:
    """Calculate rewards for generated responses and get environment feedback.

    Args:
        batch: Batch containing message_log (LLMMessageLogType) with generated responses
        task_to_env: Dictionary mapping task names to their corresponding environments

    Returns:
        Tuple containing:
            - rewards: Tensor of rewards for the last turn.
            - env_observations: List of observations from the environment for the next turn.
            - terminateds: Tensor of booleans indicating if an episode ended naturally.
            - next_stop_strings: List of stop strings for the next generation step.
            - metadata: List of extracted metadata from the environment.
    """
    # Extract message logs for environment (most recent interaction)
    to_env = [
        get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
        for i in range(len(batch["message_log"]))
    ]
    task_names = [batch["task_name"][i] for i in range(len(batch["task_name"]))]

    # Group messages by task type
    task_groups = {}
    for i, task_name in enumerate(task_names):
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append((i, to_env[i]))

    # Calculate rewards for each task group concurrently
    futures = []
    future_to_indices = {}  # Map future to its corresponding indices
    for task_name, group in task_groups.items():
        if task_name not in task_to_env:
            raise ValueError(f"No environment found for task type: {task_name}")

        # Extract indices and messages for this group
        indices = [idx for idx, _ in group]
        messages = [msg for _, msg in group]

        # Get corresponding environment info
        env_info = [batch["extra_env_info"][i] for i in indices]

        # Submit task to environment and store future
        future = task_to_env[task_name].step.remote(messages, env_info)
        futures.append(future)
        future_to_indices[future] = indices

    results = ray.get(futures)
    all_rewards = []
    all_env_observations = []
    all_terminateds = []
    all_next_stop_strings = []
    all_metadata = []  # Store extracted metadata
    all_indices_order = []

    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        # Environment step returns: EnvironmentReturn
        env_observations, metadata, next_stop_strings, task_rewards, terminateds = (
            result
        )

        # Store results with their original indices
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            all_rewards.append(task_rewards[i])
            all_env_observations.append(env_observations[i])
            all_terminateds.append(terminateds[i])
            all_next_stop_strings.append(next_stop_strings[i])
            # Extract metadata specifically
            all_metadata.append(metadata[i])

    # Sort results by original index to maintain order
    sorted_indices = sorted(
        range(len(all_indices_order)), key=lambda k: all_indices_order[k]
    )
    rewards = torch.tensor([all_rewards[i] for i in sorted_indices])
    env_observations = [all_env_observations[i] for i in sorted_indices]
    terminateds = torch.tensor([all_terminateds[i] for i in sorted_indices])
    next_stop_strings = [all_next_stop_strings[i] for i in sorted_indices]
    metadata = [all_metadata[i] for i in sorted_indices]  # Sort metadata

    # Ensure tensors are on CPU
    rewards = rewards.cpu()
    terminateds = terminateds.cpu()

    return RewardsOutput(
        rewards=rewards,
        env_observations=env_observations,
        terminateds=terminateds,
        next_stop_strings=next_stop_strings,
        metadata=metadata,
    )


def run_multi_turn_rollout(
    policy_generation: GenerationInterface,
    initial_batch: BatchedDataDict[DatumSpec],
    tokenizer: AutoTokenizer,
    task_to_env: Dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_turns: int = 999999,
    greedy: bool = False,
) -> BatchedDataDict[DatumSpec]:
    """Runs a multi-turn rollout loop, interacting with the environment.

    Args:
        policy_generation: The generation interface (policy).
        initial_batch: The starting batch containing initial message logs.
        tokenizer: The tokenizer.
        task_to_env: Dictionary mapping task names to environment instances.
        max_turns: Maximum number of agent-environment interaction turns.
        max_seq_len: Maximum sequence length allowed.

    Returns:
        BatchedDataDict containing the full interaction history in 'message_log'
        and accumulated rewards in a new 'total_reward' field.
    """
    current_batch = initial_batch.copy()  # Work on a copy
    batch_size = len(current_batch["message_log"])
    active_indices = torch.arange(batch_size)
    turn_rewards = torch.zeros(batch_size, dtype=torch.float32)
    total_rewards = torch.zeros(batch_size, dtype=torch.float32)

    # Initialize stop_strings from the initial batch if present
    current_stop_strings = current_batch.get("stop_strings", [None] * batch_size)

    for turn in range(max_turns):
        if len(active_indices) == 0:
            print(f"  Turn {turn + 1}/{max_turns}: All samples finished.")
            break

        print(
            f"  Turn {turn + 1}/{max_turns}: Processing {len(active_indices)} active samples..."
        )

        # --- Prepare data for active samples ---
        active_batch = current_batch.select_indices(active_indices)
        active_stop_strings = [current_stop_strings[i] for i in active_indices.tolist()]

        # --- Prepare Generation Input ---
        # Instead of using batched_message_log_to_flat_message,
        # we prepare the input specifically for generation here.
        # Concatenate token_ids from the message log for each active sample.
        active_input_ids_list = []
        for log in active_batch["message_log"]:
            # Ensure all messages have token_ids (should be true due to previous fix)
            concatenated_ids = torch.cat(
                [msg["token_ids"] for msg in log if "token_ids" in msg]
            )
            active_input_ids_list.append(concatenated_ids)

        # Pad the concatenated sequences for the current turn
        active_input_lengths = torch.tensor(
            [len(ids) for ids in active_input_ids_list], dtype=torch.int32
        )
        padded_active_input_ids = torch.nn.utils.rnn.pad_sequence(
            active_input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        # --------------------------------

        # Check max sequence length before generation
        if torch.any(active_input_lengths >= max_seq_len):
            exceeded_mask = active_input_lengths >= max_seq_len
            print(
                f"    WARNING: {exceeded_mask.sum()} active samples reached max_seq_len ({max_seq_len}) before generation."
            )
            # Mark exceeded as truncated and remove from active set for this turn
            truncated_indices_local = torch.where(exceeded_mask)[0]
            # Map local indices back to original batch indices
            truncated_indices_global = active_indices[truncated_indices_local]
            # (We'll handle removal after reward calculation)

        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": padded_active_input_ids,  # Use padded sequences
                "input_lengths": active_input_lengths,  # Use correct lengths
                "stop_strings": active_stop_strings,  # Pass current stop strings
            }
        )

        # --- Generate assistant response ---
        # generate_responses updates active_batch["message_log"] in-place
        active_batch, _, gen_metrics = generate_responses(
            policy_generation,
            generation_input_data,
            active_batch,
            tokenizer,
            active_input_lengths,
            greedy=greedy,
        )
        print(
            f"    Generated responses (Avg len: {gen_metrics['mean_generation_length']:.1f})"
        )

        # --- Calculate rewards and get environment feedback ---
        env_output: RewardsOutput = calculate_rewards(active_batch, task_to_env)

        turn_rewards[active_indices] = env_output.rewards
        total_rewards[active_indices] += turn_rewards[active_indices]
        print(
            f"    Calculated rewards (Avg: {turn_rewards[active_indices].mean():.3f})"
        )

        # --- Update message log for ALL active samples with env observation ---
        # This must happen BEFORE filtering based on done flags
        for i, global_idx in enumerate(active_indices.tolist()):
            env_obs_content = env_output.env_observations[i]["content"]
            env_obs_role = env_output.env_observations[i]["role"]
            # Tokenize the raw content from the environment
            # The chat template will be applied implicitly during the next
            # generation step when the full message log is processed.
            tokenized_obs = tokenizer(
                env_obs_content, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            tokenized_env_obs_message = {
                "role": env_obs_role,  # Use the role provided by env (now 'environment')
                "content": env_obs_content,
                "token_ids": tokenized_obs,
            }
            current_batch["message_log"][global_idx].append(tokenized_env_obs_message)

        # --- Determine done samples and update active set ---
        done = env_output.terminateds
        active_mask = ~done

        # Identify samples that just finished this turn
        newly_finished_indices_local = torch.where(done)[0]
        newly_finished_indices_global = active_indices[newly_finished_indices_local]
        print(
            f"    {len(newly_finished_indices_global)} samples finished this turn."
            f" (Terminated: {env_output.terminateds.sum()})"
        )

        # Update active indices for the next iteration
        active_indices_local_next = torch.where(active_mask)[0]
        active_indices = active_indices[active_indices_local_next]

        # --- Update state only for CONTINUING samples ---
        continuing_indices_global = active_indices  # Indices relative to original batch
        # Get next stop strings and infos corresponding to the indices that are *continuing*
        continuing_next_stops = [
            env_output.next_stop_strings[i] for i in active_indices_local_next.tolist()
        ]
        # Get metadata corresponding to continuing indices, using the correct field name
        continuing_metadata = [
            env_output.metadata[i] for i in active_indices_local_next.tolist()
        ]

        for i, global_idx in enumerate(continuing_indices_global.tolist()):
            # Update stop strings for the next turn
            current_stop_strings[global_idx] = continuing_next_stops[i]
            # Update metadata (extra_env_info) using info from environment
            if continuing_metadata[i] is not None:
                current_batch["extra_env_info"][global_idx] = continuing_metadata[i]

        # Check max sequence length for continuing samples AFTER env feedback
        if len(continuing_indices_global) > 0:
            continuing_batch = current_batch.select_indices(continuing_indices_global)
            # Calculate lengths based on the updated message log
            continuing_lengths_list = []
            for log in continuing_batch["message_log"]:
                concatenated_ids = torch.cat(
                    [msg["token_ids"] for msg in log if "token_ids" in msg]
                )
                continuing_lengths_list.append(len(concatenated_ids))
            continuing_lengths = torch.tensor(
                continuing_lengths_list, dtype=torch.int32
            )

            if torch.any(continuing_lengths >= max_seq_len):
                exceeded_mask = continuing_lengths >= max_seq_len
                exceeded_indices_local = torch.where(exceeded_mask)[0]
                exceeded_indices_global = continuing_indices_global[
                    exceeded_indices_local
                ]
                print(
                    f"    WARNING: {len(exceeded_indices_global)} active samples reached max_seq_len ({max_seq_len}) after env feedback."
                )
                # Remove these from active set for the *next* turn
                active_indices = torch.tensor(
                    [
                        idx
                        for idx in active_indices.tolist()
                        if idx not in exceeded_indices_global.tolist()
                    ],
                    dtype=torch.long,
                )

    # Add total rewards to the final batch
    current_batch["total_reward"] = total_rewards
    print(f"Rollout finished. Final avg total reward: {total_rewards.mean():.3f}")
    return current_batch
