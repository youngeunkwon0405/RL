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

import threading as _threading
import time
from typing import Any, Optional

import ray
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase


@ray.remote
class ReplayBuffer:
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.trajectories = []
        self.trajectory_versions = []  # weight-version used for generation

    def push_with_wait_signal(
        self, trajectory: dict[str, Any], weight_version: int
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
        """
        if len(self.trajectories) > self.max_size:
            return "full"

        print("🔍 ReplayBuffer.push_with_wait_signal: Adding trajectory")
        self.trajectories.append(trajectory)
        self.trajectory_versions.append(weight_version)
        print(
            f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}"
        )
        return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "max_size": self.max_size,
        }

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[list]:
        """Sample per-prompt trajectory groups that fit within the age window and removes the sampled groups from the buffer."""
        if not self.trajectories:
            return None

        # Treat all trajectories as valid; training will consume and evict
        total_trajectories = len(self.trajectories)
        print("🔍 ReplayBuffer sampling debug:")
        print(
            f"   current_weight_version={current_weight_version}, max_age_steps={max_age_steps}"
        )
        print(f"   trajectory_versions={self.trajectory_versions}")
        valid_indices = list(range(total_trajectories))
        if not valid_indices:
            print("No trajectories available for sampling.")
            return None

        # Enforce exact number of groups if available; otherwise, signal to wait
        if len(valid_indices) < num_prompt_groups:
            print(
                f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
            )
            return None

        # FIFO selection of earliest trajectories to maintain order
        selected: list[int] = valid_indices[:num_prompt_groups]

        from collections import Counter

        sampled_weights = [self.trajectory_versions[i] for i in selected]
        print(f"✅ Selected counts by weight-version: {Counter(sampled_weights)}")

        sampled_items = [self.trajectories[i] for i in selected]

        for idx in sorted(selected, reverse=True):
            self.trajectory_versions.pop(idx)
            self.trajectories.pop(idx)
        print(
            f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}"
        )

        return sampled_items

    def size(self) -> int:
        """Return current buffer size."""
        return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        self.trajectories.clear()
        self.trajectory_versions.clear()


@ray.remote
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.running = False

        self._pg_lock: _threading.Lock = _threading.Lock()

        # Event for manual pause/resume control
        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self.current_weight_version: int = start_step

        # Track generations per weight version to prevent length bias
        self.generations_per_weight_version: dict[int, int] = {}

        # Track when generation limits cause collection to pause
        self._last_limit_warning_version = None

        # Event to signal when generation limits are cleared (more efficient than polling)
        self._generation_limit_cleared = _threading.Event()
        self._generation_limit_cleared.set()  # Start in cleared state

        # Check if we should use async rollouts
        self._use_async_rollouts = False
        if (
            hasattr(policy_generation, "cfg")
            and "vllm_cfg" in policy_generation.cfg
            and policy_generation.cfg["vllm_cfg"].get("async_engine", False)
        ):
            self._use_async_rollouts = True
            print(
                "Trajectory collector: Detected vLLM async engine; enabling async rollouts in collector"
            )

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()
        # Limit in-flight generator requests to num_prompts_per_step
        max_inflight = int(self.master_config["grpo"]["num_prompts_per_step"]) or 1
        self._inflight_sema = _threading.Semaphore(max_inflight)

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        # # Clean up generation counts for old weight versions outside the window
        # max_age_steps = self.master_config["async_grpo"]["max_trajectory_age_steps"]
        # min_valid_version = version - max_age_steps

        # # Remove counts for weight versions that are too old
        # old_versions = [v for v in self.generations_per_weight_version.keys() if v < min_valid_version]
        # for old_version in old_versions:
        #     del self.generations_per_weight_version[old_version]

        # # Resume collection if it was paused due to generation limits
        was_paused = not self._generation_limit_cleared.is_set()
        if was_paused:
            self._generation_limit_cleared.set()  # Signal that collection can resume
            print(f"🔄 Updated weight version to {version}, resuming collection")
        else:
            print(f"🔄 Updated weight version to {version}")

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits."""
        num_prompts_per_step = self.master_config["grpo"]["num_prompts_per_step"]
        max_age_steps = self.master_config["async_grpo"]["max_trajectory_age_steps"]
        max_generations_per_version = num_prompts_per_step * max_age_steps

        current_count = self.generations_per_weight_version.get(
            self.current_weight_version, 0
        )
        return current_count >= max_generations_per_version

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader
        print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                # Check if manually paused and wait
                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                # Check if generation limits require pausing collection
                if self._should_pause_for_generation_limits() and self.running:
                    # Only log warning once per weight version
                    if self._last_limit_warning_version != self.current_weight_version:
                        current_count = self.generations_per_weight_version.get(
                            self.current_weight_version, 0
                        )
                        num_prompts_per_step = self.master_config["grpo"][
                            "num_prompts_per_step"
                        ]
                        max_age_steps = self.master_config["async_grpo"][
                            "max_trajectory_age_steps"
                        ]
                        max_generations = num_prompts_per_step * max_age_steps

                        print(
                            f"⏸️ Pausing collection: weight version {self.current_weight_version} reached "
                            f"generation limit ({current_count}/{max_generations}). "
                            f"Waiting for weight update..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                        self._generation_limit_cleared.clear()  # Clear the event to pause

                    # Efficiently wait for generation limits to be cleared (no polling!)
                    self._generation_limit_cleared.wait()

                    # Double-check we're still running after being woken up
                    if not self.running:
                        break

                if not self.running:
                    break

                self._process_batch(batch)

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and add per-prompt groups to replay buffer.

        This function handles only 1 prompt * num_generations_per_prompt at a time.
        """
        try:
            generation_weight_version = self.current_weight_version

            # For each prompt in the incoming batch, build a per-prompt group by
            # repeating it num_generations_per_prompt times, run rollout, then push.
            num_prompts = batch.size
            num_generations = self.master_config["grpo"]["num_generations_per_prompt"]

            # Track current generation count for this weight version
            current_count = self.generations_per_weight_version.get(
                generation_weight_version, 0
            )

            for prompt_idx in range(num_prompts):
                single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                repeated_batch = single_prompt_batch.repeat_interleave(num_generations)

                # Increment generation count for this weight version
                self.generations_per_weight_version[generation_weight_version] = (
                    current_count + 1
                )
                current_count += 1

                # Prevent flooding generator: at most num_prompts_per_step in-flight
                self._inflight_sema.acquire()
                worker = _threading.Thread(
                    target=self._run_prompt_group_worker,
                    args=(
                        repeated_batch,
                        generation_weight_version,
                        prompt_idx,
                    ),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                worker.start()
                # Opportunistically reap finished threads
                self._cleanup_finished_threads()

            # Log generation status for this weight version (less verbose)
            final_count = self.generations_per_weight_version.get(
                generation_weight_version, 0
            )
            if (
                final_count % 10 == 0 or final_count == 1
            ):  # Log every 10th generation or first one
                num_prompts_per_step = self.master_config["grpo"][
                    "num_prompts_per_step"
                ]
                max_age_steps = self.master_config["async_grpo"][
                    "max_trajectory_age_steps"
                ]
                max_generations = num_prompts_per_step * max_age_steps
                print(
                    f"📊 Weight version {generation_weight_version}: {final_count}/{max_generations} generations"
                )

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> None:
        """Pause trajectory collection."""
        self._manual_pause_cleared.clear()  # Signal collection to pause
        print("Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()  # Signal collection to resume
        print("Trajectory collection resumed")

    def stop(self) -> None:
        """Stop trajectory collection."""
        self.running = False
        # Signal all events to wake up any waiting threads so they can exit cleanly
        self._manual_pause_cleared.set()
        self._generation_limit_cleared.set()

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        prompt_idx: int,
    ) -> None:
        try:
            # Run rollout for this prompt group
            if self._use_async_rollouts:
                # Async engine supports concurrent generation; avoid locking
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config["policy"][
                        "max_total_sequence_length"
                    ],
                    max_rollout_turns=self.master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
            else:
                # Fallback to sync rollout; serialize access to generation
                with self._pg_lock:
                    final_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=self.policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=self.tokenizer,
                        task_to_env=self.task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            # Use exponential backoff when buffer is full
            try:
                backoff_delay = 0.01
                while self.running:
                    status = ray.get(
                        self.replay_buffer.push_with_wait_signal.remote(
                            trajectory_group, generation_weight_version
                        )
                    )
                    if status == "success":
                        print(f"📦 Buffered per-prompt group (prompt_idx {prompt_idx})")
                        break
                    elif status == "full":
                        # Exponential backoff up to 1 second
                        time.sleep(min(backoff_delay, 1.0))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Detach thread record when finished
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                pass
