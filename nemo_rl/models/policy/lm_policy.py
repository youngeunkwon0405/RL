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
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import ray
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SlicedDataDict,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)

PathLike = Union[str, "os.PathLike[Any]"]


class Policy(ColocatablePolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        name_prefix: str = "lm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[PathLike] = None,
        optimizer_path: Optional[PathLike] = None,
        init_reference_model: bool = True,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        node_bundle_indices = None
        tp_size = 1
        cp_size = 1

        worker_builder_cls: str
        if config["dtensor_cfg"]["enabled"]:
            worker_builder_cls = (
                "nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker"
            )
            tp_size = config["dtensor_cfg"]["tensor_parallel_size"]
            cp_size = config["dtensor_cfg"]["context_parallel_size"]
        else:
            worker_builder_cls = (
                "nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker"
            )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                -1,  # DP
                cp_size,  # CP
                tp_size,  # TP
            ),
            names=["data_parallel", "context_parallel", "tensor_parallel"],
        )

        worker_builder = RayWorkerBuilder(
            worker_builder_cls,
            config,
            tokenizer=tokenizer,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
        )

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            workers_per_node=workers_per_node,
            sharding_annotations=self.sharding_annotations,
        )

        if config["dynamic_batching"]["enabled"]:
            assert config["dtensor_cfg"]["enabled"], (
                "Dynamic batch is only supported for DTensor policy."
            )
            self.use_dynamic_batches = True
            self.dynamic_batching_args: DynamicBatchingArgs = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": config["dynamic_batching"][
                    "sequence_length_round"
                ],
                "max_tokens_per_microbatch": 0,  # Override this in each different call (presumably different sizes)
            }
        else:
            self.use_dynamic_batches = False

        self.cfg = config

    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "init_collective", ip=ip, port=port, world_size=world_size
        )
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        cp_size = self.sharding_annotations.get_axis_size("context_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]

        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                cp_size * dp_size,
                batch_size=None,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(  # type: ignore
                cp_size * dp_size,
                batch_size=None,
            )

        sharded_data_2d = []
        shard_idx = 0
        # Convert to 2d dim array
        for _ in range(dp_size):
            cp_data = []
            for _ in range(cp_size):
                cp_data.append(sharded_data[shard_idx])
                shard_idx += 1
            sharded_data_2d.append(cp_data)

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
            sharded_data_2d,
            in_sharded_axes=["data_parallel", "context_parallel"],
            replicate_on_axes=["tensor_parallel"],
            output_is_replicated=["tensor_parallel"],
        )
        logprobs: BatchedDataDict[LogprobOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        cp_size = self.sharding_annotations.get_axis_size("context_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]
        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                cp_size * dp_size,
                batch_size=None,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(  # type: ignore
                cp_size * dp_size,
                batch_size=None,
            )

        sharded_data_2d = []
        shard_idx = 0
        # Convert to 2d dim array
        for _ in range(dp_size):
            cp_data = []
            for _ in range(cp_size):
                cp_data.append(sharded_data[shard_idx])
                shard_idx += 1
            sharded_data_2d.append(cp_data)

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs",
            sharded_data_2d,
            in_sharded_axes=["data_parallel", "context_parallel"],
            replicate_on_axes=["tensor_parallel"],
            output_is_replicated=["tensor_parallel"],
            common_kwargs={"micro_batch_size": micro_batch_size},
        )
        logprobs: BatchedDataDict[ReferenceLogprobOutputSpec] = (
            BatchedDataDict.from_batches(
                self.worker_group.get_all_worker_results(futures)
            )
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        # Shard and replicate the batch
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["train_mb_tokens"]
            sharded_data, _ = data.shard_by_batch_size(
                dp_size,
                batch_size=batch_size,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(
                dp_size,
                batch_size=batch_size,
            )

        # Train each shard in parallel
        futures = self.worker_group.run_all_workers_sharded_data(
            "train",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel"],
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "gbs": batch_size,
                "mbs": micro_batch_size,
            },
        )
        results = self.worker_group.get_all_worker_results(futures)

        # Aggregate the results
        aggregated_results = {
            "loss": results[0]["global_loss"],
            "grad_norm": results[0]["grad_norm"],
        }

        # Aggregate metrics across all workers
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated_results

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        # Verify input data is right-padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "generate",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["tensor_parallel"],
            output_is_replicated=["tensor_parallel"],
            common_kwargs={"greedy": greedy},
        )
        assert self.cfg["generation"] is not None, "Generation config is not set"
        result: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["pad_token_id"]},
        )

        # Verify the output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data("prepare_for_training")
        ray.get(futures)

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference"
        )
        ray.get(futures)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        # Placeholder implementation
        pass

    def prepare_weights_for_ipc(
        self, _refit_buffer_size_gb: Optional[int] = None
    ) -> list[list[str]]:
        """Prepare the weights for IPC.

        Returns:
            list: A list containing the keys of the parameters, which is grouped by size.
        """
        # Get the state_dict_info and available memory from all workers
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_weights_for_ipc"
        )
        results = ray.get(futures)

        # Only get the first worker's state_dict_info since all workers will have the same result
        state_dict_info = results[0][0]

        if _refit_buffer_size_gb is not None:
            total_available_bytes = _refit_buffer_size_gb * (1024**3)
        else:
            # Get the minimum available memory from all workers
            total_available_bytes = min(result[1] for result in results)

        # Group tensors by size
        cur_available_bytes = total_available_bytes
        grouped_param_keys: list[list[str]] = []
        keys: list[str] = []

        for key, size_in_bytes in state_dict_info:
            if size_in_bytes > cur_available_bytes:
                if keys:
                    grouped_param_keys.append(keys)
                    keys = []
                cur_available_bytes = total_available_bytes

            keys.append(key)
            cur_available_bytes -= size_in_bytes

        if keys:
            grouped_param_keys.append(keys)

        return grouped_param_keys

    def get_weights_ipc_handles(self, keys: list[str]) -> dict[str, Any]:
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles: list[dict[str, Any]] = ray.get(
            [
                worker.get_weights_ipc_handles.remote(keys)
                for worker in self.worker_group.workers
            ]
        )

        # Combine all worker handles into a single dictionary
        all_handles = {}
        for handle in worker_handles:
            all_handles.update(handle)

        return all_handles

    def prepare_info_for_collective(self) -> dict[str, Any]:
        """Prepare the info for collective communication.

        Returns:
            dict: A dictionary containing the info for collective communication.
        """
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_info_for_collective"
        )
        results = ray.get(futures)
        # Only get the first worker's info since all workers will have the same result
        return results[0]

    def broadcast_weights_for_collective(self) -> list[ray.ObjectRef]:
        """Broadcast the weights for collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "broadcast_weights_for_collective"
        )
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    def offload_before_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_before_refit")
        ray.get(futures)

    def offload_after_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_after_refit")
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path,
            optimizer_path,
            tokenizer_path,
        )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all HF workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        self.worker_group.shutdown()
