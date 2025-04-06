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
import gc
import warnings
import os
from copy import deepcopy
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import ray
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
    FSDPModule,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.utils import import_class_from_path
from nemo_reinforcer.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)

from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
    SequenceParallel,
)
import random
import numpy as np
from nemo_reinforcer.models.policy.dtensor_policy_worker import DTensorPolicyWorker


class HfPolicy(PolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        name_prefix: str = "hf_policy",
        workers_per_node: Optional[Union[int, List[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_reference_model: bool = True,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)
        self.cfg = config

        node_bundle_indices = None

        if config["tensor_parallel_size"] > 1:
            worker_builder_cls = DTensorPolicyWorker
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)
        else:
            # TODO: add back hfpolicy worker
            worker_builder_cls = HfPolicyWorker

        worker_builder = RayWorkerBuilder(
            worker_builder_cls,
            config,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
        )

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            bundle_indices_list=node_bundle_indices,
        )
        self.dp_size = self.worker_group.world_size // config["tensor_parallel_size"]

    def _get_tied_worker_bundle_indices(self, cluster):
        """Calculate bundle indices for tensor parallel workers."""
        # Get the placement groups (nodes) from the cluster
        placement_groups = cluster.get_placement_groups()

        tied_worker_groups = []

        # For each node (placement group), create tied worker groups of size tensor_parallel_size
        for node_idx, pg in enumerate(placement_groups):
            # How many bundles (GPUs) are on this node
            bundles_on_node = pg.bundle_count
            tied_worker_groups_on_node = (
                bundles_on_node // self.cfg["tensor_parallel_size"]
            )

            if tied_worker_groups_on_node > 0:
                for group_idx in range(tied_worker_groups_on_node):
                    # Local bundle indices for this tied worker group (consecutive GPUs on this node)
                    start_idx = group_idx * self.cfg["tensor_parallel_size"]
                    end_idx = start_idx + self.cfg["tensor_parallel_size"]
                    local_bundle_indices = list(range(start_idx, end_idx))
                    tied_worker_groups.append((node_idx, local_bundle_indices))

        if not tied_worker_groups:
            raise ValueError(
                f"Cannot create any tensor parallel tied worker groups with size {self.tensor_parallel_size}. "
                f"Make sure each node has at least {self.tensor_parallel_size} GPUs."
            )

        return tied_worker_groups

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "get_logprobs", sharded_data, run_even_if_tp=True
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )
        return logprobs

    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "get_reference_policy_logprobs", sharded_data, run_even_if_tp=True
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )
        return logprobs

    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ):
        """Train the policy on a batch of data with a given loss function."""
        # Shard and replicate the batch
        shards = self.dp_size
        sharded_data = data.shard_by_batch_size(
            shards, batch_size=self.cfg["train_global_batch_size"]
        )

        # Train each shard in parallel
        futures = self.worker_group.run_all_workers_multiple_data(
            "train",
            sharded_data,
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "gbs": gbs,
                "mbs": mbs,
            },
            run_even_if_tp=True,
        )
        results = self.worker_group.get_all_worker_results(futures)

        # Aggregate the results
        aggregated_results = {}
        aggregated_results["loss"] = results[0]["global_loss"]

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

        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "generate",
            sharded_data,
            common_kwargs={"greedy": greedy},
            run_even_if_tp=True,
        )
        result = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
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

    def prepare_for_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def prepare_for_training(self, *args, **kwargs):
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_training", respect_tied_workers=True, run_even_if_tp=True
        )
        ray.get(futures)
        pass

    def prepare_for_lp_inference(self, *args, **kwargs):
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", respect_tied_workers=True, run_even_if_tp=True
        )
        ray.get(futures)

    def finish_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def finish_training(self, *args, **kwargs):
        # Placeholder implementation
        pass

    def get_weights_ipc_handles(self):
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles = ray.get(
            [
                worker.get_weight_ipc_handles.remote()
                for worker in self.worker_group.workers
            ]
        )

        # Combine all worker handles into a single dictionary
        all_handles = {}
        for handle in worker_handles:
            all_handles.update(handle)

        return all_handles

    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_before_refit", respect_tied_workers=True, run_even_if_tp=True
        )
        ray.get(futures)

    def offload_after_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_after_refit", respect_tied_workers=True, run_even_if_tp=True
        )
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path,
            optimizer_path,
            offload_to_cpu=offload_to_cpu,
            respect_tied_workers=True,
            run_even_if_tp=True,
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

    def __del__(self):
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        self.worker_group.shutdown()
