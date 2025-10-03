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
from collections import defaultdict
from typing import Any, Optional

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from nemo_rl.utils.nsys import wrap_with_nvtx_name

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


class VllmInternalWorkerExtension:
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            pg, device=self.device
        )

    def report_device_id(self) -> str:
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]] = None
    ) -> None:
        """Prepare the info for refit.

        DtensorPolicyWorker:
            colocated inference: state_dict_info is None
            non-colocated inference: state_dict_info is a dict of {tensor_name: (shape, dtype)}

        MegatronPolicyWorker:
            colocated inference: state_dict_info is a dict of {tensor_name: (shape, dtype, numel)}
            non-colocated inference: state_dict_info is a dict of {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_global_ipc_handles"
    )
    def update_weights_from_global_ipc_handles(self, global_device_ipc_handles):
        """Update weights from global IPC handles.

        Args:
            global_device_ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated.
        """
        device_uuid = self.report_device_id()
        local_device_ipc_handles = global_device_ipc_handles[device_uuid]
        return self.update_weights_from_local_ipc_handles(local_device_ipc_handles)

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_local_ipc_handles"
    )
    def update_weights_from_local_ipc_handles(self, local_device_ipc_handles):
        """Update weights from local IPC handles.

        Args:
            local_device_ipc_handles (dict): parameter IPC handles for local device.

        Returns:
            bool: True if weights were successfully updated.
        """
        try:
            is_tensor_packed = local_device_ipc_handles[0]
            if is_tensor_packed:
                _, all_handles, list_keys = local_device_ipc_handles
            else:
                _, name_and_handle_list = local_device_ipc_handles

            device_id = self.device.index
            weights = []

            if is_tensor_packed:
                assert self.state_dict_info is not None, (
                    "state_dict_info is not prepared. "
                    "Please call prepare_refit_info when initializing the worker."
                )

                # Extract packed tensor from IPC handle
                dtype_to_packed_tensor = {}
                for dtype, tensor_handle in all_handles:
                    func = rebuild_cuda_tensor
                    args = tensor_handle[0]
                    list_args = list(args)
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    dtype_to_packed_tensor[dtype] = tensor

                weights = []
                dtype_to_offset = defaultdict(lambda: 0)
                for key in list_keys:
                    shape, dtype, size = self.state_dict_info[key]
                    weights.append(
                        (
                            key,
                            dtype_to_packed_tensor[dtype][
                                dtype_to_offset[dtype] : dtype_to_offset[dtype] + size
                            ].view(*shape),
                        )
                    )
                    dtype_to_offset[dtype] += size

                expected_sizes = {
                    dtype: tensor.numel()
                    for dtype, tensor in dtype_to_packed_tensor.items()
                }
                assert dtype_to_offset == expected_sizes, (
                    f"Packed tensor size mismatch: expected sizes from keys list {expected_sizes} != actual packed tensor sizes {dtype_to_offset}. "
                    f"This indicates the keys list order doesn't match the order used when packing tensors."
                )
            else:
                # Process each handle to get the tensor
                for name, handle in name_and_handle_list:
                    func = rebuild_cuda_tensor
                    args = handle[0]
                    list_args = list(args)
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    weights.append((name, tensor))

            # Load weights into the model
            from nemo_rl.models.generation import fp8

            if fp8.is_fp8_model(self.model_runner.vllm_config):
                # the fp8 load_weights additionally casts bf16 weights into fp8
                fp8.load_weights(weights, self.model_runner)
            else:
                self.model_runner.model.load_weights(weights=weights)

            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_ipc_handles: {e}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        try:
            for name, (shape, dtype) in self.state_dict_info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(weight, src=0)

                from nemo_rl.models.generation import fp8

                if fp8.is_fp8_model(self.model_runner.vllm_config):
                    # the fp8 load_weights additionally casts bf16 weights into fp8
                    fp8.load_weights([(name, weight)], self.model_runner)
                else:
                    self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
