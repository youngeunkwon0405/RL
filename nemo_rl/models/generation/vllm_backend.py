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
from typing import Any

import torch

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
    )


class VllmInternalWorkerExtension:
    def init_collective(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        rank = rank_prefix + local_rank + 1  # 1 is the head node of the train cluster

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(pg, device=self.device)

    def report_device_id(self) -> str:
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def kitchen_block_scale(
        self,
        data_hp,
        param_name,
        weight_block_size,
    ):
        if not len(data_hp.shape) == 2:
            import pdb; pdb.set_trace()
        assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

        block_size1 = weight_block_size[1]
        block_size0 = weight_block_size[0]
        assert (
            data_hp.shape[1] % block_size1 == 0
        ), f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
        assert (
            data_hp.shape[0] % block_size0 == 0
        ), f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."

        # FP8
        max_dtype = torch.finfo(torch.float8_e4m3fn).max
  
        original_shape = data_hp.shape
        blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    
        assert block_size1 == block_size0
        data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

        # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
        data_hp = data_hp.permute(0, 2, 1, 3)
        # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
        data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

        # Calculate max absolute value per block
        max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
        # Calculate descale factor
        descale = max_abs / max_dtype
        if False:
            # Calculate exponent HW instruction: cvt.rp.satfinite.ue8m0x2.f32
            exponent = torch.ceil(torch.log2(descale))
            # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
            exponent = torch.clamp(exponent, min=-127, max=127) + 127
            # Convert to uint8 container
            exponent = exponent.to(torch.uint8)
            # Calculate descale_fp to apply to data_hp
            descale_fp = torch.where(
                # If exponent is 0, descale_fp is 1.0 rather than 2^127
                exponent == 0,
                1.0,
                torch.exp2(127 - exponent.to(torch.float32)),
            )
        else:
            descale_fp = max_dtype / max_abs
            descale_fp = torch.where(
                max_abs == 0, 1.0, descale_fp
            )  # preserve the behavior for 0 amax case
            descale_fp = torch.where(
                descale_fp == torch.inf, torch.finfo(data_hp.dtype).max, descale_fp
            )
            exponent = torch.reciprocal(descale_fp)

        # Scale and saturate cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_dtype, max=max_dtype)

        fp_data = data_lp.to(torch.float8_e4m3fn)

        # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
        fp_data = (
            fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
            .permute(0, 2, 1, 3)
            .reshape(original_shape)
        )
   
        # Convert to target format, but still in original precision container
        return fp_data, exponent 


    def is_fp8_weight(self, name):
        fp8_params = [
            "q_proj.weight", 
            "k_proj.weight", 
            "v_proj.weight", 
            "o_proj.weight", 
            "down_proj.weight", 
            "up_proj.weight", 
            "gate_proj.weight"
        ]
        
        return any([param in name for param in fp8_params])

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update weights from IPC handles.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated.
        """

        try:
            with torch.no_grad():
                # Get handles for this device
                device_uuid = self.report_device_id()
                handles = ipc_handles[device_uuid]
                device_id = self.device.index
                weights = []

                # Process each handle to get the tensor
                for name, handle in handles:
                    func, args = handle
                    list_args = list(args)
                    # Update device ID to match the current device
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    weights.append((name, tensor))

                from vllm.model_executor.layers.quantization.fp8 import Fp8Config
                if hasattr(self.model_runner.model, "quant_config"):
                    quant_config = self.model_runner.model.quant_config
                    if isinstance(quant_config, Fp8Config):
                        assert quant_config.weight_block_size is not None, \
                            "Only block scaling is currently supported in NeMo-RL!"

                        weights_quantized = []
                        for k, v in weights:
                            if not self.is_fp8_weight(k):
                                weights_quantized.append((k,v))
                                continue

                            lp, scale = self.kitchen_block_scale(
                                v.to(torch.float), 
                                param_name=k, 
                                weight_block_size=quant_config.weight_block_size
                            )
                            scale = torch.squeeze(scale)
                            weights_quantized.append([k, lp])
                            weights_quantized.append([k + "_scale_inv", scale])

                        weights = weights_quantized

                for name, param in self.model_runner.model.named_parameters():
                    if hasattr(param, "subclass_type"):
                        param.orig_type = param.__class__
                        param.__class__ = param.subclass_type
                        
                for name, weight in weights:
                    print(f"UPDATING {name}")
                    self.model_runner.model.load_weights([[name, weight]])

                for name, param in self.model_runner.model.named_parameters():
                    if hasattr(param, "subclass_type"):
                        param.__class__ = param.orig_type

                torch.cuda.synchronize()
                return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_ipc_handles: {e}"
            )
            return False

    def update_weights_from_collective(self, info: dict[str, Any]) -> bool:
        """Update the model weights from collective communication."""
        try:
            for name, (shape, dtype) in info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(weight, src=0)
                self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True
