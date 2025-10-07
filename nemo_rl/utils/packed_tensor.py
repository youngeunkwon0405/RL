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
from typing import Any, List, Tuple

import torch


def get_target_packed_tensor_size():
    packed_tensor_bucket_size = os.getenv("NRL_PACKED_TENSOR_SIZE_TARGET_IN_MB", "500")
    return int(packed_tensor_bucket_size) * 1024 * 1024


def pack_tensor(packed_tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Pack a list of tensors into a single tensor."""
    # Perform batched concatenation with torch.cat
    return torch.cat(
        [tensor.view(torch.uint8).view(-1) for _, tensor in packed_tensor_list], dim=0
    )


def unpack_tensor(
    packed_tensor: torch.Tensor, meta_data_list: list[Any]
) -> List[Tuple[str, torch.Tensor]]:
    """Unpack a single tensor into a list of tensors.

    Args:
        packed_tensor: the packed torch.uint8 tensor to unpack
        meta_data_list: List[(name, shape, dtype, offset, tensor_size)]

    Returns:
        unpacked List[(name, tensor)]
    """
    unpacked_list = []
    # Perform batched split with torch.split_with_sizes
    packed_tensor_sizes = [tensor_size for _, _, _, _, tensor_size in meta_data_list]
    unpacked_tensor = packed_tensor.split_with_sizes(packed_tensor_sizes)

    for i, tensor in enumerate(unpacked_tensor):
        # unpacked_list = List[(name, torch.Tensor.view(dtype).view(*shape))]
        unpacked_list.append(
            (
                meta_data_list[i][0],
                tensor.view(meta_data_list[i][2]).view(*meta_data_list[i][1]),
            )
        )

    return unpacked_list
