
import functools
import json
import os
from typing import Any, Callable, Optional, Union

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_dequantize)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op

def process_weights_after_loading(self, layer) -> None:

    from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                    ModelWeightParameter,
                                    PerTensorScaleParameter)
    from torch.nn import Parameter
    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_param_dir 
            if attr not in base_param_dir and not attr.startswith('__')
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    weight = layer.weight.data
    weight_scale_inv = layer.weight_scale_inv.data
    weight = self._maybe_pad_weight(weight)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(data=weight, output_dim=0, input_dim=1,
            weight_loader=layer.weight.weight_loader)
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv, output_dim=0, input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader)
    )


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    pow2_scale: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)

    if pow2_scale:
        inv_scale = fp8_max / _absmax
        exponent = tl.floor(tl.log2(inv_scale))
        # exponent is an integer
        exponent = tl.minimum(exponent, 126.0)

        # after rounding to exponent, round back to floating
        inv_scale_pow2 = tl.exp2(exponent)
        
        is_nan = inv_scale_pow2 != inv_scale_pow2
        is_inf = (inv_scale_pow2 == 1.0 / 0.0) | (inv_scale_pow2 == -1.0 / 0.0)
        
        # If the value is NaN or infinity, default it to 1.0,
        # otherwise keep its original value.
        inv_scale_pow2 = tl.where(is_nan | is_inf, 1.0, inv_scale_pow2) 
        # finally uninverse        
        y_s = 1.0 / inv_scale_pow2

    else:
        # Original scaling logic
        y_s = _absmax / fp8_max

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    pow2_scale: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)

    # Quant
    if pow2_scale:
        inv_scale = fp8_max / _absmax
        # calculate the nearest pow2 integer
        exponent = tl.floor(tl.log2(inv_scale))
        exponent = tl.minimum(exponent, 126.0)
        # round inv_scale to the nearest pow2 with the exp we just calculated
        inv_scale_pow2 = tl.exp2(exponent)
        # If the value is NaN or infinity, default it to 1.0,
        # otherwise keep its original value.
        is_nan = inv_scale_pow2 != inv_scale_pow2
        is_inf = (inv_scale_pow2 == float('inf')) | (inv_scale_pow2 == float('-inf'))
        inv_scale_pow2 = tl.where(is_nan | is_inf, 1.0, inv_scale_pow2) 
        # finally uninverse        
        y_s = 1.0 / inv_scale_pow2
    else:
        y_s = _absmax / fp8_max

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    pow2_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert (x.shape[-1] % group_size == 0), (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    pow2_scale = True

    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            pow2_scale=pow2_scale,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            pow2_scale=pow2_scale,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s
