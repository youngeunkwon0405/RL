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

import einops
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
)
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations.accelerate import init_empty_weights
from nemo.lightning.io.state import TransformCTX, _ModelState, StateDictTransform

import nemo_rl.models.megatron.converters.qwen2 as qwen2_converter

_GROUP_TO_RANKS_CACHE = {}


def get_all_rank_ids_in_group(group):
    """Get all rank ids in a group."""
    if group in _GROUP_TO_RANKS_CACHE:
        return _GROUP_TO_RANKS_CACHE[group]

    curr_global_rank = int(torch.distributed.get_rank())
    group_size = torch.distributed.get_world_size(group=group)
    global_rank_tensor = torch.tensor(
        [curr_global_rank], dtype=torch.int, device=torch.cuda.current_device()
    )
    global_ranks = [
        torch.empty(1, dtype=torch.int, device=torch.cuda.current_device())
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(global_ranks, global_rank_tensor, group=group)
    _GROUP_TO_RANKS_CACHE[group] = [
        int(global_ranks[i].item()) for i in range(group_size)
    ]
    return _GROUP_TO_RANKS_CACHE[group]


def get_local_layer_num(s):
    """Assumes layer number is preceeded by 'layers.'."""
    segments = s.split(".")
    number = None
    for i, segment in enumerate(segments):
        if segment == "layers":
            if segments[i + 1].isdigit():
                number = int(segments[i + 1])
                break
    return number


def get_global_layer_num(s, cfg):
    """Assumes layer number is preceeded by 'layers.'.

    Assumes pipeline model parallel size is set.
    In the state dict, the layer number is the local layer number (PP local).
    This function converts the local layer number to the global layer number.
    """
    local_layer_num = get_local_layer_num(s)
    global_layer_num = (
        parallel_state.get_pipeline_model_parallel_rank()
        * cfg.num_layers
        // parallel_state.get_pipeline_model_parallel_world_size()
        + local_layer_num
    )
    return global_layer_num


def get_tp_dim(model, param_name, named_modules_dict):
    # pass in named_modules_dict so we can get it ahead of time instead
    # of once for each param
    if not param_name.endswith(".weight") and not param_name.endswith(".bias"):
        return None

    prefix = ""
    if hasattr(model, "module"):
        prefix = "module."
        if hasattr(model.module, "module"):
            prefix = "module.module."
    key = prefix + ".".join(param_name.split(".")[:-1])
    module = named_modules_dict.get(key)
    if module is None:
        print(f"Module {key} not found in named_modules_dict")
        return None
    if hasattr(module, "parallel_mode"):
        # TE layers have parallel_mode we can check directly
        if module.parallel_mode == "column":
            return 0
        elif module.parallel_mode == "row":
            return 1
        else:
            return None
    elif isinstance(module, VocabParallelEmbedding) or isinstance(
        module, ColumnParallelLinear
    ):
        return 0
    elif isinstance(module, RowParallelLinear):
        return 1
    # TODO(yifu): moe layers
    else:
        return None


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def split_fc1_tp(ctx: TransformCTX, linear_fc1: torch.Tensor):
    # gate proj and up proj are mixed right now, and we need to reshape them
    # [ gate_tp0 ]     [ gate_tp0 ]
    # [  up_tp0  ] --\ [ gate_tp1 ] --\ (split gate)
    # [ gate_tp1 ] --/ [  up_tp0  ] --/ (split  up)
    # [  up_tp1  ]     [  up_tp1  ]
    megatron_config = ctx.source.config
    tp = megatron_config.tensor_model_parallel_size
    linear_fc1 = einops.rearrange(linear_fc1, "(t c d) a1 ->  c (t d) a1", c=2, t=tp)
    mlp_gate_proj_weight = linear_fc1[0]
    mlp_up_proj_weight = linear_fc1[1]
    return mlp_gate_proj_weight, mlp_up_proj_weight

def split_qkv_gpu(ctx: TransformCTX, linear_qkv: torch.Tensor):
    """
    Split interleave-concatenated qkv to q, k, v

    Example: export layer linear_qkv to HF {q|k|v}_proj
    """
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    # hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    # when converting base model (linear_qkv), hidden size = megatron_config.hidden_size
    # when converting lora (linear_qkv.adapter.linear_out), hidden size = lora_r
    hidden_size = linear_qkv.size(-1)
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size)
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size)
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size)

    return q_proj, k_proj, v_proj

def split_qkv_bias_gpu(ctx: TransformCTX, qkv_bias: torch.Tensor):
    """
    Split interleave-concatenated qkv bias to separate q, k, v bias

    Example: export layer linear_qkv bias to HF {q|k|v}_proj bias
    """
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_bias = qkv_bias[q_slice].reshape(-1)
    k_bias = qkv_bias[k_slice].reshape(-1)
    v_bias = qkv_bias[v_slice].reshape(-1)

    return q_bias, k_bias, v_bias

def update_transforms_for_nemorl(export_transforms):
    # In place update
    for transform in export_transforms:
        if transform.transform.__name__ == "split_fc1":
            # Need to modify this transform to take into account the TP size
            transform.transform = split_fc1_tp
        elif transform.transform.__name__ == "split_qkv":
            transform.transform = split_qkv_gpu
        elif transform.transform.__name__ == "split_qkv_bias":
            transform.transform = split_qkv_bias_gpu
    return export_transforms


class MegatronToHFConverter:
    def __init__(self, hf_model_name):
        # We only care about the state_dict keys and the config, so we
        # don't need to load the model weights
        config = AutoConfig.from_pretrained(hf_model_name)
        with init_empty_weights():
            self.target_model = AutoModelForCausalLM.from_config(config)
        if "qwen" in hf_model_name.lower():
            self.export_mapping = qwen2_converter.get_export_mapping()
            self.export_transforms = qwen2_converter.get_export_transforms()

    def _get_empty_state_dict(self):
        state_dict = {}
        for k in self.target_model.state_dict().keys():
            state_dict[k] = None
        return state_dict

    def convert(self, state_dict, megatron_config):
        export_transforms = update_transforms_for_nemorl(self.export_transforms)
        source = _ModelState(state_dict)
        source.config = megatron_config
        ctx = TransformCTX(
            source=source,
            source_state=state_dict,
            target=self.target_model,
            target_state=self._get_empty_state_dict(),
        )
        for key, val in self.export_mapping.items():
            ctx = StateDictTransform(key, val)(ctx)
        for transform in export_transforms:
            ctx = transform(ctx)

        converted_state_dict = ctx.target_state
        return converted_state_dict
