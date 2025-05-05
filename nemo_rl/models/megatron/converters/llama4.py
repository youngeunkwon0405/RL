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

import torch
import einops

from megatron.core import parallel_state

_GROUP_TO_RANKS_CACHE = {}


def split_qkv_llama(gathered_mcore_qkv_layer, cfg):
    hidden_size = cfg.hidden_size
    head_num = cfg.num_attention_heads
    num_query_groups = (
        cfg.num_query_groups if hasattr(cfg, 'num_query_groups') and cfg.num_query_groups is not None else head_num
    ) # different num_query_groups for 70B / Llama 3

    head_size = cfg.kv_channels if hasattr(cfg, 'kv_channels') and cfg.kv_channels is not None else (
        hidden_size // head_num
    ) # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_weights = gathered_mcore_qkv_layer
    # qkv_weights shape: [qkv_total_dim * head_size, hidden_size] (for TP=0)
    # Reshape to: [qkv_total_dim, head_size, hidden_size]
    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_name = "model.layers.{gl}.self_attn.q_proj.weight"
    k_name = "model.layers.{gl}.self_attn.k_proj.weight"
    v_name = "model.layers.{gl}.self_attn.v_proj.weight"
    # Extract Q, K, V and reshape back to [num_heads * head_size, hidden_size]
    q = qkv_weights[q_slice].reshape(-1, hidden_size)
    k = qkv_weights[k_slice].reshape(-1, hidden_size)
    v = qkv_weights[v_slice].reshape(-1, hidden_size)

    return {q_name: q, k_name: k, v_name: v}


def split_fc1_gate_down_llama_common(gathered_mcore_fc1, cfg, gate_proj_name, up_proj_name):
    """Common function to split interleaved gate/up projections."""
    # Input shape: [intermediate_size * 2 // tp_size, hidden_size]
    tp = cfg.tensor_model_parallel_size
    # Rearrange to split gate and up: 2, [intermediate_size // tp_size, hidden_size]
    # Then combine tp dim back: 2, [intermediate_size, hidden_size]
    gathered_mcore_fc1 = einops.rearrange(
        gathered_mcore_fc1, "(t c d) a1 ->  c (t d) a1", c=2, t=tp
    )
    mlp_gate_proj_weight = gathered_mcore_fc1[0]
    mlp_up_proj_weight = gathered_mcore_fc1[1]
    return {
        up_proj_name: mlp_up_proj_weight,
        gate_proj_name: mlp_gate_proj_weight,
    }

def split_fc1_gate_down_llama4_dense(gathered_mcore_fc1, cfg):
    # For dense layers in Llama4
    gate_proj_name = "model.layers.{gl}.feed_forward.gate_proj.weight"
    up_proj_name = "model.layers.{gl}.feed_forward.up_proj.weight"
    return split_fc1_gate_down_llama_common(gathered_mcore_fc1, cfg, gate_proj_name, up_proj_name)

def split_shared_fc1_llama4(gathered_mcore_fc1, cfg):
    # For the shared expert in Llama4 MoE layers
    gate_proj_name = "model.layers.{gl}.feed_forward.shared_expert.gate_proj.weight"
    up_proj_name = "model.layers.{gl}.feed_forward.shared_expert.up_proj.weight"
    return split_fc1_gate_down_llama_common(gathered_mcore_fc1, cfg, gate_proj_name, up_proj_name)

def transpose_expert_weight(gathered_expert_weight, cfg, target_name):
    """Transpose expert weights from (e, c, h) or (e, h, c) depending on TP."""
    # NeMo Megatron stores expert weights potentially differently based on TP/EP strategy.
    # Common format is (num_experts, intermediate_size_per_expert, hidden_size) for fc1 (TP=0)
    # and (num_experts, hidden_size, intermediate_size_per_expert) for fc2 (TP=1).
    # The exporter code transposes with permute(0, 2, 1). Let's replicate that.
    # We assume the input 'gathered_expert_weight' is the fully gathered weight across TP.
    transposed_weight = gathered_expert_weight.permute(0, 2, 1).contiguous()
    return {target_name: transposed_weight}

def transpose_expert_fc1(gathered_expert_weight, cfg):
    # Target name for Llama4 HF MoE FC1 (gate_up_proj combines gate and up)
    target_name = "model.layers.{gl}.feed_forward.experts.gate_up_proj"
    return transpose_expert_weight(gathered_expert_weight, cfg, target_name)

def transpose_expert_fc2(gathered_expert_weight, cfg):
    # Target name for Llama4 HF MoE FC2 (down_proj)
    target_name = "model.layers.{gl}.feed_forward.experts.down_proj"
    return transpose_expert_weight(gathered_expert_weight, cfg, target_name)


# Mapping for Llama4 (including MoE layers)
# Assumes the checkpoint being converted is from a Llama4 model architecture
mcore_te_to_hf_llama4 = {
    # Embeddings
    "embedding.word_embeddings.weight": {"tp": 0, "hf": "model.embed_tokens.weight"}, # Needs vocab pruning if nemo vocab > hf vocab
    "output_layer.weight": {"tp": 0, "hf": "lm_head.weight"}, # Needs vocab pruning if nemo vocab > hf vocab

    # Final LayerNorm
    "decoder.final_layernorm.weight": {"hf": "model.norm.weight"},

    # Attention Layers
    "decoder.layers.{l}.self_attention.linear_proj.weight": { # o_proj
        "tp": 1,
        "hf": "model.layers.{gl}.self_attn.o_proj.weight",
    },
    "decoder.layers.{l}.self_attention.linear_qkv.weight": { # q, k, v
        "tp": 0,
        "hf_func": split_qkv_llama,
    },
    "decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight": { # input_layernorm
        "hf": "model.layers.{gl}.input_layernorm.weight"
    },

    # MLP Layers (Dense Layers in Llama4 Arch)
    "decoder.layers.{l}.pre_mlp_layernorm.weight": { # post_attention_layernorm (used for dense layers)
        "hf": "model.layers.{gl}.post_attention_layernorm.weight"
    },
    "decoder.layers.{l}.mlp.linear_fc1.weight": { # gate_proj + up_proj (used for dense layers)
        "tp": 0,
        "hf_func": split_fc1_gate_down_llama4_dense,
    },
     "decoder.layers.{l}.mlp.linear_fc2.weight": { # down_proj (used for dense layers)
        "tp": 1,
        "hf": "model.layers.{gl}.feed_forward.down_proj.weight",
    },

    # MoE Layers (Specific to Llama4 MoE Arch)
    "decoder.layers.{l}.mlp.router.weight": { # MoE router
        "hf": "model.layers.{gl}.feed_forward.router.weight",
    },
    "decoder.layers.{l}.mlp.shared_experts.linear_fc1.weight": { # Shared Expert gate_proj + up_proj
         "tp": 0, # Matches fc1 row-parallel
         "hf_func": split_shared_fc1_llama4,
    },
    "decoder.layers.{l}.mlp.shared_experts.linear_fc2.weight": { # Shared Expert down_proj
         "tp": 1, # Matches fc2 column-parallel
         "hf": "model.layers.{gl}.feed_forward.shared_expert.down_proj.weight",
    },
    "decoder.layers.{l}.mlp.experts.linear_fc1.weight": { # MoE Experts gate_up_proj (needs transpose)
         "tp": 0, # Matches fc1 row-parallel for the weight matrix itself (before expert dim)
         "hf_func": transpose_expert_fc1,
    },
    "decoder.layers.{l}.mlp.experts.linear_fc2.weight": { # MoE Experts down_proj (needs transpose)
         "tp": 1, # Matches fc2 column-parallel for the weight matrix itself (before expert dim)
         "hf_func": transpose_expert_fc2,
    },

    # NOTE: The original Llama 2/3 mapping for mlp.linear_fc1.layer_norm_weight is intentionally omitted
    # as Llama4 uses pre_mlp_layernorm for dense layers.
    # "decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight": { ... }
}

