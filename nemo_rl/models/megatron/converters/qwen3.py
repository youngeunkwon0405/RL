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

import functools

from nemo_rl.models.megatron.converters.llama import (
    split_qkv_llama,
)
from nemo_rl.models.megatron.converters.llama4 import (
    split_fc1_gate_down_llama4_common,
)

## TODO: assuming MoE for now. Add support for dense
## TODO: rearrage into logical order
mcore_te_to_hf_qwen3 = {
    "embedding.word_embeddings.weight": {"tp": 0, "hf": "model.embed_tokens.weight"},
    "decoder.final_layernorm.weight": {"hf": "model.norm.weight"},
    "output_layer.weight": {"tp": 0, "hf": "lm_head.weight"},
    ## TODO: l --> gl? what are l and gl?
    ## TODO: need to match full beginning of the string. Not sure how to do this
    "decoder.layers.{l}.self_attention.linear_proj.weight": {
        "tp": 1,
        "hf": "language_model.model.layers.{gl}.self_attn.o_proj.weight",
    },
    "decoder.layers.{l}.linear_qkv.weight": {
        "tp": 0,
        "hf_func": split_qkv_llama,
    },
    "decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight": {
        "hf": "language_model.model.layers.{gl}.input_layernorm.weight"
    },
    "decoder.layers.{l}.mlp.experts.linear_fc1.weight{e}": {
        "tp": 1,
        "hf_func": functools.partial(
            split_fc1_gate_down_llama4_common,
            ## TODO: when using expert parallel, do we need local and global expert indices?
            gate_proj_name="language_model.model.layers.{gl}.mlp.experts.{e}.up_proj.weight",
            up_proj_name="language_model.model.layers.{gl}.mlp.experts.{e}.up_proj.weight",
        ),
    },
    "decoder.layers.{l}.mlp.router.weight": {
        "hf": "language_model.model.layers.{gl}.mlp.gate.weight"  ## TODO: check
    },
    "decoder.layers.{l}.mlp.experts.linear_fc2.weight{e}": {  ## TODO: what to do about asterisk at the end?
        "tp": 2,
        ## only for llama models
        # hf_func": functools.partial(
        #    transpose_expert_weight,
        #    down_proj_name="language_model.model.layers.{gl}.mlp.experts.down_proj.weight",
        # ),
        "hf": "language_model.model.layers.{gl}.mlp.experts.{e}.down_proj.weight",
    },
    "decoder.layers.{l}.pre_mlp_layernorm.weight": {
        "hf": "language_model.model.layers.{gl}.post_attention_layernorm.weight",
    },
}

## TODO: add support for dense
