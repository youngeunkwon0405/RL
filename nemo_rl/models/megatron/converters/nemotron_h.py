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

from nemo.lightning import io
from nemo.lightning.io.state import TransformFns


def get_export_mapping(source):
    mapping = {
        'decoder.layers.*.mixer.A_log': 'backbone.layers.*.mixer.A_log',
        'decoder.layers.*.mixer.D': 'backbone.layers.*.mixer.D',
        'decoder.layers.*.mixer.conv1d.weight': 'backbone.layers.*.mixer.conv1d.weight',
        'decoder.layers.*.mixer.conv1d.bias': 'backbone.layers.*.mixer.conv1d.bias',
        'decoder.layers.*.mixer.in_proj.weight': 'backbone.layers.*.mixer.in_proj.weight',
        'decoder.layers.*.mixer.dt_bias': 'backbone.layers.*.mixer.dt_bias',
        'decoder.layers.*.mixer.out_proj.weight': 'backbone.layers.*.mixer.out_proj.weight',
        'decoder.layers.*.mixer.norm.weight': 'backbone.layers.*.mixer.norm.weight',
        'decoder.layers.*.mlp.linear_fc1.weight': 'backbone.layers.*.mixer.up_proj.weight',
        'decoder.layers.*.mlp.linear_fc2.weight': 'backbone.layers.*.mixer.down_proj.weight',
        'decoder.layers.*.self_attention.linear_proj.weight': 'backbone.layers.*.mixer.o_proj.weight',
        'decoder.final_norm.weight': 'backbone.norm_f.weight',
    }

    for i, layer_type in enumerate(source.config.hybrid_override_pattern):
        if layer_type == "M":
            mapping[f'decoder.layers.{i}.mixer.in_proj.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
        elif layer_type == "-":
            mapping[f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
        elif layer_type == "*":
            mapping[f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'] = (
                f'backbone.layers.{i}.norm.weight'
            )
        else:
            raise AttributeError(f"layer type {layer_type} not found.")
    return mapping


def get_export_transforms(hf_config):
    transforms = [
        # _export_qkv from nemo.collections.llm.gpt.model.ssm
        io.state_transform(
            source_key="decoder.layers.*.self_attention.linear_qkv.weight",
            target_key=(
                "backbone.layers.*.mixer.q_proj.weight",
                "backbone.layers.*.mixer.k_proj.weight",
                "backbone.layers.*.mixer.v_proj.weight",
            ),
            fn=TransformFns.split_qkv,
        ),
        # _export_embedding from nemo.collections.llm.gpt.model.ssm
        io.state_transform(
            source_key="embedding.word_embeddings.weight",
            target_key="backbone.embeddings.weight",
            fn=TransformFns.prune_padding,
        ),
    ]

    if not hf_config.tie_word_embeddings:
        # _export_head from nemo.collections.llm.gpt.model.ssm
        transforms.append(
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
        )

    return transforms
