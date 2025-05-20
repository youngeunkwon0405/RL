import torch

# NOTE: Not explicitly naming router part of MoE - may need to fix this later
def is_moe(param_name):
    return "expert" in param_name

# Below function logic taken from HFDeepSeekExporter https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/deepseek.py#L454
# NOTE: LLamaExporter uses same split_fc fn, keeping it in else statement for now
def split_fc1(gathered_mcore_fc1, cfg, gk):
    """Split fc1 into {gate|up} proj in HF."""
    if is_moe(gk):
        if "shared" in gk:
            gate_key = "model.layers.{gl}.mlp.shared_experts.gate_proj.weight"
            up_key = "model.layers.{gl}.mlp.shared_experts.up_proj.weight"
        else:
            gate_key = "model.layers.{gl}.mlp.experts.{egl}.gate_proj.weight"
            up_key = "model.layers.{gl}.mlp.experts.{egl}.up_proj.weight"
    else:
        gate_key = "model.layers.{gl}.mlp.gate_proj.weight"
        up_key = "model.layers.{gl}.mlp.up_proj.weight"
    gate_proj, up_proj = torch.chunk(gathered_mcore_fc1, 2, dim=0)
    return {gate_key: gate_proj, up_key: up_proj}


mcore_te_to_hf_deepseek = {
    # Embed
    "embedding.word_embeddings.weight": {"tp": 0, "hf": "model.embed_tokens.weight"},
    # Attention
    "decoder.layers.{l}.input_layernorm.weight": {
        "hf": "model.layers.{gl}.input_layernorm.weight"
    },
    "decoder.layers.{l}.self_attention.linear_proj.weight": {
        "tp": 1,
        "hf": "model.layers.{gl}.self_attn.o_proj.weight",
    },
    "decoder.layers.{l}.self_attention.linear_q_down_proj.weight": {
        "tp": 0,
        "hf": "model.layers.{gl}.self_attn.q_a_proj.weight",
    },
    "decoder.layers.{l}.self_attention.linear_q_up_proj.weight": {
        "tp": 0,
        "hf": "model.layers.{gl}.self_attn.q_b_proj.weight",
    },
    "decoder.layers.{l}.self_attention.linear_kv_down_proj.weight": {
        "tp": 0,
        "hf": "model.layers.{gl}.self_attn.kv_a_proj_with_mqa.weight",
    },
    "decoder.layers.{l}.self_attention.linear_kv_up_proj.weight": {
        "tp": 0,
        "hf": "model.layers.{gl}.self_attn.kv_b_proj.weight",
    },
    "decoder.layers.{l}.self_attention.linear_q_up_proj.layer_norm_weight": {
        "hf": "model.layers.{gl}.self_attn.q_a_layernorm.weight"
    },
    "decoder.layers.{l}.self_attention.linear_kv_up_proj.layer_norm_weight": {
        "hf": "model.layers.{gl}.self_attn.kv_a_layernorm.weight"
    },
    # Dense MLP
    "decoder.layers.{l}.pre_mlp_layernorm.weight": {
        "hf": "model.layers.{gl}.post_attention_layernorm.weight"
    },
    "decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight": {
        "hf": "model.layers.{gl}.post_attention_layernorm.weight"
    },
    "decoder.layers.{l}.mlp.linear_fc1.weight": {
        "tp": 0,
        "hf_func": split_fc1,
    },  # model.layers.*.mlp.{gate|up}_proj.weight taken from exporter
    "decoder.layers.{l}.mlp.linear_fc2.weight": {
        "tp": 1,
        "hf": "model.layers.{gl}.mlp.down_proj.weight",
    },  # Treating like llama
    # MoE
    "decoder.layers.{l}.mlp.router.weight": {
        "etp": 0,
        "hf": "model.layers.{gl}.mlp.gate.weight",
    },  # Does ETP affect this? TODO: Figure this out
    "decoder.layers.{l}.mlp.router.expert_bias": {
        "hf": "model.layers.{gl}.mlp.gate.e_score_correction_bias"
    },
    "decoder.layers.{l}.mlp.experts.linear_fc1.weight{el}": {
        "etp": 0,
        "hf_func": split_fc1,
    },  # model.layers.*.mlp.experts.*.{gate|up}_proj.weight',
    "decoder.layers.{l}.mlp.experts.linear_fc2.weight{el}": {
        "etp": 1,
        "hf": "model.layers.{gl}.mlp.experts.{egl}.down_proj.weight",
    },  # treating like llama linear_fc2.weight
    "decoder.layers.{l}.mlp.shared_experts.linear_fc1.weight": {
        "etp": 0,
        "hf_func": split_fc1,
    },  # model.layers.*.mlp.shared_experts.{gate|up}_proj.weight
    "decoder.layers.{l}.mlp.shared_experts.linear_fc2.weight": {
        "etp": 1,
        "hf": "model.layers.{gl}.mlp.shared_experts.down_proj.weight",
    },  # come back to shared experts
    # LM Head
    "decoder.final_layernorm.weight": {"hf": "model.norm.weight"},
    "output_layer.weight": {"tp": 0, "hf": "lm_head.weight"},
}
