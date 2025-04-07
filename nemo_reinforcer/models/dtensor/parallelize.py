import torch

from torch.distributed.tensor import DTensor
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
# Load model directly
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
    SequenceParallel,
)

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

def _parallelize_model(
    model,
    dp_mesh,
    tp_mesh,
    param_dtype,
    sequence_parallel=False,
    activation_checkpointing=False,
    cpu_offload=False,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    offload_policy = (
        CPUOffloadPolicy(pin_memory=False)
        if cpu_offload
        else torch.distributed.fsdp.OffloadPolicy
    )
    
    model_cls = type(model)
    if model_cls not in PARALLIZE_FUNCTIONS:
        raise ValueError(f"Model {model_cls} not supported as part of dtensor")

    func = PARALLIZE_FUNCTIONS[type(model)]

    return func(
        model,
        dp_mesh,
        tp_mesh,
        mp_policy,
        offload_policy,
        sequence_parallel,
        activation_checkpointing,
    )

def _parallelize_llama(
    model,
    dp_mesh,
    tp_mesh,
    mp_policy,
    offload_policy,
    sequence_parallel=False,
    activation_checkpointing=False,
):
    base_model_tp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "model.layers.*.post_attention_layernorm": SequenceParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(base_model_sp_plan)
    
    parallelize_module(model, tp_mesh, base_model_tp_plan)

    if activation_checkpointing:
        for i in range(len(model.model.layers)):
            model.model.layers[i].mlp = checkpoint_wrapper(model.model.layers[i].mlp)

    for layer in model.model.layers:
        fully_shard(
            layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )

    return fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)

def _parallelize_qwen(
    model,
    dp_mesh,
    tp_mesh,
    mp_policy,
    offload_policy,
    sequence_parallel=False,
    activation_checkpointing=False,
):
    class Qwen2RotaryEmbedParallel(SequenceParallel):
        @staticmethod
        def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
            """NOTE: this function will hang if the sequence length is not properly divisible by TP size
            """
            new_inputs = list(inputs)

            if not isinstance(inputs[0], DTensor):
                new_inputs[0] = DTensor.from_local(local_tensor=inputs[0], device_mesh=device_mesh, placements=sequence_sharding, run_check=False)

            if not isinstance(inputs[1], DTensor):
                # new_inputs[1] = DTensor.from_local(local_tensor=inputs[1], device_mesh=device_mesh, placements=sequence_sharding, run_check=False)
                new_inputs[1] = DTensor.from_local(local_tensor=inputs[1], device_mesh=device_mesh, placements=(Replicate(),), run_check=False)

            return type(inputs)(new_inputs)

    if sequence_parallel:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
            ),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.rotary_emb": Qwen2RotaryEmbedParallel(),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }

    else:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(
                output_layouts=Shard(-1), use_local_output=False
            ),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        }

    parallelize_module(model, tp_mesh, base_model_tp_plan)

    if activation_checkpointing:
        for i in range(len(model.model.layers)):
            model.model.layers[i].mlp = checkpoint_wrapper(model.model.layers[i].mlp)

    for layer in model.model.layers:
        fully_shard(
            layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )

    return fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)

PARALLIZE_FUNCTIONS = {
    Qwen2ForCausalLM: _parallelize_qwen,
    LlamaForCausalLM: _parallelize_llama,
}
