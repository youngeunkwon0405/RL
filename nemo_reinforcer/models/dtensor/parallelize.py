import torch

from torch.distributed.tensor import DTensor
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
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
    if tp_mesh.size() > 1:
        base_model_tp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(
                output_layouts=Shard(-1), use_local_output=False
            ),
        }

        base_model_sp_plan = {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
            ),
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

    return fully_shard(
        model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
    )


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
            """NOTE: this function will hang if the sequence length is not properly divisible by TP size"""
            new_inputs = list(inputs)

            if not isinstance(inputs[0], DTensor):
                new_inputs[0] = DTensor.from_local(
                    local_tensor=inputs[0],
                    device_mesh=device_mesh,
                    placements=sequence_sharding,
                    run_check=False,
                )

            if not isinstance(inputs[1], DTensor):
                # new_inputs[1] = DTensor.from_local(local_tensor=inputs[1], device_mesh=device_mesh, placements=sequence_sharding, run_check=False)
                new_inputs[1] = DTensor.from_local(
                    local_tensor=inputs[1],
                    device_mesh=device_mesh,
                    placements=(Replicate(),),
                    run_check=False,
                )

            return type(inputs)(new_inputs)

    if tp_mesh.size() > 1:
        if sequence_parallel:
            base_model_tp_plan = {
                "lm_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
                "model.embed_tokens": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "model.rotary_emb": Qwen2RotaryEmbedParallel(),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": SequenceParallel(),
                "model.layers.*.self_attn.q_proj": ColwiseParallel(
                    use_local_output=False
                ),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(
                    use_local_output=False
                ),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(
                    use_local_output=False
                ),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
                "model.layers.*.post_attention_layernorm": SequenceParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
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

    return fully_shard(
        model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
    )


PARALLIZE_FUNCTIONS = {
    Qwen2ForCausalLM: _parallelize_qwen,
    LlamaForCausalLM: _parallelize_llama,
}


@torch.no_grad()
def _compute_distributed_log_softmax(vocab_parallel_logits, group):
    """Expects a size B x S x V//TP tensor, computes a stable distributed softmax
    return shape B x S x V//TP but softmaxed across the V dimension. More stable than just computing softmax
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


class DistributedLogprob(torch.autograd.Function):
    """Function to get logprobs out and differentiate through it"""

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only=False,
    ):
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        log_softmax_output = _compute_distributed_log_softmax(
            vocab_parallel_logits, group=group
        )
        log_probs = log_softmax_output.clone()
        softmax_output = log_softmax_output.exp_()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # 1 if it's the chosen log prob, 0 otherwise
        is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target, num_classes=partition_vocab_size
        )

        grad_input = is_chosen.float().sub_(softmax)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits,
    target,
    vocab_start_index,
    vocab_end_index,
    group,
    inference_only=False,
):
    """Get log probs out of a B x S x V//TP tensor
        NOTE: this function shifts the target, which means you must give it the unmodified targets

    Returns a B x S-1 tensor
    """
    target = target.roll(shifts=-1, dims=-1)
    probs = DistributedLogprob.apply(
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only,
    ).contiguous()
    return probs[:, :-1]


def get_logprobs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor, input_ids: torch.Tensor
):
    tp_mesh = vocab_parallel_logits.device_mesh
    tp_rank: int = tp_mesh.get_local_rank()

    vocab_interval_per_rank = vocab_parallel_logits.shape[-1] // tp_mesh.size()

    return from_parallel_logits_to_logprobs(
        vocab_parallel_logits.to_local(),
        input_ids,
        vocab_interval_per_rank * tp_rank,
        (tp_rank + 1) * vocab_interval_per_rank,
        tp_mesh.get_group(),
        inference_only=not torch.is_grad_enabled(),
    )
