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


@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L265

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.

    Returns:
        torch.Tensor: Log softmax output with the same shape as input, but values represent
            log probabilities normalized across the full vocabulary dimension.
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
    """Custom autograd function for computing log probabilities in a distributed setting.

    Taken from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
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
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
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
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
) -> torch.Tensor:
    """Get log probabilities from TP sharded vocab logits.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len, vocab_size//TP]
            where TP is the tensor parallel size.
        target (torch.Tensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L354
    """
    target = target.roll(shifts=-1, dims=-1)
    probs: torch.Tensor = DistributedLogprob.apply(  # type: ignore
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only,
    ).contiguous()
    return probs[:, :-1]


def _compute_distributed_log_softmax_with_grad(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers, with gradients.

    This function is identical to `_compute_distributed_log_softmax` but without the
    `torch.no_grad()` decorator, allowing gradients to be computed. It is intended
    for use within a custom `torch.autograd.Function`.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.

    Returns:
        torch.Tensor: Log softmax output with the same shape as input, but values represent
            log probabilities normalized across the full vocabulary dimension.
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


class DistributedEntropy(torch.autograd.Function):
    """Custom autograd function for computing entropy in a distributed setting.

    This function calculates the entropy of a distribution over a vocabulary that is
    sharded across multiple devices (tensor parallelism). It correctly handles the
    forward pass (calculating entropy) and the backward pass (propagating gradients
    back to the logits).
    """

    @staticmethod
    def forward(
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        # Get log probabilities for this rank's partition of the vocab.
        log_probs_partition = _compute_distributed_log_softmax_with_grad(vocab_parallel_logits, group=group)
        # Convert log probabilities to probabilities.
        probs_partition = torch.exp(log_probs_partition)

        # Calculate this partition's contribution to the total entropy.
        # The entropy is H(p) = -sum(p * log(p)).
        entropy_partition = -torch.sum(probs_partition * log_probs_partition, dim=-1)

        # Sum the entropy contributions from all ranks to get the total entropy.
        torch.distributed.all_reduce(entropy_partition, op=torch.distributed.ReduceOp.SUM, group=group)

        # Save tensors needed for the backward pass.
        ctx.save_for_backward(log_probs_partition, entropy_partition)

        return entropy_partition

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None]:
        grad_output = grad_outputs[0]
        log_probs_partition, entropy = ctx.saved_tensors

        probs_partition = torch.exp(log_probs_partition)

        # The gradient of the entropy H w.r.t a specific logit l_i is -p_i * (log(p_i) + H),
        # where p_i is the probability for the i-th token and H is the total entropy.
        # This is multiplied by the upstream gradient (grad_output).
        grad_input = -probs_partition * (log_probs_partition + entropy.unsqueeze(-1))
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None


def from_parallel_logits_to_entropy(
    vocab_parallel_logits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Compute entropy from TP sharded vocab logits.

    This computes the entropy of the token distribution for each position in the sequence,
    excluding the final token. This is consistent with typical language model usage where
    the logits at position `i` are used to predict the token at position `i+1`.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.

    Returns:
        torch.Tensor: Entropy tensor with shape [batch_size, seq_len - 1].
    """
    entropy: torch.Tensor = DistributedEntropy.apply(  # type: ignore
        vocab_parallel_logits, group
    ).contiguous()
    # Exclude the entropy from the last token's logits, as it's not used for prediction.
    return entropy[:, :-1]
