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

from typing import Any, NotRequired, TypedDict, Union

from nemo_rl.models.generation.interfaces import GenerationConfig


class DTensorConfig(TypedDict):
    enabled: bool
    env_vars: NotRequired[dict[str, str]]
    _v2: NotRequired[bool]
    cpu_offload: NotRequired[bool]
    sequence_parallel: NotRequired[bool]
    activation_checkpointing: NotRequired[bool]
    tensor_parallel_size: NotRequired[int]
    context_parallel_size: NotRequired[int]
    custom_parallel_plan: NotRequired[str]
    clear_cache_every_n_steps: NotRequired[int]


class SequencePackingConfig(TypedDict):
    enabled: bool
    train_mb_tokens: int
    # Not required because some algorithms like SFT don't calculate log probs
    logprob_mb_tokens: NotRequired[int]
    algorithm: str


class RewardModelConfig(TypedDict):
    enabled: bool
    reward_model_type: str


class MegatronOptimizerConfig(TypedDict):
    optimizer: str
    lr: float
    min_lr: float
    weight_decay: float
    bf16: bool
    fp16: bool
    params_dtype: str
    # adam
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    # sgd
    sgd_momentum: float
    # distributed optimizer
    use_distributed_optimizer: bool
    use_precision_aware_optimizer: bool
    clip_grad: float
    # knob to enable optimizer cpu offload
    optimizer_cpu_offload: bool
    # knob to set the fraction of parameters to keep on CPU
    # currently if optimizer_cpu_offload is true, this knob must be 1.0
    optimizer_offload_fraction: float


class MegatronSchedulerConfig(TypedDict):
    start_weight_decay: float
    end_weight_decay: float
    weight_decay_incr_style: str
    lr_decay_style: str
    lr_decay_iters: int
    lr_warmup_iters: int
    lr_warmup_init: float


class MegatronDDPConfig(TypedDict):
    grad_reduce_in_fp32: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    average_in_collective: bool
    use_custom_fsdp: bool
    data_parallel_sharding_strategy: str


class MegatronConfig(TypedDict):
    enabled: bool
    env_vars: NotRequired[dict[str, str]]
    empty_unused_memory_level: int
    activation_checkpointing: bool
    converter_type: str
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int
    num_layers_in_last_pipeline_stage: int
    context_parallel_size: int
    pipeline_dtype: str
    sequence_parallel: bool
    freeze_moe_router: bool
    expert_tensor_parallel_size: int
    expert_model_parallel_size: int
    defer_fp32_logits: NotRequired[bool]
    # gives ~20% training perf speedup with sequence packing
    apply_rope_fusion: bool
    # gives ~25% training perf speedup with sequence packing and apply_rope_fusion
    bias_activation_fusion: bool

    optimizer: NotRequired[MegatronOptimizerConfig]
    scheduler: NotRequired[MegatronSchedulerConfig]
    distributed_data_parallel_config: MegatronDDPConfig


class TokenizerConfig(TypedDict):
    name: str
    chat_template: NotRequired[str]
    # Arguments to pass to tokenizer.apply_chat_template(...). This can be used to pass kwargs like enable_thinking=true
    chat_template_kwargs: NotRequired[dict[str, Any]]


class PytorchOptimizerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]
    milestones: NotRequired[list[int]]  # Used in SequentialLR configuration


SchedulerMilestones = dict[str, list[int]]


class DynamicBatchingConfig(TypedDict):
    # dynamic_batching improves performance by ensuring logprob and training microbatches
    # have a sufficent number of tokens to maximize GPU utilization. Specifically, variable length
    # responses are sorted by sequence length and bucketed into microbatches with a total
    # amount of tokens is approximately close to 'train_mb_tokens' and 'logprob_mb_tokens' for the
    # training and logprob stages respectively.
    enabled: bool

    ## required if enabled is true
    train_mb_tokens: NotRequired[int]
    logprob_mb_tokens: NotRequired[int]
    sequence_length_round: NotRequired[int]


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    logprob_batch_size: NotRequired[int]
    logprob_chunk_size: NotRequired[int]
    generation: NotRequired[GenerationConfig]
    generation_batch_size: NotRequired[
        int
    ]  # used in static batched (framework) generation
    precision: str
    # offload_optimizer_states_for_logprob:
    # - True: offload optimizer states while calculating logprobs, can increase
    # logprob batch size for higher GEMM efficiency but needs to pay offloading/onloading overhead
    # - False (default): keep optimizer states on GPU, avoid extra offloading/onloading latency penalty
    offload_optimizer_states_for_logprob: bool
    reward_model_cfg: NotRequired[RewardModelConfig]
    dtensor_cfg: DTensorConfig
    megatron_cfg: NotRequired[MegatronConfig]
    hf_config_overrides: NotRequired[dict[str, Any]]
    dynamic_batching: DynamicBatchingConfig
    sequence_packing: NotRequired[SequencePackingConfig]
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    max_grad_norm: NotRequired[Union[float, int]]
    refit_buffer_size_gb: NotRequired[float]
    optimizer: NotRequired[PytorchOptimizerConfig]
    scheduler: NotRequired[list[SinglePytorchSchedulerConfig] | SchedulerMilestones]
