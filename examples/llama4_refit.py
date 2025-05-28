# python /lustre/fsw/portfolios/coreai/users/yuya/nemo-rl/examples/llama4_refit.py

import ray
import torch

from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.megatron.converters import REGISTRY, ModelType
from nemo_rl.models.policy.megatron_policy import MegatronPolicy
from nemo_rl.models.generation.interfaces import GenerationOutputSpec, configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration

ray.init()

model_name = "/root/checkpoints/llama4-scout-custom-init"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"

converter_type = ModelType.LLAMA4
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU Allocation
MEGATRON_TP = 2
VLLM_TP = 2

# --- Megatron Config (Restored) ---
config = {
    "model_name": model_name,
    "train_global_batch_size": 1,
    "train_micro_batch_size": 1,
    "generation_batch_size": 2,
    "learning_rate": 0.0001,
    "logprob_batch_size": 1,
    "generation": {
        "max_total_sequence_length": 1024,
        "max_new_tokens": 52,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    },
    "precision": "bfloat16",
    "tensor_model_parallel_size": MEGATRON_TP,
    "expert_tensor_parallel_size": MEGATRON_TP,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "parallel_output": True,
    "max_total_sequence_length": 1024,
    "activation_checkpointing_enabled": False,
    "fsdp_offload_enabled": False,
    "optimizer": {
        "type": "adam",
        "kwargs": {
            "lr": 0.0001,
            "weight_decay": 0.0,
            "eps": 1e-8,
        },
    },
    "dtensor_cfg": {
        "enabled": False,
    },
    "megatron_cfg": {
        "converter_type": converter_type,
        "enabled": True,
        "empty_unused_memory_level": 1,
        "converter_class": "Llama4ForCausalLM",
        "optimizer": {
            "optimizer": "adam",
            "lr": 5.0e-6,
            "min_lr": 5.0e-7,
            "weight_decay": 0.01,
            "bf16": False,
            "fp16": False,
            "params_dtype": "float32",

            # adam
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,

            # sgd
            "sgd_momentum": 0.9,

            # distributed optimizer
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": True,

            "clip_grad": 1.0,
        },
        "scheduler": {
            "start_weight_decay": 0.01,
            "end_weight_decay": 0.01,
            "weight_decay_incr_style": "constant",
            "lr_decay_style": "constant",
            "lr_decay_iters": None,
            "lr_warmup_iters": 50,
            "lr_warmup_init": 5.0e-7,
        },
    },
}

# --- VLLM Config ---
vllm_config = {
    "backend": "vllm",
    "model_name": model_name,
    "tokenizer": { # Based on test_vllm_generation.py
        "name": model_name,
    },
    "dtype": "bfloat16", # Match precision
    "max_new_tokens": 512, # Match generation config
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "pad_token_id": tokenizer.eos_token_id, # Match generation config
    "vllm_cfg": {
        "tensor_parallel_size": VLLM_TP,
        "gpu_memory_utilization": 0.6, # Default in vLLM, adjust if needed
        "max_model_len": 512, # Match generation config
    },
    # Needed by configure_generation_config
    "max_total_sequence_length": 512,
}
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
vllm_config = configure_generation_config(vllm_config, tokenizer)


# --- Clusters (Separate for each policy) ---
print(f"Setting up Megatron Cluster with TP={MEGATRON_TP}")
megatron_cluster = RayVirtualCluster(
    name="megatron_cluster", # Renamed cluster
    bundle_ct_per_node_list=[MEGATRON_TP],
    use_gpus=True,
    num_gpus_per_node=MEGATRON_TP,
    max_colocated_worker_groups=2, # Set to 1 as we only have one policy now
)

# --- Instantiate Policies ---
print("Instantiating MegatronPolicy...")
policy = MegatronPolicy(cluster=megatron_cluster, config=config, tokenizer=tokenizer, init_reference_model=False, init_optimizer=False)

# prompt = r"Think step-by-step to solve the following problem. Output your answer inside of \\boxed{} tags.:\\nAn octahedron has eight equilateral triangular faces, each with side length 1. Let $P$ be a point inside the octahedron, and let $d_1, d_2, \\ldots, d_8$ be the distances from $P$ to the centroids of the eight faces of the octahedron. Find the largest possible value of the sum $d_1 + d_2 + \\cdots + d_8.\\n\\nLet's think step-by-step"
# prompt2 = "Count from one to one hundred while laughing"
# if not tokenizer.pad_token:
#     tokenizer.pad_token = tokenizer.eos_token
#
# prompt = tokenizer.apply_chat_template(
#     [{"role": "user", "content": prompt}],
#     tokenize=False,
# )
# prompt2 = tokenizer.apply_chat_template(
#     [{"role": "user", "content": prompt2}],
#     tokenize=False,
# )

prompt = "The following are multiple choice questions (with answers) about world religions.\n\nWhen was the first Buddhist temple constructed in Japan?\nA. 325 CE\nB. 119 CE\nC. 451 CE\nD. 596 CE\nAnswer:"
# Tokenize the prompt
tokenized = tokenizer(
    [prompt],
    padding=True,
    truncation=True,
    return_tensors="pt",
    padding_side="right",
)

# Calculate input lengths from attention mask
input_ids = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]
input_lengths = attention_mask.sum(dim=1).to(torch.int32)

generation_data: BatchedDataDict = BatchedDataDict(
    {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    }
)

# --- Megatron Policy Workflow (Restored) ---
# print("\n--- Running Megatron Policy Workflow ---")
# megatron_generated_data: BatchedDataDict[GenerationOutputSpec] = policy.generate(generation_data)
#
# print("Megatron Generated Data:", megatron_generated_data)
# print([tokenizer.decode(megatron_generated_data["output_ids"][i]) for i in range(len(megatron_generated_data["output_ids"]))])

print("Instantiating VllmGeneration...")
vllm_policy = VllmGeneration(cluster=megatron_cluster, config=vllm_config)

# from transformers import AutoModelForCausalLM
# hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
# state_dict = hf_model.state_dict()
# shapes = {}
# for key, tensor in state_dict.items():
#     shapes[key] = tensor.shape
# print(f"hf shapes: {shapes}")
#
#
refit_buffer_size_gb = 2
sd_info = policy.prepare_weights_for_ipc()
print(sd_info)
vllm_policy.prepare_for_generation(tags=["weights"])
# Streaming update weights to save memory
state_dict_info = policy.prepare_weights_for_ipc()
# group keys to save time
available_bytes = refit_buffer_size_gb * (1024**3)
split_keys, keys = [], []
for key, size_in_bytes in state_dict_info:
    if size_in_bytes > available_bytes:
        if keys:
            split_keys.append(keys)
            keys = []
        available_bytes = refit_buffer_size_gb * (1024**3)

    keys.append(key)
    available_bytes -= size_in_bytes

if len(keys) > 0:
    split_keys.append(keys)
# do update
for keys in split_keys:
    ipc_handles = policy.get_weights_ipc_handles(keys)
    vllm_policy.update_weights(ipc_handles)
print("Done updating weights")
policy.offload_after_refit()
vllm_policy.prepare_for_generation(tags=["kv_cache"])

# #compare shapes
# all_match = True
# for key in megatron_shapes.keys():
    # if key not in state_dict.keys():
        # print(f"m key {key} not in hf state_dict")
        # all_match = False
    # else:
        # if megatron_shapes[key] != state_dict[key].shape:
            # print(f"m key {key} shape mismatch: {megatron_shapes[key]} != {state_dict[key].shape}")
            # all_match = False
# for key in state_dict.keys():
    # if key not in megatron_shapes.keys():
        # print(f"h key {key} not in megatron shapes")
#         all_match = False

# if all_match:
    # print("All shapes match")
# else:
  #   print("Shapes do not match")
#
#
#
# from nemo_reinforcer.algorithms.loss_functions import ClippedPGLossFn, ClippedPGLossConfig, ClippedPGLossDataDict
#
# loss_cfg = ClippedPGLossConfig(
#     reference_policy_kl_penalty=0.1,
#     ratio_eps_min=0.1,
#     ratio_eps_max=0.1,
#     use_on_policy_kl_approximation=False,
#     use_importance_sampling_correction=False,
# )
# loss_fn = ClippedPGLossFn(loss_cfg)
#
# print("\n--- Running VLLM Policy Workflow ---")
# print("Preparing VLLM for generation...")
# vllm_policy.prepare_for_generation()
# print("Running VLLM generation...")
# vllm_generated_data: BatchedDataDict[GenerationOutputSpec] = vllm_policy.generate(generation_data) # Use same input data
# print("Finishing VLLM generation...")
# vllm_policy.finish_generation()
#
# print("VLLM Generated Data:", vllm_generated_data)
# print("VLLM Decoded:")
# print([tokenizer.decode(vllm_generated_data["output_ids"][i]) for i in range(len(vllm_generated_data["output_ids"]))])
#
#
# # --- Shutdown ---
# print("Shutting down VLLM policy...")
# vllm_policy.shutdown() # This should handle vllm_cluster shutdown via worker group
#
# # Explicitly shutdown the megatron cluster as well
# print("Shutting down Megatron cluster...")
# megatron_cluster.shutdown()
#
# print("Script finished.")
# MegatronPolicy might not require explicit shutdown via a method call