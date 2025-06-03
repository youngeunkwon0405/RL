# uv run python3 /opt/nemo-rl/examples/llama4_refit.py

import ray
import torch

from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.megatron.converters import ModelType
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.algorithms.grpo import refit_policy_generation

ray.init()

model_name = "/root/checkpoints/llama4-scout-custom-init"

converter_type = ModelType.LLAMA4
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU Allocation
MEGATRON_TP = 2
VLLM_TP = 2

# --- Megatron Config (Updated from grpo_math_llama4_toy.yaml) ---
config = {
    "model_name": model_name,
    "training_backend": "megatron",
    "train_global_batch_size": 1,
    "train_micro_batch_size": 1,
    "generation_batch_size": 2,
    "learning_rate": 0.0001,
    "logprob_batch_size": 1,
    "generation": {
        "max_total_sequence_length": 256,
        "max_new_tokens": 256,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    },
    "precision": "bfloat16",
    "pipeline_dtype": "bfloat16",
    "parallel_output": True,
    "max_total_sequence_length": 256,
    "activation_checkpointing_enabled": False,
    "fsdp_offload_enabled": False,
    "max_grad_norm": 1.0,
    "refit_buffer_size_gb": 4,
    "make_sequence_length_divisible_by": MEGATRON_TP,
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
    "dynamic_batching": {
        "enabled": False,
        "train_mb_tokens": 256,
        "logprob_mb_tokens": 256,
        "sequence_length_round": 64,
    },
    "megatron_cfg": {
        "converter_type": "Llama4ForCausalLM",
        "enabled": True,
        "empty_unused_memory_level": 1,
        "tensor_model_parallel_size": MEGATRON_TP,
        "sequence_parallel": False,
        "expert_tensor_parallel_size": MEGATRON_TP,
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_dtype": "bfloat16",
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
    "max_new_tokens": 256, # Match generation config
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "tensor_parallel_size": VLLM_TP,
        "gpu_memory_utilization": 0.6, # Default in vLLM, adjust if needed
        "max_model_len": 256, # Match generation config
        "precision": "bfloat16", # Required for vllm_cfg
    },
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
print("Instantiating Policy with Megatron backend...")
policy = Policy(cluster=megatron_cluster, config=config, tokenizer=tokenizer, init_reference_model=False, init_optimizer=False)

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
print("\n--- Getting Megatron Policy Logprobs ---")
policy.prepare_for_lp_inference()
megatron_logprobs_data = policy.get_logprobs(generation_data)
print("Megatron Logprobs shape:", megatron_logprobs_data["logprobs"].shape)
print("Megatron Logprobs sample:", megatron_logprobs_data["logprobs"][0, :10])  # First 10 tokens

# --- COMMENTED OUT: First vLLM Policy (Generation Mode) ---
# print("Instantiating VllmGeneration...")
# vllm_policy = VllmGeneration(cluster=megatron_cluster, config=vllm_config)

# refit_buffer_size_gb = 4
# sd_info = policy.prepare_weights_for_ipc()
# print(sd_info)
# vllm_policy.prepare_for_generation(tags=["weights"])
# # Streaming update weights to save memory
# state_dict_info = policy.prepare_weights_for_ipc()
# # group keys to save time
# available_bytes = refit_buffer_size_gb * (1024**3)
# split_keys, keys = [], []
# for key, size_in_bytes in state_dict_info:
#     if size_in_bytes > available_bytes:
#         if keys:
#             split_keys.append(keys)
#             keys = []
#         available_bytes = refit_buffer_size_gb * (1024**3)
# 
#     keys.append(key)
#     available_bytes -= size_in_bytes
# 
# if len(keys) > 0:
#     split_keys.append(keys)
# # do update
# for keys in split_keys:
#     ipc_handles = policy.get_weights_ipc_handles(keys)
#     vllm_policy.update_weights(ipc_handles)
# print("Done updating weights")
# policy.offload_after_refit()
# vllm_policy.prepare_for_generation(tags=["kv_cache"])
# 
# print("\n--- Getting vLLM Policy Logprobs via Generation ---")
# # First use vLLM to generate and get logprobs during generation
# vllm_generated_data = vllm_policy.generate(generation_data)
# print("vLLM Generated Data shape:", vllm_generated_data["output_ids"].shape)
# print("vLLM Logprobs shape:", vllm_generated_data["logprobs"].shape)
# print("vLLM Logprobs sample:", vllm_generated_data["logprobs"][0, :10])  # First 10 tokens
# 
# # For a more fair comparison, create a special vLLM config with max_new_tokens=0 to get logprobs for input only
# print("\n--- Getting vLLM Logprobs for Input Only (Inference Mode) ---")
# 
# # Shutdown the first vLLM policy to avoid naming conflicts
# print("Shutting down first vLLM policy...")
# vllm_policy.shutdown()

# --- vLLM Policy (Inference-Only Mode) ---
print("\n--- Getting vLLM Logprobs for Input Only (Inference Mode) ---")

refit_buffer_size_gb = 4
sd_info = policy.prepare_weights_for_ipc()
print(sd_info)

vllm_inference_config = vllm_config.copy()
vllm_inference_config["max_new_tokens"] = 1  # Only get logprobs for input tokens
vllm_inference_config = configure_generation_config(vllm_inference_config, tokenizer)

# Create a temporary vLLM policy for inference-only logprobs
vllm_inference_policy = VllmGeneration(cluster=megatron_cluster, config=vllm_inference_config)

refit_policy_generation(policy, vllm_inference_policy, refit_buffer_size_gb)

# Get logprobs for input only
vllm_input_logprobs = vllm_inference_policy.generate(generation_data)
print(megatron_logprobs_data)
print(vllm_input_logprobs)
exit(0)
print("vLLM Input-only Logprobs shape:", vllm_input_logprobs["logprobs"].shape)
print("vLLM Input-only Logprobs sample:", vllm_input_logprobs["logprobs"][0, :10])

# Now compare the logprobs on the same input sequence
print("\n--- Comparing Logprobs ---")
print("Input prompt:", prompt)
print("Input tokens:", generation_data["input_ids"][0, :generation_data["input_lengths"][0]])

# Compare logprobs for the input tokens (should be similar)
input_length = generation_data["input_lengths"][0].item()
print(f"\nComparing first {input_length} tokens:")
print("Megatron logprobs:", megatron_logprobs_data["logprobs"][0, :input_length])
print("vLLM input logprobs:", vllm_input_logprobs["logprobs"][0, :input_length])

# Calculate absolute difference
abs_diff = torch.abs(megatron_logprobs_data["logprobs"][0, :input_length] - vllm_input_logprobs["logprobs"][0, :input_length])
print("Absolute difference:", abs_diff)
print("Mean absolute difference:", torch.mean(abs_diff))
print("Max absolute difference:", torch.max(abs_diff))

# COMMENTED OUT: Generated token comparison (no longer available)
# Also compare generated logprobs if available
# if vllm_generated_data["generation_lengths"][0] > 0:
#     print(f"\n--- Comparing Generated Token Logprobs ---")
#     gen_start = input_length
#     gen_end = input_length + vllm_generated_data["generation_lengths"][0]
#     print("vLLM generated logprobs:", vllm_generated_data["logprobs"][0, gen_start:gen_end])
#     print("Generated tokens:", vllm_generated_data["output_ids"][0, gen_start:gen_end])
#     print("Decoded generated text:", tokenizer.decode(vllm_generated_data["output_ids"][0, gen_start:gen_end]))

# Detailed token-by-token comparison
print("\n--- Token-by-Token Comparison ---")
input_length = generation_data["input_lengths"][0].item()
input_tokens = generation_data["input_ids"][0, :input_length]
megatron_logprobs = megatron_logprobs_data["logprobs"][0, :input_length]
vllm_logprobs = vllm_input_logprobs["logprobs"][0, :input_length]

print(f"{'Token':<15} {'Token ID':<10} {'Megatron':<12} {'vLLM':<12} {'Diff':<12}")
print("-" * 65)
for i in range(min(input_length, 10)):  # Show first 10 tokens
    token_id = input_tokens[i].item()
    token_text = tokenizer.decode([token_id])
    megatron_lp = megatron_logprobs[i].item()
    vllm_lp = vllm_logprobs[i].item()
    diff = abs(megatron_lp - vllm_lp)
    print(f"{token_text:<15} {token_id:<10} {megatron_lp:<12.6f} {vllm_lp:<12.6f} {diff:<12.6f}")

# Final cleanup
print("\n--- Cleaning up ---")
vllm_inference_policy.shutdown()
print("Script completed successfully!")