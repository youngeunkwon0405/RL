# uv run python3 /opt/nemo-rl/examples/llama4_refit.py

import ray
import torch
import copy

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

# --- Megatron Policy Workflow (Updated to use generation) ---
print("\n--- Getting vLLM Policy Logprobs ---")

refit_buffer_size_gb = 4
sd_info = policy.prepare_weights_for_ipc()
print(sd_info)

vllm_inference_config = vllm_config.copy()
vllm_inference_config["max_new_tokens"] = 10
vllm_inference_config = configure_generation_config(vllm_inference_config, tokenizer)

# Create a temporary vLLM policy for inference-only logprobs
vllm_inference_policy = VllmGeneration(cluster=megatron_cluster, config=vllm_inference_config)

refit_policy_generation(policy, vllm_inference_policy, refit_buffer_size_gb)

# Get logprobs from vLLM for input only
vllm_logprobs_data = vllm_inference_policy.generate(generation_data, greedy=True)
print("vLLM Logprobs shape:", vllm_logprobs_data["logprobs"].shape)
print("vLLM Logprobs sample:", vllm_logprobs_data["logprobs"][0, -10:])
print(vllm_logprobs_data, generation_data)
# --- Megatron Policy (Generation Mode) ---
print("\n--- Getting Megatron Generation ---")

# Prepare policy for generation
policy.prepare_for_generation()
megatron_input_data = copy.deepcopy(generation_data)
megatron_input_data["input_ids"] = vllm_logprobs_data["output_ids"]
megatron_input_data["input_lengths"] = vllm_logprobs_data["unpadded_sequence_lengths"]
megatron_generation_data = policy.get_logprobs(megatron_input_data,)

print("Megatron Generation shape:", megatron_generation_data["logprobs"].shape)
print("Megatron Generation sample:", megatron_generation_data["logprobs"][0, -10:])

# Now compare the logprobs on the same input sequence
print("\n--- Comparing Logprobs ---")
print("Input prompt:", prompt)
print("Input tokens:", generation_data["input_ids"][0, :generation_data["input_lengths"][0]])

# Compare logprobs for the generated tokens only (skip input tokens)
input_length = generation_data["input_lengths"][0].item()
total_length = min(vllm_logprobs_data["logprobs"].shape[1], megatron_generation_data["logprobs"].shape[1])
generated_length = vllm_logprobs_data["generation_lengths"][0].item()

if generated_length > 0:
    print(f"\nComparing {generated_length} generated tokens (from position {input_length} to {total_length-1}):")
    print("vLLM generated logprobs:", vllm_logprobs_data["logprobs"][0, input_length:total_length])
    print("Megatron generated logprobs:", megatron_generation_data["logprobs"][0, input_length:total_length])

    # Calculate absolute difference for generated tokens only
    abs_diff = torch.abs(vllm_logprobs_data["logprobs"][0, input_length:total_length] - megatron_generation_data["logprobs"][0, input_length:total_length])
    print("Absolute difference:", abs_diff)
    print("Mean absolute difference:", torch.mean(abs_diff))
    print("Max absolute difference:", torch.max(abs_diff))
else:
    print(f"No generated tokens to compare (input_length: {input_length}, total_length: {total_length})")


# Detailed token-by-token comparison for generated tokens only
print("\n--- Token-by-Token Comparison (Generated Tokens Only) ---")
input_length = generation_data["input_lengths"][0].item()
total_length = min(vllm_logprobs_data["logprobs"].shape[1], megatron_generation_data["logprobs"].shape[1])

if total_length > input_length:
    # Get generated tokens and logprobs
    if "input_ids" in megatron_generation_data:
        generated_tokens = megatron_generation_data["input_ids"][0, input_length:total_length]
    else:
        generated_tokens = torch.arange(input_length, total_length)  # Fallback if no input_ids
    
    vllm_generated_logprobs = vllm_logprobs_data["logprobs"][0, input_length:total_length]
    megatron_generated_logprobs = megatron_generation_data["logprobs"][0, input_length:total_length]

    print(f"{'Token':<15} {'Token ID':<10} {'Position':<10} {'vLLM':<12} {'Megatron':<12} {'Diff':<12}")
    print("-" * 75)
    
    for i, pos in enumerate(range(input_length, total_length)):
        if "input_ids" in megatron_generation_data:
            token_id = generated_tokens[i].item()
            token_text = tokenizer.decode([token_id])
        else:
            token_id = f"pos_{pos}"
            token_text = f"tok_{pos}"
            
        vllm_lp = vllm_generated_logprobs[i].item()
        megatron_lp = megatron_generated_logprobs[i].item()
        diff = abs(vllm_lp - megatron_lp)
        print(f"{token_text:<15} {token_id:<10} {pos:<10} {vllm_lp:<12.6f} {megatron_lp:<12.6f} {diff:<12.6f}")
else:
    print("No generated tokens to compare in detail.")

# Final cleanup
print("\n--- Cleaning up ---")
vllm_inference_policy.shutdown()
print("Script completed successfully!")