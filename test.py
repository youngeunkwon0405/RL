import time
import torch

from nemo_reinforcer.algorithms.grpo import refit_policy_generation
from nemo_reinforcer.algorithms.utils import get_tokenizer
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import init_ray, RayVirtualCluster
from nemo_reinforcer.models.generation.interfaces import configure_generation_config
from nemo_reinforcer.models.generation.vllm import VllmGeneration
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
import yaml


# GPU_COUNT = 2
# TP = 1
# model_name = "/home/scratch.trt_llm_data/llm-models/llama-3.2-models/Llama-3.2-1B"

GPU_COUNT = 4
TP = 4
#model_name = "Qwen/Qwen2.5-7B"
model_name = "/home/scratch.trt_llm_data/llm-models/Qwen2.5-7B-Instruct"

config: PolicyConfig = {
    "model_name": model_name,
    "tokenizer_name": model_name,
    "generation_batch_size": 1,  # Small batch size for testing
    "train_global_batch_size": 4,
    "train_micro_batch_size": 1,
    "learning_rate": 5e-6,
    "logprob_batch_size": 1,
    "precision": "float32",
    "generation": {
        "backend": "vllm",
        "model_name": model_name,
        "dtype": "bfloat16",
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "tensor_parallel_size": TP,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 1024,
            "skip_tokenizer_init": True,
            "load_format": "dummy",
        },
    },
    "dtensor_cfg": {
        "enabled": False,
    },
    "activation_checkpointing_enabled": False,
    "fsdp_offload_enabled": False,
    "optimizer": {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "scheduler": {
        "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "kwargs": {
            "T_max": 100,
        },
    },
}


def setup():
    global cluster, policy, policy_generation, tokenizer

    tokenizer = get_tokenizer({"name": model_name})
    config["generation"] = configure_generation_config(config["generation"], tokenizer)

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[GPU_COUNT],  # 1 node with x GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=GPU_COUNT,  # Use available GPUs
        name="test-cluster",
    )

    policy = HfPolicy(cluster, config, tokenizer)
    policy_generation = VllmGeneration(cluster, config["generation"])
    policy_generation.finish_generation()


def cleanup():
    global cluster, policy, policy_generation
    policy.shutdown()
    policy_generation.shutdown()
    cluster.shutdown()


def prepare_data():
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Hello, my name is",
        "The capital of France is",
        "Hello, my name is",
        "The capital of France is",
        "Hello, my name is",
        "The capital of France is",
    ]

    expected_generations = [
        "Hello, my name is Kelsey and I am a 2018 graduate of the University of Wisconsin-M",
        "The capital of France is Paris. The city is located in the north of the country. The city is",
        "Hello, my name is Kelsey and I am a 2018 graduate of the University of Wisconsin-M",
        "The capital of France is Paris. The city is located in the north of the country. The city is",
        "Hello, my name is Kelsey and I am a 2018 graduate of the University of Wisconsin-M",
        "The capital of France is Paris. The city is located in the north of the country. The city is",
        "Hello, my name is Kelsey and I am a 2018 graduate of the University of Wisconsin-M",
        "The capital of France is Paris. The city is located in the north of the country. The city is",
    ]

    # Tokenize prompts
    encodings = tokenizer(
        prompts,
        padding="max_length",
        max_length=50,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)

    # Create input data dictionary
    data = BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    return data, expected_generations


if __name__ == "__main__":
    # setup
    init_ray()
    setup()

    # prepare data
    input_data, expected_generations = prepare_data()

    # warm up
    start_time = time.time()
    refit_policy_generation(policy, policy_generation)
    policy_generation.finish_generation()
    policy.prepare_for_training()
    end_time = time.time()
    print(f"Time taken to warmup (first time to update weights): {end_time - start_time:.2f} seconds")

    # update weights
    start_time = time.time()
    refit_policy_generation(policy, policy_generation)
    end_time = time.time()
    print(f"Time taken to update weights: {end_time - start_time:.2f} seconds")

    # generate
    results = policy_generation.generate(input_data, greedy=True)
    policy_generation.finish_generation()

    # check results
    output_ids = results["output_ids"]
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # for generated_text, expected_generation in zip(generated_texts, expected_generations):
    #     assert generated_text == expected_generation, f"\n{generated_text=}\n{expected_generation=}"
    for i in range(len(generated_texts)):
        print(f"{generated_texts[i]=}")


    # cleanup
    cleanup()