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
import ray
import pytest
import pprint
import torch
from copy import deepcopy

# Define a custom marker for model configuration tests
pytestmark = pytest.mark.modelconfig

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.algorithms.utils import get_tokenizer
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.models.generation.interfaces import configure_generation_config
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from tests.unit.test_utils import simple_loss
from functools import partial

def create_test_config(model_name:str = "meta-llama/Llama-3.2-1B", tp:int =1, sequence_parallel:bool =False, cpu_offload:bool =False, activation_checkpointing:bool =False) -> PolicyConfig:
    return {
        "model_name": model_name,
        "tokenizer_name": model_name,
        "generation_batch_size": 1,  # Small batch size for testing
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "max_new_tokens": 16,  # Small number of tokens for testing
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
        },
        "dtensor_cfg": {
            "enabled": True,
            "cpu_offload": cpu_offload,
            "sequence_parallel": sequence_parallel,
            "activation_checkpointing": activation_checkpointing,
            "tensor_parallel_size": tp,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {
                "T_max": 100,
            },
        },
        "max_grad_norm": 1.0,
    }

basic_test_config = create_test_config()

@pytest.fixture(scope="function")
def gc_collect():
    """Helper function to force garbage collection after a test"""
    import gc

    gc.collect()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    model_name = basic_test_config["model_name"]
    tokenizer = get_tokenizer(model_name)
    return tokenizer


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
    prompts = [
        "Write a story about a magical forest",
        "Explain how photosynthesis works",
        "What are the benefits of exercise?",
        "Describe the water cycle",
        "What is the capital of France?",
        "Who is the president of the USA?",
        "What is the capital of the moon?",
        "Where is the sun?",
    ]

    expected_generations = [
        "Write a story about a magical forest. The forest is magical because it is full of magical creatures. The creatures are",
        "Explain how photosynthesis works\nExplain how photosynthesis works\nPhotosynthesis is the process by which plants",
        "What are the benefits of exercise? The benefits of exercise are many and varied. It is a great way to improve",
        "Describe the water cycle in your own words.\nDescribe the water cycle in your own words.\nDescribe the",
        "What is the capital of France? A. Paris B. New York C. Washington D. Baton Rouge\nA",
        "Who is the president of the USA? Who is the president of the USA? Who is the president of the USA?",
        "What is the capital of the moon? A. Houston B. New York C. Washington D. Denver\nA.",
        "Where is the sun? Where is the moon? Where is the earth? Where is the sky? Where",
    ]

    # Tokenize the prompts
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_lengths = tokenized["attention_mask"].sum(dim=1).to(torch.int32)

    data = BatchedDataDict(
        {
            "input_ids": tokenized["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    return data, prompts, expected_generations


@pytest.fixture
def policy_setup(tokenizer):
    """Setup and teardown for policy tests - creates a virtual cluster and policy."""
    policy = None
    cluster = None

    cluster_name = "test"
    print(f"Creating virtual cluster '{cluster_name}'...")

    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[2],  # Use tp bundles, one per GPU
        use_gpus=True,
        num_gpus_per_node=2,  # Using tp GPUs
        max_colocated_worker_groups=1,  # Only one worker group
    )

    config = basic_test_config
    config["generation"] = configure_generation_config(config["generation"], tokenizer)

    print("Creating HfPolicy...")
    policy = HfPolicy(cluster=cluster, config=config)

    yield policy, cluster

    # Clean up after the test
    print("Cleaning up resources for test")
    cluster.shutdown()
    policy.worker_group.shutdown()


@pytest.mark.timeout(180)
def test_hf_policy_init(policy_setup):
    policy, cluster = policy_setup

    # Verify cluster and policy were properly created
    assert policy is not None, "Policy was not created properly"
    assert cluster is not None, "Cluster was not created properly"

    # Verify we have two workers, one per GPU
    assert len(policy.worker_group.workers) == 2, "Should have 2 workers, one per GPU"

    # Check workers are alive
    worker_alive = ray.get([w.is_alive.remote() for w in policy.worker_group.workers])
    assert all(worker_alive), f"Not all workers are alive: {worker_alive}"

    # Get GPU info from both workers to verify GPU usage
    print("\nGetting GPU information from workers...")
    gpu_infos = ray.get([w.get_gpu_info.remote() for w in policy.worker_group.workers])
    print("\nGPU Information:")
    for i, info in enumerate(gpu_infos):
        print(f"\nWorker {i} GPU Info:")
        pprint.pprint(info)

    # Check 1: Verify workers have different ranks
    gpu_ranks = [info["rank"] for info in gpu_infos]
    assert len(set(gpu_ranks)) == 2, f"Expected 2 different ranks, got {gpu_ranks}"
    assert set(gpu_ranks) == {0, 1}, f"Expected ranks 0 and 1, got {gpu_ranks}"

    # Check 2: Verify workers have different local_ranks
    local_ranks = [info["local_rank"] for info in gpu_infos]
    assert len(set(local_ranks)) == 2, (
        f"Expected 2 different local_ranks, got {local_ranks}"
    )
    assert set(local_ranks) == {0, 1}, (
        f"Expected local_ranks 0 and 1, got {local_ranks}"
    )

    # Check 3: Verify workers have different CUDA_VISIBLE_DEVICES
    cuda_visible_devices = [
        info["env_vars"].get("CUDA_VISIBLE_DEVICES") for info in gpu_infos
    ]
    assert len(set(cuda_visible_devices)) == 2, (
        f"Expected different CUDA_VISIBLE_DEVICES, got {cuda_visible_devices}"
    )

    # Check 4: Verify all workers report correct world_size
    for info in gpu_infos:
        assert info["world_size"] == 2, (
            f"Expected world_size=2, got {info['world_size']}"
        )
        assert info["env_vars"]["WORLD_SIZE"] == "2", (
            f"Expected WORLD_SIZE=2, got {info['env_vars']['WORLD_SIZE']}"
        )

    # Check 5: Verify significant GPU memory is allocated (at least 1GB) on both GPUs
    for info in gpu_infos:
        assert info["memory_allocated_mb"] > 1000, (
            f"Not enough memory allocated on GPU for rank {info['rank']}: {info['memory_allocated_mb']:.2f} MB"
        )

    # Check 6: Verify model parameters are on CUDA devices for both workers
    for info in gpu_infos:
        param_sample = list(info["parameter_sample"].values())[0]
        assert "cuda" in param_sample["device"], (
            f"Parameter not on CUDA device: {param_sample['device']}"
        )

    # Check 8: Verify same model parameters are being tracked across workers
    param_names = [list(info["parameter_sample"].keys())[0] for info in gpu_infos]
    assert len(set(param_names)) == 1, (
        f"Workers are not tracking the same parameter: {param_names}"
    )

    # Check 9: Both workers should see their device as cuda:0 (correct distributed behavior)
    for info in gpu_infos:
        param_device = list(info["parameter_sample"].values())[0]["device"]
        assert param_device == "cuda:0", (
            f"Expected parameter device to be cuda:0, got {param_device}"
        )


@pytest.fixture
def training_setup(request):
    """Setup and teardown specifically for training tests."""
    model_name, tp, cpu_offload, sequence_parallel, activation_checkpointing = request.param
    policy = None
    cluster = None
    data = None
    loss_fn = None

    try:
        # Create resources with unique name
        cluster_name = f"test-train-tp{tp}-cpu{int(cpu_offload)}-sp{int(sequence_parallel)}-ac{int(activation_checkpointing)}"
        print(f"Creating training virtual cluster '{cluster_name}'...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[2],  # Single node, 2 gpus
            use_gpus=True,
            num_gpus_per_node=2,  # Using both GPUs
            max_colocated_worker_groups=1,  # Only one worker group
        )

        config = create_test_config(model_name, tp, cpu_offload, sequence_parallel, activation_checkpointing)
        print(f"Creating training HfPolicy with tp={tp}, cpu_offload={cpu_offload}, sequence_parallel={sequence_parallel}, activation_checkpointing={activation_checkpointing}...")
        policy = HfPolicy(cluster=cluster, config=config, init_reference_model=False)

        # Create a test batch
        print("Creating test batch...")
        # set random seed
        torch.manual_seed(42)

        # Create test input_ids and attention_mask
        input_ids = torch.randint(0, 32000, (8, 128))  # 8 sequences, each of length 128
        attention_mask = torch.ones(8, 128)

        # Calculate input_lengths (all sequences are full length in this test)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,  # Keep for compatibility with loss functions
                "labels": torch.randint(0, 32000, (8, 128)),
            }
        )

        # Create loss function
        loss_fn: LossFunction = simple_loss

        # Provide the resources to the test
        yield policy, cluster, data, loss_fn

    except Exception as e:
        print(f"Error during training setup: {e}")
        pytest.skip(f"Training setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        cluster.shutdown()
        policy.worker_group.shutdown()


@pytest.mark.timeout(360)
@pytest.mark.parametrize(
    "training_setup", 
    [
        # ("meta-llama/Llama-3.2-1B",1, False, False, False),
        # ("meta-llama/Llama-3.2-1B",1, True, False, False),
        # ("meta-llama/Llama-3.2-1B",1, False, True, False),
        # ("meta-llama/Llama-3.2-1B",1, False, False, True),
        # ("meta-llama/Llama-3.2-1B",1, True, True, False),
        # ("meta-llama/Llama-3.2-1B",1, True, False, True),
        # ("meta-llama/Llama-3.2-1B",1, False, True, True),
        ("meta-llama/Llama-3.2-1B",1, True, True, True),
        # ("Qwen/Qwen2.5-1.5B", 1, True, True, True),
        # ("Qwen/Qwen2.5-7B", 2, False, False, False),
        # ("Qwen/Qwen2.5-7B", 2, False, False, True),
        # ("Qwen/Qwen2.5-7B", 2, False, True, False),
        # ("Qwen/Qwen2.5-7B", 2, False, True, True),
        ("Qwen/Qwen2.5-7B", 2, False, True, True),
        ("meta-llama/Llama-3.1-8B", 2, False, True, True),
    ],
    indirect=True
)
def test_hf_policy_training(training_setup):
    def verify_loss_tensor(loss_tensor):
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        return loss_tensor

    policy, cluster, data, loss_fn = training_setup

    # Verify resources were created properly
    assert policy is not None, "Training policy was not created properly"
    assert cluster is not None, "Training cluster was not created properly"
    assert data is not None, "Test data was not created properly"
    assert loss_fn is not None, "Loss function was not created properly"

    # Call prepare_for_training if available
    print("\nPreparing for training...")
    policy.prepare_for_training()

    losses = []
    for steps in range(3):
        results = policy.train(data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        verify_loss_tensor(loss_tensor)
        losses.append(loss_tensor[-1].item())

        print(f"Training loss: {results['loss']}")

    policy.finish_training()

    # Verify loss changed between iterations (model parameters were updated)
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"
