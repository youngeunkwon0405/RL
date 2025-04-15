# NeMo RL: Scalable and Efficient Post-Training on NVIDIA GPUs

**NeMo RL** is a scalable and efficient post-training library designed for models ranging from 1 GPU to thousands, and from tiny to over 100 billion parameters.

What you can expect:

- **Seamless integration with Hugging Face** for ease of use, allowing users to leverage a wide range of pre-trained models and tools.
- **High-performance implementation with Megatron Core**, supporting various parallelism techniques for large models (>100B) and large context lengths.
- **Efficient resource management using Ray**, enabling scalable and flexible deployment across different hardware configurations.
- **Flexibility** with a modular design that allows easy integration and customization.
- **Comprehensive documentation** that is both detailed and user-friendly, with practical examples.

## Table of Contents

- [Key Features](#key-features)
- [Install NeMo RL](#install-nemo-rl)
- [Quickstart](#quickstart)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
    - [Run Single Node SFT](#run-single-node-sft)
    - [Run Multi-node SFT](#run-multi-node-sft)
  - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
    - [Run Single Node GRPO](#run-single-node-grpo)
    - [Run Multi-node GRPO](#run-multi-node-grpo)
- [Set Up Clusters](#set-up-clusters)
- [Contributing](#contributing)
- [Licenses](#licenses)

## Key Features

_✅ Available Now | 🔜 Coming Soon (v0.2)_

- ✅ **Fast Generation:** Utilizes vLLM backend for optimized inference during evaluation and rollout.
- ✅ **Hugging Face Integration:** Seamlessly integrates with Hugging Face Transformers, supporting a wide range of pre-trained models (e.g., Qwen1.5, Llama models up to 8B parameters).
- ✅ **Scalable Distributed Training:** Leverages Fully Sharded Data Parallelism (FSDP) and a Ray-based infrastructure for efficient multi-GPU and multi-node training.
- ✅ **Multi-Environment Support:** Enables training across diverse environments and datasets.
- ✅ **Reinforcement Learning Algorithms:** Implements Group Relative Policy Optimization (GRPO) for effective preference alignment.
- ✅ **Supervised Fine-Tuning (SFT):** Supports standard supervised fine-tuning for instruction following and task adaptation.
- ✅ **Worker Isolation:** Ensures process isolation between RL actors, preventing unintended global state interference.
- 🔜 **Larger Model Support:** Native PyTorch support for models up to 70B parameters.
- 🔜 **Advanced Parallelism Techniques:** Implementation of FSDP2, Tensor Parallelism (TP), Pipeline Parallelism (PP), and sequence packing for enhanced training efficiency.
- 🔜 **Environment Isolation:** Provides dependency isolation between different components of the training pipeline.
- 🔜 **Direct Preference Optimization (DPO):** Integration of the Direct Preference Optimization algorithm for more direct preference learning.

## Install NeMo RL

Use of the `uv` Python package manager is required for setup. Python 3.12 or a compatible version is also required.

```sh
# Install uv
pip install uv

# Create a virtual environment with Python 3.12
uv venv -p python3.12 .venv

# Activate the virtual environment (optional, but recommended for consistency)
# source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows

# Install NeMo RL with vLLM support
uv pip install -e .[vllm]

# To install with development and testing dependencies:
# uv pip install -e '.[dev,test]'

# Running scripts with `uv run` ensures a consistent environment.
# Example: uv run python examples/run_grpo_math.py
```

**Important Notes:**

- It is generally recommended **not to explicitly activate the virtual environment** when using `uv`. Instead, use `uv run <command>` to execute scripts within the managed environment. This helps maintain consistency across different shells and sessions.
- Ensure you have the necessary CUDA drivers and PyTorch installed compatible with your hardware.

## Quickstart

Before running any experiments, remember to set your `HF_HOME` environment variable and your `WANDB_API_KEY` if you intend to use Weights & Biases for logging. For accessing Llama models, you might also need to log in using `huggingface-cli login`.

## Supervised Fine-Tuning (SFT)

We provide an example SFT experiment using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).

#### Run Single Node SFT

The default SFT configuration is set to run on a single GPU. To start the experiment:

```sh
uv run python examples/run_sft.py
```

This fine-tunes the `Llama3.2-1B` model on the SQuAD dataset using a 1 GPU.

To use multiple GPUs on a single node, you can modify the cluster configuration. This adjustment will also let you potentially increase the model and batch size:

```sh
uv run python examples/run_sft.py \
  policy.model_name="meta-llama/Meta-Llama-3-8B" \
  policy.train_global_batch_size=128 \
  sft.val_global_batch_size=128 \
  cluster.gpus_per_node=8
```

Refer to `examples/configs/sft.yaml` for a full list of parameters that can be overridden.

#### Run Multi-node SFT

For distributed training across multiple nodes:

Set `UV_CACHE_DIR` to a directory that can be read from all workers before running any uv run command.
```sh
export UV_CACHE_DIR=/path/that/all/workers/can/access/uv_cache
```

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2
# Add a timestamp to make each job name unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# SFT experiment uses Llama-3.1-8B model
COMMAND="uv pip install -e .; uv run ./examples/run_sft.py --config examples/configs/sft.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='sft-llama8b'" \
RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

### Group Relative Policy Optimization (GRPO)

We provide a reference GRPO experiment configuration for training on math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

#### Run Single Node GRPO

To run GRPO on a single GPU for `Llama-3.2-1B-Instruct`:

```sh
# Run the GRPO math example using a 1B parameter model
uv run python examples/run_grpo_math.py
```

By default, this uses the configuration in `examples/configs/grpo_math_1B.yaml`. You can customize parameters with command-line overrides. For example, to run on 8 gpus,

```sh
# Run the GRPO math example using a 1B parameter model using 8 GPUs
uv run python examples/run_grpo_math.py \
  cluster.gpus_per_node=8
```

You can override any of the parameters listed in the yaml configuration file. For example,

```sh
uv run python examples/run_grpo_math.py \
  policy.model_name="Qwen/Qwen2-1.5B" \
  checkpointing.checkpoint_dir="results/qwen1_5b_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-qwen1_5b_math" \
  logger.num_val_samples_to_print=10 \
```

#### Run Multi-node GRPO

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2
# Add a timestamp to make each job name unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="uv pip install -e .; uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

## Set Up Clusters

For detailed instructions on how to set up and launch NeMo RL on Slurm or Kubernetes clusters, please refer to the dedicated [Cluster Start](docs/cluster.md) documentation.

## Contributing

We welcome contributions to NeMo RL\! Please see our [Contributing Guidelines](https://github.com/NVIDIA/reinforcer/blob/main/CONTRIBUTING.md) for more information on how to get involved.

## Licenses

NVIDIA NeMo RL is licensed under the [Apache License 2.0](https://github.com/NVIDIA/reinforcer/blob/main/LICENSE).

NeMo is licensed under the [NVIDIA AI PRODUCT AGREEMENT](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/). By pulling and using the container, you accept the terms and conditions of this license.