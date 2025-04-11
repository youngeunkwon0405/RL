# NVIDIA NeMo-Reinforcer: Scalable and Efficient Post-Training for Large Language Models

<!-- markdown all in one -->

**NVIDIA NeMo-Reinforcer** is a powerful and scalable post-training library designed to efficiently align and fine-tune large language models (LLMs), ranging from small models to those exceeding 100 billion parameters. It leverages distributed computing frameworks to scale seamlessly from a single GPU to hundreds, enabling efficient training and experimentation.

This library provides a flexible and modular framework for implementing various post-training techniques, with a focus on Reinforcement Learning from Human Preferences (RLHF) and Supervised Fine-Tuning (SFT).

## Table of Contents

- [Key Features](#key-features)
- [Install NeMo-Reinforcer](#installation)
- [Quick Start](#quick-start)
  - [Post-Train with Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
    - [Run Single Node](#single-node)
    - [Run Multi-node](#multi-node)
  - [Post-Train with Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
    - [Run Single Node](#single-node-grpo)
    - [Run Multi-node](#multi-node-grpo)
- [Set Up Clusters](#cluster-setup)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Licenses](#license)
- [Citation](#citation)
- [Support](#support)

## Key Features

_✅ Available Now | 🔜 Coming Soon (v0.2)_

- ✅ **Fast Generation:** Utilizes vLLM backend for optimized inference during evaluation and rollout.
- ✅ **HuggingFace Integration:** Seamlessly integrates with Hugging Face Transformers, supporting a wide range of pre-trained models (e.g., Qwen1.5, Llama models up to 8B parameters).
- ✅ **Scalable Distributed Training:** Leverages Fully Sharded Data Parallelism (FSDP) and a Ray-based infrastructure for efficient multi-GPU and multi-node training.
- ✅ **Multi-Environment Support:** Enables training across diverse environments and datasets.
- ✅ **Reinforcement Learning Algorithms:** Implements Group Relative Policy Optimization (GRPO) for effective preference alignment.
- ✅ **Supervised Fine-Tuning (SFT):** Supports standard supervised fine-tuning for instruction following and task adaptation.
- ✅ **Worker Isolation:** Ensures process isolation between RL actors, preventing unintended global state interference.
- 🔜 **Larger Model Support:** Native PyTorch support for models up to 70B parameters.
- 🔜 **Advanced Parallelism Techniques:** Implementation of FSDP2, Tensor Parallelism (TP), Pipeline Parallelism (PP), and sequence packing for enhanced training efficiency.
- 🔜 **Environment Isolation:** Provides dependency isolation between different components of the training pipeline.
- 🔜 **Direct Preference Optimization (DPO):** Integration of the Direct Preference Optimization algorithm for more direct preference learning.

## Install NeMo-Reinforcer

For a streamlined setup, we recommend using `uv`. Ensure you have Python 3.12 or a compatible version installed.

```sh
# Install uv for faster package management
pip install uv

# Create a virtual environment with Python 3.12
uv venv -p python3.12 .venv

# Activate the virtual environment (optional, but recommended for consistency)
# source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows

# Install NeMo-Reinforcer with vLLM support
uv pip install -e .[vllm]

# To install with development and testing dependencies:
# uv pip install -e '.[dev,test]'

# Running scripts with `uv run` ensures a consistent environment.
# Example: uv run python examples/run_grpo_math.py
```

**Important Notes:**

- It is generally recommended **not to explicitly activate the virtual environment** when using `uv`. Instead, use `uv run <command>` to execute scripts within the managed environment. This helps maintain consistency across different shells and sessions.
- Ensure you have the necessary CUDA drivers and PyTorch installed compatible with your hardware.

## Quick Start

Before running any experiments, remember to set your `HF_HOME` environment variable and your `WANDB_API_KEY` if you intend to use Weights & Biases for logging. For accessing Llama models, you might also need to log in using `huggingface-cli login`.

## Post-Train with Supervised Fine-Tuning (SFT)

We provide an example SFT experiment using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).

#### Run Single Node SFT

The default SFT configuration is set to run on a single GPU. To start the experiment:

```sh
uv run python examples/run_sft.py
```

This command will fine-tune the `Llama3.2-1B` model on the SQuAD dataset using a single GPU.

To utilize more GPUs on a single node, you can modify the cluster configuration and potentially adjust the model and batch size:

```sh
uv run python examples/run_sft.py \
  policy.model_name="meta-llama/Meta-Llama-3-8B" \
  policy.train_global_batch_size=128 \
  sft.val_global_batch_size=128 \
  cluster.gpus_per_node=8
```

For a comprehensive list of configurable parameters, refer to the [sft.yaml](https://www.google.com/search?q=examples/configs/sft.yaml) file.

#### Run Multi-node SFT

For distributed training across multiple compute nodes, ensure that the `UV_CACHE_DIR` is set to a shared directory accessible by all worker nodes before executing any `uv run` commands.

```sh
export UV_CACHE_DIR=/path/that/all/workers/can/access/uv_cache
```

The following is an example Slurm script for launching a multi-node SFT experiment with the `Llama-3.1-8B` model:

```sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=sft_llama8b_2nodes
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:8

NUM_ACTOR_NODES=$SLURM_JOB_NUM_NODES
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

COMMAND="uv pip install -e .; uv run ./examples/run_sft.py --config examples/configs/sft.yaml cluster.num_nodes=$NUM_ACTOR_NODES cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='sft-llama8b'"
RAY_DEDUP_LOGS=0
UV_CACHE_DIR=YOUR_UV_CACHE_DIR
CONTAINER=YOUR_CONTAINER # Replace with your container if using one
MOUNTS="$PWD:$PWD"

srun --nodes=$NUM_ACTOR_NODES --ntasks-per-node=1 \
  --gres=gpu:8 \
  --job-name=${SLURM_JOB_NAME} \
  bash -c "source .venv/bin/activate && ${COMMAND}" # If not using uv run directly
```

**Note:** Adapt the Slurm parameters (`--account`, `--partition`, `--job-name`, `--time`, `--gres`) according to your cluster configuration. You might need to adjust the command if you are not directly using `uv run` within the Slurm script.

### Post-Train with Group Relative Policy Optimization (GRPO)

We provide a reference GRPO experiment configuration for training on math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

#### Run Single Node GRPO

To run the GRPO math example on a single GPU using the `Llama-3.2-1B-Instruct` model:

```sh
uv run python examples/run_grpo_math.py
```

This command utilizes the default configuration specified in `examples/configs/grpo_math_1B.yaml`. You can override any parameter in the configuration file using command-line arguments. For instance, to run on 8 GPUs:

```sh
uv run python examples/run_grpo_math.py \
  cluster.gpus_per_node=8
```

Here are more examples of overriding configuration parameters:

```sh
uv run python examples/run_grpo_math.py \
  policy.model_name="Qwen/Qwen2-1.5B" \
  checkpointing.checkpoint_dir="results/qwen1_5b_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-qwen1_5b_math" \
  logger.num_val_samples_to_print=10
```

#### Run Multi-node GRPO

The following is an example Slurm script for launching a multi-node GRPO experiment with the `Llama-3.1-8B-Instruct` model:

```sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=grpo_llama8b_2nodes
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:8

NUM_ACTOR_NODES=$SLURM_JOB_NUM_NODES
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

COMMAND="uv pip install -e .; uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=$NUM_ACTOR_NODES checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'"
RAY_DEDUP_LOGS=0
UV_CACHE_DIR=YOUR_UV_CACHE_DIR
CONTAINER=YOUR_CONTAINER # Replace with your container if using one
MOUNTS="$PWD:$PWD"

srun --nodes=$NUM_ACTOR_NODES --ntasks-per-node=1 \
  --gres=gpu:8 \
  --job-name=${SLURM_JOB_NAME} \
  bash -c "source .venv/bin/activate && ${COMMAND}" # If not using uv run directly
```

**Note:** Adjust the Slurm directives according to your cluster setup.

## Set Up Clusters

For detailed instructions on how to set up and launch NeMo-Reinforcer on Slurm or Kubernetes clusters, please refer to the dedicated [Cluster Setup](https://www.google.com/search?q=docs/cluster.md) documentation.

## Documentation

Comprehensive documentation, including API references and more detailed explanations of concepts and functionalities, will be available soon. Stay tuned for updates\!

## Contributing

We welcome contributions to NeMo-Reinforcer\! Please see our [Contributing Guidelines](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for more information on how to get involved.

## Licenses

NVIDIA NeMo-Reinforcer is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme).

NeMo is licensed under the [NVIDIA AI PRODUCT AGREEMENT](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/). By pulling and using the container, you accept the terms and conditions of this license.

## Support

For questions, bug reports, or feature requests, please open an issue on our [GitHub repository](https://github.com/NVIDIA/NeMo-Reinforcer/issues).