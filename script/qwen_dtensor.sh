#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=4

# grpo_math_8b uses Llama-3.1-8B-Instruct model
# HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home
# HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache
# HF_TOKEN=$HF_TOKEN
# WANDB_API_KEY=$WANDB_API_KEY

huggingface-cli login --token $HF_TOKEN 
# LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu \

# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_FORCE_REBUILD_VENVS=true \
# --config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
# NRL_NSYS_WORKER_PATTERNS="*policy*" \
# NRL_NSYS_PROFILE_STEP_RANGE=4:5 \
# RAY_LOG_SYNC_FREQUENCY=30 \

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/async_grpo_math_qwen30ba3b.yaml \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=2 \
policy.generation.colocated.resources.gpus_per_node=8 \
cluster.num_nodes=4 \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.max_num_steps=30 \
logger.wandb_enabled=True \
logger.wandb.project='async-test' \
logger.wandb.name='async-qwen-30B-2T2G-nsys'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
NRL_FORCE_REBUILD_VENVS=true \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=coreai_dlalgo_nemorl-async.test \
    --partition=batch \
    --time=00:30:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \