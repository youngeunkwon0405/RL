#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=5

# grpo_math_8b uses Llama-3.1-8B-Instruct model
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY

huggingface-cli login --token $HF_TOKEN 


COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/async_grpo_math_8B.yaml \
cluster.num_nodes=5 \
grpo.val_period=1000 \
checkpointing.enabled=false \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=4 \
policy.generation.colocated.resources.gpus_per_node=8 \
grpo.max_num_steps=30 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-profiling' \
logger.wandb.name='async-grpo-8B-1T4G'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2aea5add.squashfs \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=coreai_dlalgo_nemorl.test \
    --partition=batch \
    --time=01:30:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \