#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

bash

# grpo_math_8b uses Llama-3.1-8B-Instruct model
HF_HOME=/lustre/fs1/portfolios/coreai/users/youngeunk/hf_home
HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/users/youngeunk/hf_home/cache
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY

huggingface-cli login --token $HF_TOKEN 

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/async_grpo_math_1B.yaml \
cluster.num_nodes=2 \
cluster.gpus_per_node=8 \
grpo.val_period=1000 \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=1 \
policy.generation.colocated.resources.gpus_per_node=8 \
checkpointing.enabled=false \
policy.generation.vllm_cfg.async_engine=true \
grpo.max_num_steps=10 \
logger.wandb_enabled=True \
logger.wandb.name='async-grpo-qwen1b_math_slurm'" \
CONTAINER=/lustre/fs1/portfolios/coreai/users/youngeunk/sqsh/nemo_rl.sqsh \
HF_HOME=/lustre/fs1/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/users/youngeunk/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_llm \
    --job-name=coreai_dlalgo_llm-rl.test \
    --partition=batch \
    --time=00:30:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \
