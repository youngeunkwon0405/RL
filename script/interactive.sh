#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

# grpo_math_8b uses Llama-3.1-8B-Instruct model
HF_HOME=/lustre/fsw/coreai_dlalgo_llm/youngeunk/hf_home
HF_DATASETS_CACHE=/lustre/fsw/coreai_dlalgo_llm/youngeunk/hf_home/cache
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY

# COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
CONTAINER=/lustre/fsw/coreai_dlalgo_llm/youngeunk/sqsh/nemo_rl.sqsh \
HF_HOME=/lustre/fsw/coreai_dlalgo_llm/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/coreai_dlalgo_llm/youngeunk/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_llm \
    --job-name=coreai_dlalgo_llm-rl.test \
    --partition=batch \
    --time=03:50:00 \
    --ntasks-per-node=8 \
    ray.sub

    # --gres=gpu:8 \
