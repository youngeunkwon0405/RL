#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

# grpo_math_8b uses Llama-3.1-8B-Instruct model
HF_HOME=${HF_HOME}
HF_DATASETS_CACHE=${HF_HOME}/cache
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY

# COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
CONTAINER=/home/youngeunk/lustre_home/sqsh/nemo_rl.sqsh \
HF_HOME=${HF_HOME} \
HF_DATASETS_CACHE=${HF_HOME}/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_llm \
    --job-name=coreai_dlalgo_llm-rl.test \
    --partition=batch \
    --gres=gpu:8 \
    --time=04:00:00 \
    ray.sub

    # --ntasks-per-node=8 \
