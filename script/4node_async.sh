#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=4

# grpo_math_8b uses Llama-3.1-8B-Instruct model
HF_HOME=/lustre/fsw/coreai_dlalgo_llm/youngeunk/hf_home
HF_DATASETS_CACHE=/lustre/fsw/coreai_dlalgo_llm/youngeunk/hf_home/cache
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY

huggingface-cli login --token $HF_TOKEN 


COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml policy.generation.colocated.enabled=false policy.generation.colocated.resources.num_nodes=2 policy.generation.colocated.resources.gpus_per_node=8 cluster.num_nodes=4 policy.generation.vllm_cfg.async_engine=true checkpointing.checkpoint_dir='results/llama8b_4nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
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
    --time=0:20:0 \
    --ntasks-per-node=8 \
    ray.sub

    # --gres=gpu:8 \
