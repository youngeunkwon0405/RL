#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=4

huggingface-cli login --token $HF_TOKEN 

g_tp=4

account=coreai_dlalgo_nemorl

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.async_grpo.enabled=false \
policy.generation.vllm_cfg.async_engine=false \
policy.generation.colocated.enabled=true \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=64 \
grpo.num_generations_per_prompt=32 \
policy.train_global_batch_size=512 \
grpo.max_num_steps=100 \
logger.wandb_enabled=True \
logger.wandb.project='async-qwen-profiling' \
logger.wandb.name='sync-qwen-30B-GTP${g_tp}'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
NRL_NSYS_PROFILE_STEP_RANGE=2:4 \
RAY_LOG_SYNC_FREQUENCY=30 \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=02:00:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \