#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=32

huggingface-cli login --token $HF_TOKEN 

n_gen_nodes=16
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

account=coreai_dlalgo_nemorl
# NRL_NSYS_WORKER_PATTERNS="*policy*" \
# NRL_NSYS_PROFILE_STEP_RANGE=2:3 \
# RAY_LOG_SYNC_FREQUENCY=30 \

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_235B_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.async_grpo.enabled=true \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.max_total_sequence_length=8192 \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=16 \
grpo.num_generations_per_prompt=32 \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=512 \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-refit' \
logger.wandb.name='async-qwen-235B-8K-GBS512-opt1234-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
NRL_FORCE_REBUILD_VENVS=true \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=01:00:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \
# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_NSYS_PROFILE_STEP_RANGE=2:3 \
# RAY_LOG_SYNC_FREQUENCY=30 \