#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=12
n_gen_nodes=4

account=coreai_dlalgo_nemorl

n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

huggingface-cli login --token $HF_TOKEN 

# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_NSYS_PROFILE_STEP_RANGE=3:5 \
# RAY_LOG_SYNC_FREQUENCY=30 \


COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_8B.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=10 \
grpo.num_prompts_per_step=64 \
grpo.num_generations_per_prompt=32 \
checkpointing.enabled=true \
checkpointing.enabled=true \
checkpointing.save_period=50 \
checkpointing.checkpoint_dir="results/grpo_8b_dtensor_opt1234" \
grpo.async_grpo.enabled=true \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
grpo.max_num_steps=2000 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-refit-convergence' \
logger.wandb.name='1017-grpo-8B-opt1234-dtensor-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
NRL_FORCE_REBUILD_VENVS=true \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async-dtensor-opt1234.test \
    --partition=batch \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \