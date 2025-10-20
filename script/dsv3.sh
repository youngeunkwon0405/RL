#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=64

huggingface-cli login --token $HF_TOKEN 

account=coreai_dlalgo_nemorl

n_gen_nodes=32
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

# NRL_REFIT_BUFFER_MEMORY_RATIO=0.02 \
# NRL_REFIT_NUM_BUFFERS=3 \
# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_NSYS_PROFILE_STEP_RANGE=2:3 \
# RAY_LOG_SYNC_FREQUENCY=30 \

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_dsv3_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=1 \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=32 \
policy.generation.vllm_cfg.expert_parallel_size=1 \
policy.megatron_cfg.pipeline_model_parallel_size=16 \
policy.megatron_cfg.expert_model_parallel_size=16 \
policy.megatron_cfg.tensor_model_parallel_size=8 \
policy.megatron_cfg.sequence_parallel=True \
policy.megatron_cfg.num_layers_in_first_pipeline_stage=3 \
policy.megatron_cfg.num_layers_in_last_pipeline_stage=2 \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-refit' \
logger.wandb.name='async-grpo-dsv3-test-TP32-1020-overlap-branch-head-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
NRL_FORCE_REBUILD_VENVS=true \
NCCL_NVLS_ENABLE=0 \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=00:50:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \