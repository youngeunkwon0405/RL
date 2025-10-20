#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=4

huggingface-cli login --token $HF_TOKEN 

n_gen_nodes=2
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

g_tp=4
g_ep=1

account=coreai_dlalgo_nemorl

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.async_grpo.enabled=true \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.generation.vllm_cfg.expert_parallel_size=${g_ep} \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=8 \
grpo.num_generations_per_prompt=8 \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=64 \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-refit' \
logger.wandb.name='async-qwen-30B-test_new_default_st3_buf0.01-GTP${g_tp}-GEP${g_ep}-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
NRL_REFIT_BUFFER_MEMORY_RATIO=0.01 \
NRL_REFIT_NUM_BUFFERS=3 \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
NRL_NSYS_PROFILE_STEP_RANGE=2:3 \
RAY_LOG_SYNC_FREQUENCY=30 \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=00:30:00 \
    --gres=gpu:8 \
    ray.sub

    # --ntasks-per-node=8 \