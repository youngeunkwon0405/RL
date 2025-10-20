#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=4

huggingface-cli login --token $HF_TOKEN 

# NRL_REFIT_BUFFER_MEMORY_RATIO=0.01 \
# NRL_REFIT_NUM_BUFFERS=3 \

account=coreai_dlalgo_nemorl

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
cluster.num_nodes=4 \
grpo.async_grpo.enabled=true \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=2 \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=8 \
grpo.val_period=1000 \
checkpointing.enabled=false \
policy.max_total_sequence_length=256 \
grpo.num_prompts_per_step=8 \
grpo.num_generations_per_prompt=4 \
policy.train_global_batch_size=32 \
grpo.max_num_steps=30 \
logger.wandb_enabled=True \
logger.wandb.project='async-refit-test-1020' \
logger.wandb.name='async-qwen-30B-2T2G-no-cuda_sync'" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2b55598e.squashfs \
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