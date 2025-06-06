# Run from the root of NeMo RL repo
#CONTAINER='/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-383ed0b.squashfs' \
NUM_ACTOR_NODES=8
export WANDB_API_KEY="..."
export HF_TOKEN=...
HF_TOKEN=...\
WANDB_API_KEY="..." \
COMMAND="uv run ./examples/run_grpo_math_tools.py --config examples/configs/grpo_math_tools.yaml cluster.num_nodes=${NUM_ACTOR_NODES} cluster.gpus_per_node=8 logger.wandb_enabled=True checkpointing.checkpoint_dir='results/grpo_math_tools_qwen3_8b'" \
CONTAINER="..." \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=... \
    --job-name=... \
    --partition=... \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub

