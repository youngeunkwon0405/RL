# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=2

# Set up virtual environment directory
VENV_DIR="$PWD/reinforcer_venv"
mkdir -p $VENV_DIR

# Set environment variables for UV and virtual environment
export UV_CACHE_DIR="$PWD/uv_cache"
export UV_LINK_MODE=copy
export VENV_DIR=$VENV_DIR

# Set vLLM port range to avoid conflicts
export VLLM_PORT_RANGE="20000-30000"


JOB_NAME="grpo_nano_large_context_aggressive_L1"

source ~/secrets
echo WANDB_API_KEY=${WANDB_API_KEY}

CONTAINER='gitlab-master.nvidia.com/deci/research/lit-llama/rl_uv_amnon:latest' \
MOUNTS="/lustre:/lustre,$UV_CACHE_DIR:/home/ray/.cache/uv,$VENV_DIR:/opt/reinforcer_venv" \
COMMAND="uv run python tests/run_grpo_math_nano.py --config tests/configs/grpo_math_8B_nano_large_context_aggressive_L1.yaml cluster.num_nodes=${NUM_ACTOR_NODES} logger.wandb.name=${JOB_NAME} checkpointing.checkpoint_dir=results/${JOB_NAME} logger.wandb.id=${JOB_NAME}" WANDB_API_KEY=${WANDB_API_KEY}  \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account='llmservice_deci_llm' \
    --job-name="${JOB_NAME}" \
    --partition='batch' \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub