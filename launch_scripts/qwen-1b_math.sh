# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=16

# Set up virtual environment directory
VENV_DIR="$PWD/reinforcer_venv"
#mkdir -p $VENV_DIR

# Set environment variables for UV and virtual environment
export UV_CACHE_DIR="$PWD/uv_cache"
export UV_LINK_MODE=copy
export VENV_DIR=$VENV_DIR

# Set vLLM port range to avoid conflicts
export VLLM_PORT_RANGE="20000-30000"

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_1B.yaml cluster.num_nodes=${NUM_ACTOR_NODES}" \
CONTAINER='gitlab-master.nvidia.com/deci/research/lit-llama/rl_uv_amnon:latest' \
MOUNTS="/lustre:/lustre,$VENV_DIR:/opt/reinforcer_venv,$UV_CACHE_DIR:/home/ray/.cache/uv" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account='llmservice_deci_llm' \
    --job-name='RL_1b_MATH' \
    --partition='batch' \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub