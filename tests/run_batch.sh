# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=8

# Set up virtual environment directory
VENV_DIR="$PWD/reinforcer_venv"
#mkdir -p $VENV_DIR

# Set environment variables for UV and virtual environment
export UV_CACHE_DIR="$PWD/uv_cache"
export UV_LINK_MODE=copy
export VENV_DIR=$VENV_DIR

# Set vLLM port range to avoid conflicts
export VLLM_PORT_RANGE="20000-30000"


JOB_NAME=${JOB_NAME:-"RL2_1b_MATH2_debug_no_importance"}
LR=${LR:-"5.0e-6"}
KL=${KL:-"0.001"}
TEMP1=${TEMP1:-"0.0"}
TEMP2=${TEMP2:-"1.0"}


#echo $COMMAND
# grpo_math_8b uses Llama-3.1-8B-Instruct model  policy.model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#COMMAND="uv run ./examples/run_grpo_math_speculative.py --config examples/configs/grpo_math_1B.yaml cluster.num_nodes=${NUM_ACTOR_NODES}" \
CONTAINER='gitlab-master.nvidia.com/deci/research/lit-llama/rl_uv_amnon:latest' \
MOUNTS="/lustre:/lustre,$VENV_DIR:/opt/reinforcer_venv,$UV_CACHE_DIR:/home/ray/.cache/uv" \
COMMAND="uv run ./examples/run_grpo_math_speculative.py --config examples/configs/grpo_math_1B.yaml cluster.num_nodes=${NUM_ACTOR_NODES} policy.optimizer.kwargs.lr=${LR} \
 loss_fn.reference_policy_kl_penalty=${KL} logger.wandb.project=amnon_rl_experiments_sweep logger.wandb.id=${JOB_NAME} \
 logger.wandb.name=${JOB_NAME} checkpointing.checkpoint_dir=results/${JOB_NAME} target_policy.generation.temperature=${TEMP1} policy.generation.temperature=${TEMP2}" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account='llmservice_deci_llm' \
    --job-name="${JOB_NAME}" \
    --partition='batch' \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub
