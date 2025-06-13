export HF_TOKEN=YOUR_HF_TOKEN
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export RAY_DEDUP_LOGS=0

export CONTAINER="/home/zhaochengz/containers/nemo-rl.sqsh"
export COMMAND="uv run examples/run_sft.py"
# store experiment logs in BASE_LOG_DIR
export BASE_LOG_DIR="/home/zhaochengz/lustre/experiments"
# recursively expand soft links
export MOUNTS="$(readlink -f /home/zhaochengz/lustre/reinforcer):$(readlink -f /home/zhaochengz/lustre/reinforcer),\
$(readlink -f /home/zhaochengz/lustre/reinforcer):/home/zhaochengz/lustre/reinforcer,\
$(readlink -f /home/zhaochengz/lustre/datasets):/home/zhaochengz/lustre/datasets,\
$(readlink -f /home/zhaochengz/lustre/experiments):/home/zhaochengz/lustre/experiments"

sbatch --account=llmservice_modelalignment_ppo --job-name=sft \
    --nodes=1 --partition=interactive --time=4:0:0 --gres=gpu:8 \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
    /home/zhaochengz/lustre/reinforcer/ray.sub