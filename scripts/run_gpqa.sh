CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/home/zhaochengz/lustre/reinforcer/results/Qwen2.5XXX"}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-"-1"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

set -e
model_name=$(basename "$CHECKPOINT_DIR")
summary_file=logs/${model_name}_summary.txt
hyperparameters="temperature: $TEMPERATURE, top_p: $TOP_P, top_k: $TOP_K, max_model_len: $MAX_MODEL_LEN"
if [ ! -f $summary_file ]; then
    echo $hyperparameters >> $summary_file
elif ! grep -Fq "$hyperparameters" $summary_file; then
    echo "Found existing evaluation summary $summary_file, but the hyperparameters don't match the current script."
    echo "Please manually delete the sumamry file to start a new evaluation run."
    exit 1
else
    echo "Resume from existing summary $summary_file."
fi

step_dirs=$(ls -d ${CHECKPOINT_DIR}/hf_step_* | sort -V)
for step_dir in $step_dirs; do
    step_dir=$(realpath "$step_dir")
    step=$(basename "$step_dir")
    step=${step#hf_}
    output_file=logs/${model_name}_${step}.txt
    # if results are already in the summary, skip
    if [ -f $output_file ] && grep -q "$step" $summary_file; then
       continue
    fi
    uv run python examples/run_eval.py --config examples/configs/eval_gpqa.yaml \
        generation.temperature=$TEMPERATURE generation.top_p=$TOP_P generation.top_k=$TOP_K \
        generation.model_name="$step_dir" generation.vllm_cfg.max_model_len=$MAX_MODEL_LEN \
        | tee $output_file
    # add evaluation results to summary
    line_num=$(grep -n "============================================================" $output_file \
        | awk -F: '{print $1}' | tail -n 2 | head -n 1)
    tail -n +$line_num $output_file >> $summary_file
done