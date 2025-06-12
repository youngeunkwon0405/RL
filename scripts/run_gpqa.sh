CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/home/zhaochengz/lustre/reinforcer/results/Qwen2.5XXX"}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-"-1"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
if [ -n "$TAG" ]; then
    TAG=${TAG}_ 
fi

set -e
model_name=$(basename "$CHECKPOINT_DIR")
summary_file=${SUMMARY_FILE:-"logs/${model_name}_${TAG}summary.txt"}
hyperparameters="temperature: $TEMPERATURE, top_p: $TOP_P, top_k: $TOP_K, max_model_len: $MAX_MODEL_LEN"
if [ ! -f "$summary_file" ]; then
    echo "$hyperparameters" >> "$summary_file"
elif ! grep -Fq "$hyperparameters" "$summary_file"; then
    echo "Found existing evaluation summary $summary_file, but the hyperparameters don't match the current script."
    echo "Please manually delete the summary file to start a new evaluation run."
    exit 1
else
    echo "Resume from existing summary $summary_file."
fi

step_dirs=$(ls -d ${CHECKPOINT_DIR}/hf_step_* | sort -V)
for step_dir in $step_dirs; do
    step_dir=$(realpath "$step_dir")
    step=$(basename "$step_dir") # hf_step_*
    record="model_name='${step}'"
    step=${step#hf_} # step_*
    output_file=logs/${model_name}_${TAG}${step}.txt
    if [ -f "$output_file" ]; then
        output_line_num=$(grep -n "$record" "$output_file" | head -n1 | cut -d: -f1)
        summary_line_num=$(grep -n "$record" "$summary_file" | head -n1 | cut -d: -f1)
        if [ -n "$output_line_num" ]; then
            # if output contains a record
            if [ -n "$summary_line_num" ]; then
                # if summary also contains a record
                output_record=$(tail -n +"$((output_line_num + 1))" "$output_file" | head -n2)
                summary_record=$(tail -n +"$((summary_line_num + 1))" "$summary_file" | head -n2)
                # if the record is already in the summary, skip
                if [ "$output_record" == "$summary_record" ]; then
                    continue
                fi
            fi
            echo "Found unmatched output $output_file and summary $summary_file."
            echo "Please manually rename or delete the old output file."
            exit 1
        else
            # output doesn't contain a record
            : # proceed to overwrite
        fi
    fi
    uv run python examples/run_eval.py --config examples/configs/eval_gpqa.yaml \
        generation.temperature=$TEMPERATURE generation.top_p=$TOP_P generation.top_k=$TOP_K \
        generation.model_name="$step_dir" generation.vllm_cfg.max_model_len=$MAX_MODEL_LEN \
        | tee "$output_file"
    # add evaluation results to summary
    line_num=$(grep -a -n "============================================================" "$output_file" \
        | awk -F: '{print $1}' | tail -n 2 | head -n 1)
    if [ -n "$line_num" ]; then
        tail -n +$line_num "$output_file" >> "$summary_file"
    else
        echo "Can't find evaluation record in $output_file. Skipping it in summary."
    fi
done