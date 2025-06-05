CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/home/zhaochengz/lustre/reinforcer/results/Qwen2.5XXX"}

set -e
step_dirs=$(ls -d ${CHECKPOINT_DIR}/step_* | sort -V)
for step_dir in $step_dirs; do
    new_dir=$(dirname $step_dir)/hf_$(basename $step_dir)
    uv run python examples/convert_dcp_to_hf.py --config ${step_dir}/config.yaml \
        --dcp-ckpt-path ${step_dir}/policy/weights --hf-ckpt-path ${new_dir}
done