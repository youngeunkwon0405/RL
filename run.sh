
./build.sh; rm -rf venvs/ ; uv run --with 'debugpy>=1.8.0' examples/run_grpo_math.py --config=examples/configs/grpo_math_1B_megatron.yaml cluster.gpus_per_node=2
