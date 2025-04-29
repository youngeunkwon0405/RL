
./build.sh; rm -rf venvs/ ; uv run examples/run_grpo_math.py --config=examples/configs/grpo_math_1B_megatron.yaml cluster.gpus_per_node=2 logger.wandb.name=tmp_testing_megatron_2gpus logger.wandb_enabled=true 2>&1 | tee mcore.log
