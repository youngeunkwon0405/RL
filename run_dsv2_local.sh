source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
export HF_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/hf
uv run python examples/run_grpo_math.py --config=examples/configs/grpo_math_1B_megatron.yaml \
    grpo.val_batch_size=2 \
    policy.model_name=deepseek-ai/DeepSeek-V2-Chat \
    cluster.num_nodes=16 \
    cluster.gpus_per_node=8 \
    policy.megatron_cfg.pipeline_model_parallel_size=4 \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.expert_tensor_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=False \
    policy.megatron_cfg.expert_model_parallel_size=32 \
    policy.refit_buffer_size_gb=6 \
    policy.max_total_sequence_length=128 \
    checkpointing.enabled=False \
    checkpointing.save_period=5 \
    grpo.val_period=-1 \
    grpo.max_val_samples=16 \
    grpo.val_batch_size=4 \
    checkpointing.keep_top_k=100 \
    checkpointing.checkpoint_dir=results/dsv2-lite \
    grpo.val_at_start=False \
    grpo.max_val_samples=16 \
    policy.generation.vllm_cfg.async_engine=False \
    policy.generation.vllm_cfg.tensor_parallel_size=16 \
    policy.megatron_cfg.activation_checkpointing=True \
    grpo.max_num_steps=10 \
    grpo.num_prompts_per_step=16 \
    grpo.num_generations_per_prompt=16 \
    policy.train_global_batch_size=256 \
    logger.wandb_enabled=True \
    logger.wandb.project='grpo-dev-yifu' \
    logger.wandb.name='guyueh/megatron_refitopt'

#    policy.sequence_packing.enabled=False \
# ps aux | grep python | grep run_grpo_math.py | grep -v grep | awk '{print $2}' | xargs kill -9