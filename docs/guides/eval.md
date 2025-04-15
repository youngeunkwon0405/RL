# Evaluation

This document explains how to use an evaluation script for assessing model capabilities.

## Start Evaluation

To run the evaluation, you can use the default configuration file or specify a custom one.

### Start Script

```sh
# To run the evaluation with default config (examples/configs/eval.yaml)
uv run python examples/run_eval.py

# Specify a custom config file
uv run python examples/run_eval.py --config path/to/custom_config.yaml

# Override specific config values via command line
uv run python examples/run_eval.py generation.model_name="Qwen/Qwen2.5-Math-7B-Instruct"
```

### Example Output

```
============================================================
model_name='Qwen2.5-Math-1.5B-Instruct' dataset_name='aime_2024'
score=0.10 (3.0/30)
============================================================
```

## Configuration File

You can find an example evaluation configuration file [here](../../examples/configs/eval.yaml).

### Prompt Template Configuration

Always remember to use the same `prompt_file` and `system_prompt_file` that were used during training.

For open-source models, we recommend setting `prompt_file=null` and `system_prompt_file=null` to allow them to use their native chat templates.
