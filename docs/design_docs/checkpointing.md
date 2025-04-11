# Checkpointing with Hugging Face Models

## Checkpoint Format

NeMo RL provides two checkpoint formats for Hugging Face models: Torch distributed and Hugging Face format. Torch distributed is used by default for efficiency, while Hugging Face format is provided for compatibility with Hugging Face's `AutoModel.from_pretrained` API. Note that Hugging Face format checkpoints save only the model weights, excluding the optimizer states. It is recommended to use Torch distributed format to save intermediate checkpoints and to save a Hugging Face checkpoint only at the end of training.

## Generate a NeMo RL Checkpoint in Hugging Face Format

There are two ways to get a NeMo RL checkpoint in Hugging Face format.

1. (Recommended) Save the Hugging Face checkpoint directly by passing `save_hf=True` to `HFPolicy`'s `save_checkpoint`:
    
    ```python
    policy.save_checkpoint(
        weights_path=<WHERE_TO_SAVE_MODEL_WEIGHTS>,
        optimizer_path=<WHERE_TO_SAVE_OPTIM_STATE>,
        save_torch_dist=True,
        save_hf=True,
    )
    ```
2. Convert a Torch-distributed checkpoint to Hugging Face format after training. We provide a conversion script for this purpose.

    ```python
    uv run examples/convert_dcp_to_hf.py --config=<YAML CONFIG USED DURING TRAINING> <ANY CONFIG OVERRIDES USED DURING TRAINING> --dcp-ckpt-path=<PATH TO DIST CHECKPOINT TO CONVERT> --hf-ckpt-path=<WHERE TO SAVE HF CHECKPOINT>
    ```