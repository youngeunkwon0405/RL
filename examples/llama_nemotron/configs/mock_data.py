from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from datasets import Dataset  # HF dataset wrapper for list based dataset
from nemo_rl.data import DataConfig  # TypedDict holding dataset related config

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.data.datasets import AllTaskProcessedDataset

def _append_message(tokenizer: AutoTokenizer, msg_log: LLMMessageLogType, role: str, text: str, gen_prompt: bool):
    """Render a single chat message and append token_ids in-place."""
    message = {"role": role, "content": text}
    rendered = tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=gen_prompt, add_special_tokens=False
    )
    message["token_ids"] = tokenizer(rendered, return_tensors="pt")["input_ids"][0]
    message["content"] = rendered
    msg_log.append(message)


def _base_processor(
    prompt_text: str,
    extra_info: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    idx: int,
    task_name: str,
    row_system_prompt: str | None = None,
) -> DatumSpec:
    """Common logic shared by all processors."""
    message_log: LLMMessageLogType = []

    # optional system prompt – row-level overrides spec-level
    sys_prompt = row_system_prompt if row_system_prompt is not None else task_data_spec.system_prompt
    if sys_prompt:
        _append_message(tokenizer, message_log, "system", sys_prompt, gen_prompt=False)

    # user message
    user_prompt = task_data_spec.prompt.format(prompt_text) if task_data_spec.prompt else prompt_text
    _append_message(tokenizer, message_log, "user", user_prompt, gen_prompt=True)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if max_seq_length and length > max_seq_length:
        # truncate aggressively and mask out of loss
        for m in message_log:
            m["token_ids"] = m["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_name,
    }


# -----------------------------------------------------------------------------
# Processors per task
# -----------------------------------------------------------------------------


def llm_judge_scp_116k_processor(
    datum_dict: Dict[str, Any], task_data_spec: TaskDataSpec, tokenizer, max_seq_length: int, idx: int
) -> DatumSpec:
    prompt_text = datum_dict.get("prompt", "")
    args_dict = datum_dict.get("args", {})
    extra_info = {
        "ground_truth": args_dict.get("expected_answer"),
        "prompt": args_dict.get("prompt"),
        "extract_box": args_dict.get("extract_box", False),
    }
    return _base_processor(
        prompt_text,
        extra_info,
        task_data_spec,
        tokenizer,
        max_seq_length,
        idx,
        datum_dict.get("task_name", "llm_judge_scp116k"),
        datum_dict.get("system_prompt"),
    )

def build_task_processors(prompt_file: str, system_prompt_file: str):
    """Return a mapping from task-name -> (TaskDataSpec, processor_fn)."""
    # TODO: ykarnati - we need different specs for each task
    base_spec = TaskDataSpec(
        task_name="llm_judge_scp116k",
        prompt_file=prompt_file,
        system_prompt_file=system_prompt_file,
    )

    return {
        "llm_judge_scp116k": (base_spec, llm_judge_scp_116k_processor),
    }

def get_mock_llm_judge_data(tokenizer: AutoTokenizer, data_config: DataConfig, task_processors):
    print("\n▶ Loading datasets...")

    # ---------------------------------------------------------------------
    # Build a small synthetic raw dataset based on the provided sample.
    # The AllTaskProcessedDataset expects a HF `Dataset` (or any index-able
    # collection of dicts).  We replicate the same example multiple times so
    # that the dataloader has enough items to iterate over.
    # ---------------------------------------------------------------------

    mock_example = {
        "prompt": (
            "Below is a math question. I want you to reason through the steps "
            "and then give a final answer. Your final answer should be in \\boxed{}."\
            "\nQuestion: The operation $\\otimes$ is defined for all nonzero numbers "
            "by $a \\otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \\otimes 2) "
            "\\otimes 3] - [1 \\otimes (2 \\otimes 3)]$."
        ),
        "args": {
            "expected_answer": "-\\frac{2}{3}",
            "prompt": (
                "The operation $\\otimes$ is defined for all nonzero numbers "
                "by $a \\otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \\otimes 2) "
                "\\otimes 3] - [1 \\otimes (2 \\otimes 3)]$."
            ),
            "extract_box": True,
        },
        "task_name": "llm_judge_scp116k",
        "system_prompt": "detailed thinking on",
    }

    # Number of times to replicate the mock example – configurable via the
    # DataConfig (falls back to 16 if not present).
    num_train_examples = data_config.get("num_mock_train_examples", 16)
    num_val_examples = data_config.get("num_mock_val_examples", 4)

    raw_train_ds = Dataset.from_list([mock_example] * num_train_examples)
    raw_val_ds = Dataset.from_list([mock_example] * num_val_examples)

    # Wrap with task-aware processing dataset so it can be consumed by the
    # dataloaders downstream.
    train_dataset = AllTaskProcessedDataset(
        raw_train_ds,
        tokenizer,
        task_processors["llm_judge_scp116k"][0],
        task_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        raw_val_ds,
        tokenizer,
        task_processors["llm_judge_scp116k"][0],
        task_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset