from collections import defaultdict
from typing import Dict, Any, Optional, List, TypedDict, Union
import random
from collections import deque

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# ===============================================================================
#                             Replay Buffer Dataset
# ===============================================================================

class ReplayBufferItem(TypedDict):
    """Structure for items stored in the ReplayBuffer."""
    question: str
    answer: str
    reward: float
    critique: Optional[str]
    verdict: Optional[float]
    task_name: str = "critic"
    # Store original messages if needed for complex reconstruction, or other metadata
    # For example, if the original prompt formatting was complex.
    # original_datum_dict: Optional[Dict[str, Any]] = None 


def convert_actor_rollouts_to_buffer_items(rollout_batch: "BatchedDataDict[DatumSpec]") -> List[ReplayBufferItem]:
    """
    Converts a batch of rollouts into a list of ReplayBufferItems.

    Args:
        rollout_batch: A BatchedDataDict containing DatumSpec items from run_multi_turn_rollout.
                       This batch is expected to have 'message_log', 'extra_env_info', 
                       and 'total_reward' keys.
    """
    buffer_items: List[ReplayBufferItem] = []
    
    if not rollout_batch or not rollout_batch.size:
        return

    if "message_log" not in rollout_batch or \
       "extra_env_info" not in rollout_batch or \
       "total_reward" not in rollout_batch:
        # Consider logging a warning or raising an error if essential keys are missing
        # For example: print("Warning: rollout_batch is missing essential keys.")
        return

    batch_size = rollout_batch.size

    for i in range(batch_size):
        try:
            message_log = rollout_batch["message_log"][i]
            extra_info = rollout_batch["extra_env_info"][i]
            # Ensure total_reward exists for the item and is a tensor
            reward_tensor = rollout_batch["total_reward"][i]
            if not hasattr(reward_tensor, "item"): # Basic check if it's tensor-like
                # print(f"Warning: reward for item {i} is not a tensor. Skipping.")
                continue
            reward_value = reward_tensor.item()
        except (IndexError, KeyError) as e:
            # print(f"Warning: Data for item {i} is incomplete (error: {e}). Skipping.")
            continue

        question = extra_info.get("question")
        if question is None:
            # print(f"Warning: 'question' not found in extra_env_info for item {i}. Skipping.")
            continue

        answer = None
        # The last assistant message is considered the answer for the current interaction
        for message_item in reversed(message_log):
            if message_item.get("role") == "assistant":
                answer_content = message_item.get("content")
                if answer_content is not None:
                    answer = answer_content
                    break
        
        if answer is None:
            # print(f"Warning: No 'assistant' message with content found for item {i}. Skipping.")
            continue

        replay_item = ReplayBufferItem(
            question=str(question),
            answer=str(answer),
            reward=float(reward_value),
            critique=None,
            verdict=None,
        )
        buffer_items.append(replay_item)

    return buffer_items


class ReplayBuffer(Dataset):
    """FIFO dataset for storing and sampling from the replay buffer.
    
    Stores ReplayBufferItem TypedDicts.
    """

    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)
        # Removed tokenizer, task_data_spec, max_seq_length

    def push(self, item: ReplayBufferItem):
        """Adds an item to the buffer. If the buffer is full, the oldest item is discarded."""
        self.buffer.append(item)

    def push_batch(self, items: List[ReplayBufferItem]):
        """Adds a batch of items to the buffer. If the buffer exceeds its max size,
        older items will be discarded from the left (FIFO)."""
        self.buffer.extend(items)

    def sample(self, batch_size: int) -> List[ReplayBufferItem]:
        """Samples a batch of items from the buffer.
        
        Args:
            batch_size: The number of items to sample.
            
        Returns:
            A list of ReplayBufferItem, or fewer if buffer is smaller than batch_size.
        """
        if not self.buffer:
            return []
        
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), actual_batch_size)

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""
        return len(self.buffer)

    def __getitem__(self, idx: int) -> ReplayBufferItem: # Return type changed to ReplayBufferItem
        """Allows direct indexing into the buffer, returning a raw ReplayBufferItem."""
        if idx < 0 or idx >= len(self.buffer):
            raise IndexError("Index out of bounds for ReplayBuffer")
        return self.buffer[idx] # Returns the raw item

    def can_sample(self, batch_size: int) -> bool:
        """Checks if the buffer contains enough items to sample a batch of the given size."""
        return len(self.buffer) >= batch_size

# ===============================================================================
#                             Math Data Processor
# ===============================================================================

def prompt_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Math Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {
        "question": problem,    # raw question required for assembling quack triplet
        "ground_truth": user_message[1]["content"],
    }

    message_log: LLMMessageLogType = []
    user_message = {
        "role": "user",
        "content": task_data_spec.prompt.format(problem),
    }
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output

# ===============================================================================
#                             Critique Data Processor
# ===============================================================================

def critique_data_processor(
    datum_dict: ReplayBufferItem,
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a ReplayBufferItem into a DatumSpec for the Critique Environment."""
    question = datum_dict["question"]
    answer = datum_dict["answer"]
    reward = datum_dict["reward"]

    extra_env_info = {"reward": reward}     # unused for now, later we can use it as privileged information

    message_log: LLMMessageLogType = []
    user_message = {
        "role": "user",
        "content": task_data_spec.prompt.format(question, answer),
    }
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": "critic",
    }
    return output
    
# ===============================================================================

TASK_TO_DATA_PROCESSOR = {
    "math": prompt_data_processor,
    "critic": critique_data_processor,
    #TODO: Add fit data processor
}

def setup_data(data: Union[Dataset, Any], tokenizer: AutoTokenizer, data_config: DataConfig, task_name: str):
    print(f"\n▶ Setting up {task_name} data...")
    task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )
    data_processor = TASK_TO_DATA_PROCESSOR[task_name]
    # task_data_processors = defaultdict(
    #     lambda: (task_spec, data_processor)
    # )
    # task_data_processors[task_name] = (task_spec, data_processor)

    dataset = AllTaskProcessedDataset(
        data,
        tokenizer,
        task_spec,
        data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset