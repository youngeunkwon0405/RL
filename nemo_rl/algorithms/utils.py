# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import warnings
from collections import defaultdict
from functools import wraps
from typing import Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer

from nemo_rl.data import hf_datasets
from nemo_rl.models.policy import TokenizerConfig


def calculate_kl_penalty_joschu2020(
    logprobs_policy: torch.Tensor, logprobs_reference: torch.Tensor
):
    """Calculates a per-token estimate of the KL Divergence between two log_probs.

    From Schulman 2020, always positive.

    logprobs_policy:    torch.Tensor (b, s)
    logprobs_reference: torch.Tensor (b, s)
    """
    r = logprobs_reference - logprobs_policy
    return torch.exp(r) - r - 1


def calculate_baseline_and_std_per_prompt(
    prompts: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    leave_one_out_baseline: bool = True,
    message_logs: Optional[List] = None,
):
    """Function to compute a baseline for each (prompt, response) pair in the batch.

    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask'
    are not included in the baseline calculation.

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute an unbiased baseline by leaving out the sample that
                                  the baseline is for (from RLOO https://arxiv.org/abs/2402.14740)
    message_logs: Optional[List]  List of message logs containing the conversations

    Returns:
    tensor (b,) of baselines on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = torch.device("cpu")

    metrics = defaultdict(list)

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[
            is_matching_prompt
        ]

        if leave_one_out_baseline:
            baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(
                reward_device
            )

        if valid_mask[prompt_idx].sum() <= 1:
            # Ignore sample: there are no valid responses, so set baseline equal to reward
            # to ignore it in the loss computation
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            rewards_reshaped = rewards[prompt_idx] * valid_mask[prompt_idx]

            metrics["average_reward_per_prompt"].append(rewards_reshaped.mean())
            metrics["average_pass_at_k_per_prompt"].append((rewards_reshaped > 0).any())

                

            num_valid = valid_mask[prompt_idx].float().sum() - int(
                leave_one_out_baseline
            )
            prompt_baseline = (
                torch.matmul(
                    baseline_mask_matrix, rewards[prompt_idx] * valid_mask[prompt_idx]
                )
                / num_valid
            )
            prompt_baseline_square = (
                torch.matmul(
                    baseline_mask_matrix,
                    (rewards[prompt_idx] ** 2) * valid_mask[prompt_idx],
                )
                / num_valid
            )

            baseline[prompt_idx] = prompt_baseline
            sq_baseline[prompt_idx] = prompt_baseline_square

    std = (sq_baseline - baseline.square()).sqrt().nan_to_num(0)
    
    # Calculate answer-based majority@K if message logs are available
    if message_logs is not None:
        math_majority_at_k = calculate_math_majority_at_k(message_logs, prompts, rewards, valid_mask)
        metrics["math_majority_at_k"] = math_majority_at_k
    
    return (
        baseline,
        std,
        {
            k: torch.as_tensor(v, dtype=torch.float32).mean().item()
            for k, v in metrics.items()
        },
    )


def surpress_user_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            output = f(*args, **kwargs)
        return output

    return wrapper


def masked_mean(
    values,
    mask,
    dim: Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor | float] = None,
):
    """Computes the mean of a microbatch, using a global statistic as the normalization factor."""
    normalization_factor = (
        torch.sum(mask, dim=dim)
        if global_normalization_factor is None
        else global_normalization_factor
    )
    return torch.sum(values * mask, dim=dim) / (normalization_factor + 1e-8)


def set_seed(seed: int):
    """Sets the seed for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_config: TokenizerConfig) -> AutoTokenizer:
    """Get the tokenizer and set pad token to eos token if it is not already set.

    This function initializes a tokenizer from the Hugging Face transformers library
    and configures it with appropriate chat templates and padding tokens.

    Args:
        tokenizer_config: A dictionary containing tokenizer configuration.
            Required keys:
                - name: The name or path of the pretrained tokenizer
            Optional keys:
                - chat_template: The chat template to use. Can be:
                    - None: Uses a passthrough template that just returns message content
                    - "default": Uses the tokenizer's default template
                    - A custom jinja2 template string
                    If not specified, the tokenizer's default template will be used.

    Returns:
        AutoTokenizer: The configured tokenizer instance

    Examples:
        ```{doctest}
        >>> from transformers import AutoTokenizer
        >>> from nemo_rl.algorithms.utils import get_tokenizer
        >>> # not specifying a chat template uses the tokenizer's default
        >>> config = {"name": "meta-llama/Llama-3.2-1B-Instruct"}
        >>> tokenizer = get_tokenizer(config)
        No chat template provided, using tokenizer's default
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful AI assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").apply_chat_template(messages, tokenize=False)

        >>> # Using a passthrough template
        >>> config = {
        ...     "name": "meta-llama/Llama-3.2-1B-Instruct",
        ...     "chat_template": None
        ... }
        >>> tokenizer = get_tokenizer(config)
        Using passthrough chat template
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == "".join(msg["content"] for msg in messages)

        >>> # Using a custom template
        >>> config = {
        ...     "name": "meta-llama/Llama-3.2-1B-Instruct",
        ...     "chat_template": "{% for message in messages %}{{ ' START: ' + message['content'] + ' END.' }}{% endfor %}"
        ... }
        >>> tokenizer = get_tokenizer(config)
        Using custom chat template
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == " START: You are a helpful AI assistant. END. START: Hello! END."
        ```
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "chat_template" in tokenizer_config:
        if tokenizer_config["chat_template"] is None:
            print("Using passthrough chat template")
            tokenizer.chat_template = (
                hf_datasets.COMMON_CHAT_TEMPLATES.passthrough_prompt_response
            )
        elif tokenizer_config["chat_template"].lower() == "default":
            print("Using tokenizer's default chat template")
        else:
            print("Using custom chat template")
            tokenizer.chat_template = tokenizer_config["chat_template"]
    else:
        print("No chat template provided, using tokenizer's default")

    return tokenizer


def calculate_math_majority_at_k(message_logs, prompts, rewards, valid_mask):
    """Calculate majority@K for math problems based on extracted answers.
    
    For each unique prompt, extract answers from responses, count votes for each answer,
    find the most voted answer, and check if it's correct.
    
    Args:
        message_logs: List of message logs containing the conversations
        prompts: tensor (b, s) Tensor of prompts 
        rewards: tensor (b,) Float-valued rewards
        valid_mask: tensor (b,) Vector of 0/1, where 0 is to ignore and 1 is to keep
        
    Returns:
        float: Average majority@K score across all prompts
    """
    import re
    from collections import Counter, defaultdict
    
    unique_prompts = torch.unique(prompts, dim=0)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = torch.device("cpu")
    
    total_score = 0.0
    num_prompts = 0
    
    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]
        
        if valid_mask[prompt_idx].sum() <= 1:
            continue  # Skip if not enough valid samples
            
        # Extract answers for this prompt
        answers = []
        prompt_rewards = []
        
        for idx in prompt_idx:
            if valid_mask[idx] == 0:
                continue
                
            # Extract answer from message log
            message_log = message_logs[idx.item()]
            extracted_answer = ""
            
            # Find last assistant response and extract answer
            for message in reversed(message_log):
                if message["role"] == "assistant":
                    response = message["content"]
                    # Try to extract from \boxed{}
                    boxed_match = re.search(r'\\boxed\{([^}]*)\}', response)
                    if boxed_match:
                        extracted_answer = boxed_match.group(1).strip()
                    break
            
            answers.append(extracted_answer)
            prompt_rewards.append(rewards[idx].item())
        
        total_score += _calculate_single_majority_at_k(answers, prompt_rewards)
        num_prompts += 1
    
    return total_score / num_prompts if num_prompts > 0 else 0.0


def _calculate_single_majority_at_k(answers, rewards):
    """Calculate majority@K score for a single prompt's responses.
    
    Args:
        answers: List of extracted answers
        rewards: List of corresponding rewards (0/1 for incorrect/correct)
        
    Returns:
        float: Majority@K score (0.0 to 1.0)
    """
    from collections import Counter
    
    if len(answers) == 0:
        return 0.0
    
    # Count votes for each answer
    answer_counts = Counter(answers)
    
    # Find the most frequent answer(s)
    max_count = max(answer_counts.values())
    most_frequent_answers = [answer for answer, count in answer_counts.items() if count == max_count]
    
    # Check if any of the most frequent answers are correct
    correct_most_frequent = False
    for answer in most_frequent_answers:
        for extracted_answer, reward in zip(answers, rewards):
            if extracted_answer == answer and reward > 0:
                correct_most_frequent = True
                break
        if correct_most_frequent:
            break
    
    if correct_most_frequent:
        # Give partial credit if there are tied answers
        return 1.0 / len(most_frequent_answers)
    else:
        return 0.0


def calculate_majority_at_k_advantages(message_logs, prompts, rewards, valid_mask, variance_reduction=False):
    """Calculate majority@K advantages for each response.
    
    For each response y, the advantage is:
    maj@k(all responses) - maj@k(all responses except y)
    
    Args:
        message_logs: List of message logs containing the conversations
        prompts: tensor (b, s) Tensor of prompts 
        rewards: tensor (b,) Float-valued rewards
        valid_mask: tensor (b,) Vector of 0/1, where 0 is to ignore and 1 is to keep
        variance_reduction: bool Whether to subtract the mean advantage per prompt (variance reduction)
        
    Returns:
        torch.Tensor: Advantages for each response (b,)
    """
    import re
    from collections import Counter
    
    unique_prompts = torch.unique(prompts, dim=0)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = torch.device("cpu")
    
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    
    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]
        
        if valid_mask[prompt_idx].sum() <= 1:
            # Not enough samples for majority@K, set advantages to 0
            advantages[prompt_idx] = 0.0
            continue
            
        # Extract answers and rewards for this prompt
        answers = []
        prompt_rewards = []
        valid_indices = []
        
        for idx in prompt_idx:
            if valid_mask[idx] == 0:
                continue
                
            # Extract answer from message log
            message_log = message_logs[idx.item()]
            extracted_answer = ""
            
            # Find last assistant response and extract answer
            for message in reversed(message_log):
                if message["role"] == "assistant":
                    response = message["content"]
                    # Try to extract from \boxed{}
                    boxed_match = re.search(r'\\boxed\{([^}]*)\}', response)
                    if boxed_match:
                        extracted_answer = boxed_match.group(1).strip()
                    break
            
            answers.append(extracted_answer)
            prompt_rewards.append(rewards[idx].item())
            valid_indices.append(idx)
        
        if len(answers) == 0:
            continue
            
        # Calculate majority@K with all responses
        full_majority_score = _calculate_single_majority_at_k(answers, prompt_rewards)
        
        # Calculate advantage for each response
        for j, (answer, reward, idx) in enumerate(zip(answers, prompt_rewards, valid_indices)):
            # Create subset without current response
            subset_answers = answers[:j] + answers[j+1:]
            subset_rewards = prompt_rewards[:j] + prompt_rewards[j+1:]
            
            if len(subset_answers) == 0:
                # If removing this response leaves no responses, advantage is the full score
                advantage = full_majority_score
            else:
                # Calculate majority@K without current response
                subset_majority_score = _calculate_single_majority_at_k(subset_answers, subset_rewards)
                advantage = full_majority_score - subset_majority_score
            
            advantages[idx] = advantage
    
    if variance_reduction:
        for i in range(len(unique_prompts)):
            is_matching_prompt = (prompts == unique_prompts[i]).all(1)
            prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]
            if valid_mask[prompt_idx].sum() <= 1:
                continue
            
            advantages[prompt_idx] = advantages[prompt_idx] - advantages[prompt_idx].mean()
        
    return advantages
