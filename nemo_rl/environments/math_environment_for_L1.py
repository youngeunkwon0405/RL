from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.environments.math_environment import MathEnvConfig, MathEnvironmentMetadata, HFVerifyWorker


# this is adapted from the MathEnvironment class in nemo_rl/environments/math_environment.py
# it is used to compute the L1 loss on top of the math reward

@ray.remote
class MathEnvironmentForL1(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: MathEnvConfig, tokenizer):

        self.cfg = cfg
        print (f"Initializing MathEnvironment with cfg={cfg}")
        self.num_workers = cfg["num_workers"]
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": HFVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]
        self.tokenizer = tokenizer
        
    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[MathEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        
        #print (f"message_log_batch={message_log_batch}")
        #print (f"metadata={metadata}")
        
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        # incorporate the L1 length reward into the reward
        accuracy_rewards = torch.tensor(results).cpu()
        L1_penalties = torch.zeros(len(results)).cpu()
        for i in range(len(results)):
            #import pdb; p=pdb.Pdb(); p.prompt="debug math environemnt L1"; p.set_trace()
            baseline_answer_length = metadata[i]["L1_metadata"]["baseline_answer_length"]
            penalty_factor = metadata[i]["L1_metadata"]["penalty_factor"]
            lower_bound_factor = metadata[i]["L1_metadata"]["lower_bound_factor"]
            upper_bound_factor = metadata[i]["L1_metadata"]["upper_bound_factor"]
            actual_answer_length = 0
            observations[i]["content"] += f"\n\nbaseline_answer_length: {baseline_answer_length}"
            observations[i]["content"] += f"\n\npenalty_factor: {penalty_factor}"
            observations[i]["content"] += f"\n\nlower_bound_factor: {lower_bound_factor}"
            observations[i]["content"] += f"\n\nupper_bound_factor: {upper_bound_factor}"
            for message in message_log_batch[i]:
                if message["role"] == "assistant":
                    actual_answer_length += len(self.tokenizer.encode(message["content"]))
            observations[i]["content"] += f"\n\nactual_answer_length: {actual_answer_length}"
            lower_bound = baseline_answer_length * lower_bound_factor
            upper_bound = baseline_answer_length * upper_bound_factor
            if actual_answer_length >= upper_bound:
                L1_penalties[i] = 1.0 * penalty_factor
            elif actual_answer_length <= lower_bound:
                L1_penalties[i] = 0.0 * penalty_factor
            else:
                L1_penalties[i] = (actual_answer_length - lower_bound) / (upper_bound - lower_bound) * penalty_factor

        # create a tensor of rewards and done flags
        rewards = accuracy_rewards - L1_penalties
        #rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        for i in range(len(observations)):
            observations[i]["content"] += f"\n\nl1_penalty: {L1_penalties[i].item()}"
            observations[i]["content"] += f"\n\naccuracy_reward: {accuracy_rewards[i].item()}"
            observations[i]["content"] += f"\n\nreward: {rewards[i].item()}"
            #observations[i]["L1_penalty"] = L1_penalties[i].item()
            #observations[i]["accuracy_reward"] = accuracy_rewards[i].item()
            #observations[i]["reward"] = rewards[i].item()
            
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """

        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
