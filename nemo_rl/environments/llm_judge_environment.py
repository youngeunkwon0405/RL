import random
from typing import Dict, List, Optional, Tuple

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MockLLMJudgeEnvConfig(dict):
    pass


@ray.remote
class MockLLMJudgeEnvironment(EnvironmentInterface):
    """A mock environment that assigns a random reward of 0 or 1 until we have a judge model."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: Optional[MockLLMJudgeEnvConfig] = None):
        # Keep the cfg around for future extensions
        self.cfg = cfg or {}

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[dict]],
    ) -> EnvironmentReturn:
        batch_size = len(message_log_batch)

        # Produce dummy observations acknowledging receipt of the last message
        observations = [{"role": "environment", "content": "Environment: received"} for _ in range(batch_size)]

        # Random reward 0 or 1 for each sample
        rewards = torch.tensor([random.randint(0, 1) for _ in range(batch_size)])

        # One-step episodic task – terminate immediately
        terminateds = torch.ones_like(rewards)

        # No special stop-strings or metadata updates
        next_stop_strings = [None] * batch_size
        metadata = [None] * batch_size

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
        )

    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> Tuple[BatchedDataDict, dict]:
        """Provide minimal metrics required by the trainer."""

        metrics = {"accuracy": batch["rewards"].float().mean().item()}
        return batch, metrics
