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
import abc
from typing import Dict, List, Tuple, NamedTuple, Optional

from torch import Tensor

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.data.interfaces import LLMMessageLogType


class EnvironmentReturn(NamedTuple):
    """Standard return type for environment step methods."""

    observations: List[Dict[str, str]]
    metadata: List[Optional[dict]]
    next_stop_strings: List[Optional[List[str]]]
    rewards: Tensor
    terminated: Tensor


class EnvironmentInterface(abc.ABC):
    @abc.abstractmethod
    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[Optional[dict]],
        *args,
        **kwargs,
    ) -> EnvironmentReturn:
        """Runs a step in the environment. Allows for asynchrony with remote servers, but it's not required (this function is a ray remote).

        message_log_batch: batch of OpenAI-API-like message logs that represent interactions with the LLM.
                  Each element is a List[Dict[str, Union[str, torch.Tensor]]].
                  For example, if this were a Math Environment, then the message log
                  would be
                  [
                    {"role": "user", "content": "problem"},
                    {"role": "assistant", "content": "response"},
                  ]
                  but if this were a code environment
                  with feedback, it would be:
                  [
                    {"role": "user", "content": "problem"},
                    {"role": "assistant", "content": "response"},
                    {"role": "user", "content": "code result"},
                    {"role": "assistant", "content": "model response"},
                  ]
        metadata:     batch of whatever the environment needs to keep track of. I.e.
                      math solutions, code unit tests, or agent states. Can be None if episode terminated.

        Returns:
        - EnvironmentReturn NamedTuple containing observations, metadata, next_stop_strings, rewards, and terminated flags.
        """

    @abc.abstractmethod
    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Post processing function after all rollouts are done for the batch and returns metrics."""
