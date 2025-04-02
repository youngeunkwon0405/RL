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
from functools import partial
from typing import Iterable

from megatron.core.models.gpt import GPTModel

from nemo.tron.llm.gpt import get_batch
from nemo.tron.losses import masked_next_token_loss
from nemo.tron.state import GlobalState

from nemo_reinforcer.algorithms.loss_functions import LossFunction


def forward_step_arbitrary_loss(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, loss_fn: LossFunction
):
    """Forward training step.

    Args:
        state (GlobalState): Global state for the run
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        tokens, labels, attention_mask, position_ids, loss_data = get_batch(
            data_iterator, state.cfg
        )
    timers("batch-generator").stop()

    with straggler_timer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_fn, data=loss_data)
