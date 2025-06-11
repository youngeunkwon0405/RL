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
import re
from typing import Dict, List, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from nemo.collections.llm.gpt.model.base import GPTConfig
import nemo_rl.models.megatron.converters as model_converters
from nemo_rl.models.megatron.converters.common import get_global_key_from_local_key

def single_rank_print(*args, **kwargs):
    if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_pipeline_model_parallel_rank() == 0 and parallel_state.get_expert_model_parallel_rank() == 0:
        print(*args, **kwargs)

def get_tp_dim(model, param_name, named_modules_dict):
    # pass in named_modules_dict so we can get it ahead of time instead
    # of once for each param
    pattern = re.compile(r"\.(?:weight|bias)\d*$")
    if not pattern.search(param_name):
        return None

    prefix = ""
    if hasattr(model, "module"):
        prefix = "module."
        if hasattr(model.module, "module"):
            prefix = "module.module."
    key = prefix + ".".join(param_name.split(".")[:-1])
    module = named_modules_dict.get(key)
    if module is None:
        single_rank_print(f"Module {key} not found in named_modules_dict")
        return None
    if hasattr(module, "parallel_mode") and module.parallel_mode is not None:
        # TE layers sometimes have parallel_mode we can check directly
        if module.parallel_mode == "column":
            return 0
        elif module.parallel_mode == "row":
            return 1
        else:
            return None
    elif isinstance(
        module,
        (
            VocabParallelEmbedding,
            ColumnParallelLinear,
            TEColumnParallelGroupedLinear,
            TEColumnParallelLinear,
        ),
    ):
        return 0
    elif isinstance(
        module, (RowParallelLinear, TERowParallelGroupedLinear, TERowParallelLinear)
    ):
        return 1
    else:
        return None


@torch.no_grad()
def gather_params(
    model,
    keys,
    key_to_global_keys: Optional[Dict[str, List[str]]] = None,
):
    import time

    st = time.time()

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = torch.distributed.get_world_size(tp_group)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    ep_world_size = torch.distributed.get_world_size(ep_group)

    et = time.time()
    single_rank_print(f"get world size time: {et - st}")
    st = et

    named_modules_dict = dict(model.named_modules())
    state_dict = model.state_dict()
    gathered_params = {}
    ep_pattern = re.compile(r"mlp\.experts.*\.weight\d*$")

    et = time.time()
    single_rank_print(f"get state dict time: {et - st}")
    st = et

    for local_key, shape, dtype in sorted(keys):
        if local_key in state_dict:
            param = state_dict[local_key]

            # Check if param is TP-sharded
            tp_dim = get_tp_dim(model, local_key, named_modules_dict)

            # If the parameter is TP-sharded, gather its slices on GPU.
            if tp_dim is not None:
                gathered_slices = [
                    torch.empty_like(param) for _ in range(tp_world_size)
                ]
                torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                # TODO: why cast to torch.bfloat16 instead of param.dtype?
                full_param = torch.cat(gathered_slices, dim=tp_dim)
                
                et = time.time()
                single_rank_print(f"{local_key} {shape} gather tp time: {et - st}")
                st = et
            else:
                # TODO: why do we need to clone?
                full_param = param
            if key_to_global_keys is None:
                global_key = get_global_key_from_local_key(local_key, model.config)

            et = time.time()
            single_rank_print(f"{local_key} {shape} get global key time: {et - st}")
            st = et
        else:
            #  params that may not be on every rank, e.g. the embedding layer
            if key_to_global_keys is None:
                global_key = None
            full_param = torch.empty(*shape, dtype=dtype, device=torch.cuda.current_device())
        
            et = time.time()
            single_rank_print(f"{local_key} {shape} create full param time: {et - st}")
            st = et

        # gather across PP group
        if key_to_global_keys is None:
            pp_gathered_global_keys = [None] * pp_world_size
            torch.distributed.all_gather_object(
                pp_gathered_global_keys, global_key, group=pp_group
            )
            # To test no gather:
            # pp_gathered_global_keys = [global_key] * pp_world_size

            et = time.time()
            single_rank_print(f"{local_key} {shape} gather pp global key time: {et - st}")
            st = et

        pp_gathered_params = [
            torch.empty(*shape, dtype=dtype, device=torch.cuda.current_device())
            for _ in range(pp_world_size)
        ]
        torch.distributed.all_gather(pp_gathered_params, full_param, group=pp_group)

        et = time.time()
        single_rank_print(f"{local_key} {shape} gather pp params time: {et - st}")
        st = et

        # gather across EP group
        if ep_pattern.search(local_key):
            if key_to_global_keys is None:
                ep_gathered_global_keys = [None] * ep_world_size
                torch.distributed.all_gather_object(
                    ep_gathered_global_keys, pp_gathered_global_keys, group=ep_group
                )
                # To test no gather:
                # ep_gathered_global_keys = [pp_gathered_global_keys] * ep_world_size

                et = time.time()
                single_rank_print(f"{local_key} {shape} gather ep global keys time: {et - st}")
                st = et

            stacked_pp_gathered_params = torch.stack(pp_gathered_params)
            ep_gathered_params = [
                torch.empty(
                    stacked_pp_gathered_params.shape,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                )
                for _ in range(ep_world_size)
            ]
            torch.distributed.all_gather(
                ep_gathered_params, stacked_pp_gathered_params, group=ep_group
            )

            if key_to_global_keys is None:
                flat_gathered_global_keys = [x for y in ep_gathered_global_keys for x in y]
            flat_gathered_params = [x for y in ep_gathered_params for x in torch.unbind(y)]

            et = time.time()
            single_rank_print(f"{local_key} {shape} gather ep params time: {et - st}")
            st = et
        else:
            if key_to_global_keys is None:
                flat_gathered_global_keys = pp_gathered_global_keys
            flat_gathered_params = pp_gathered_params

        if key_to_global_keys is not None:
            flat_gathered_global_keys = key_to_global_keys[local_key]

        for k, p in zip(flat_gathered_global_keys, flat_gathered_params):
            if k is not None:
                gathered_params[k] = p

        et = time.time()
        single_rank_print(f"{local_key} {shape} zip time: {et - st}")
        st = et
        
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    single_rank_print(f"Time taken to gather params: {time.time() - st}")
    return gathered_params
