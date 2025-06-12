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
import time

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from nemo_rl.models.megatron.converters.common import get_global_key_from_local_key


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
        print(f"Module {key} not found in named_modules_dict")
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


def find_global_rank_that_has_param(
    ep_rank,
    pp_rank,
    ep_group,  ## ep group that this rank belongs to
    pp_group,  ## pp group that this rank belongs to
    all_ep_groups,  ## global view of all groups
    all_pp_groups,
):
    current_rank = torch.distributed.get_rank()

    this_rank_has_param = ep_group[ep_rank] == pp_group[pp_rank]
    if this_rank_has_param:
        assert (
            ep_group[ep_rank] == current_rank
        )  ## TODO: should this always be true? I think so
        return current_rank, "pp"

    ## keep track of the ranks that have the param,
    ## then the ranks that will have the param after the PP all-gather
    global_ranks_that_have_param = set()
    for ep_ids in all_ep_groups:
        for pp_ids in all_pp_groups:
            if ep_ids[ep_rank] == pp_ids[pp_rank]:
                global_ranks_that_have_param.add(ep_ids[ep_rank])

    pp_intersection = set(pp_group).intersection(global_ranks_that_have_param)
    if len(pp_intersection) > 0:
        ## this means the rank that has the param is in the pp group
        ## so we can get the param using pp communication
        for s in pp_intersection:
            return s, "pp"

    ep_intersection = set(ep_group).intersection(global_ranks_that_have_param)
    if len(ep_intersection) > 0:
        ## this means the rank that has the param is in the ep group
        ## so we can get the param using pp communication
        for s in ep_intersection:
            return s, "ep"

    ## final case is that we need to get the param by first doing a pp gather, then an ep gather
    for ep_id in ep_group:
        if ep_id == current_rank:
            continue
        for pp_ids in all_pp_groups:
            pp_intersection = set(pp_ids).intersection(global_ranks_that_have_param)
            ## this means that the current rank is in an ep group with a rank who
            ## will have the param after pp gather
            ## so the current rank will get the param after doing an ep gather
            ## following the pp gather
            if len(pp_intersection) > 0 and ep_id in pp_ids:
                return ep_id, "ep"


@torch.no_grad()
def gather_params(
    model,
    keys,
):
    st = time.time()

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = torch.distributed.get_world_size(tp_group)
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    ep_world_size = torch.distributed.get_world_size(ep_group)

    named_modules_dict = dict(model.named_modules())
    state_dict = model.state_dict()
    gathered_params = {}
    ep_pattern = re.compile(r"mlp\.experts.*\.weight\d*$")
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
            else:
                # TODO: why do we need to clone?
                full_param = param
            global_key = get_global_key_from_local_key(local_key, model.config)
        else:
            #  params that may not be on every rank, e.g. the embedding layer
            global_key = None
            full_param = torch.empty(
                *shape, dtype=dtype, device=torch.cuda.current_device()
            )

        # gather across PP group
        pp_gathered_global_keys = [None] * pp_world_size
        torch.distributed.all_gather_object(
            pp_gathered_global_keys, global_key, group=pp_group
        )
        # To test no gather:
        # pp_gathered_global_keys = [global_key] * pp_world_size

        pp_gathered_params = [
            torch.empty(*shape, dtype=dtype, device=torch.cuda.current_device())
            for _ in range(pp_world_size)
        ]
        torch.distributed.all_gather(pp_gathered_params, full_param, group=pp_group)

        # gather across EP group
        if ep_pattern.search(local_key):
            ep_gathered_global_keys = [None] * ep_world_size
            torch.distributed.all_gather_object(
                ep_gathered_global_keys, pp_gathered_global_keys, group=ep_group
            )
            # To test no gather:
            # ep_gathered_global_keys = [pp_gathered_global_keys] * ep_world_size

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

            flat_gathered_global_keys = [x for y in ep_gathered_global_keys for x in y]
            flat_gathered_params = [
                x for y in ep_gathered_params for x in torch.unbind(y)
            ]
        else:
            flat_gathered_global_keys = pp_gathered_global_keys
            flat_gathered_params = pp_gathered_params

        for k, p in zip(flat_gathered_global_keys, flat_gathered_params):
            if k is not None:
                gathered_params[k] = p

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Time taken to gather params: {time.time() - st}")
    return gathered_params
