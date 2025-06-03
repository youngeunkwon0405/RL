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


def get_tp_dim(model, param_name, named_modules_dict):
    # if param_name == "decoder.layers.3.mlp.shared_experts.linear_fc1.weight":
    #     if torch.distributed.get_rank() == 0:
    #         import pdb

    #         pdb.set_trace()
    #     torch.distributed.barrier()
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
    elif (
        isinstance(module, VocabParallelEmbedding)
        or isinstance(module, ColumnParallelLinear)
        # for MoE, parallel_mode isn't set for these
        or isinstance(
            module,
            TEColumnParallelGroupedLinear,
        )
        or isinstance(module, TEColumnParallelLinear)
    ):
        return 0
    elif (
        isinstance(module, RowParallelLinear)
        # for MoE, parallel_mode isn't set for these
        or isinstance(
            module,
            TERowParallelGroupedLinear,
        )
        or isinstance(module, TERowParallelLinear)
    ):
        return 1
    # TODO(yifu): moe layers
    else:
        return None


def get_global_rank_helper(
    ep_rank,
    pp_rank,
    ep_group, ## ep group that this rank belongs to
    pp_group, ## pp group that this rank belongs to
    all_ep_groups, ## global view of all groups
    all_pp_groups
):
    current_rank = torch.distributed.get_rank()

    this_rank_has_param = (ep_group[ep_rank] == pp_group[pp_rank])
    if this_rank_has_param:
        assert ep_group[ep_rank] == current_rank ## TODO: should this always be true? I think so
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
    global_ranks_that_have_param_after_ep_gather = set()
    for ep_id in ep_group:
        if ep_id == current_rank:
            continue
        for pp_ids in all_pp_groups:
            pp_intersection = set(pp_ids).intersection(global_ranks_that_have_param)
            ## this means that the current rank is in an ep group with a rank who
            ## will have the param after pp gather
            ## so the current rank will get the param after doing an ep gather
            ## following the pp gather
            if ep_id in pp_ids and len(pp_intersection) > 0:
                return ep_id, "ep"


@torch.no_grad()
def get_global_param_key_to_local_key_map(
    model, model_cfg: GPTConfig, keys: List[Tuple[str, str]]
) -> Dict[str, Tuple[int, str]]:
    """Get a mapping from global parameter keys to local parameter keys.

    Args:
        model: The model to get the mapping for.
        model_cfg: The model configuration.
        keys: The local_keys to get the mapping for.

    Returns:
        A dictionary mapping global parameter keys to a tuple of (rank, local parameter key).
    """
    # Initialize pipeline parallel group information.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_global_rank_ids = model_converters.get_all_rank_ids_in_group(pp_group)

    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_world_size = parallel_state.get_expert_model_parallel_world_size()
    ep_global_rank_ids = model_converters.get_all_rank_ids_in_group(ep_group)

    ## TODO: cache these?
    all_ep_rank_ids = [None] * torch.distributed.get_world_size()
    all_pp_rank_ids = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_ep_rank_ids, ep_global_rank_ids)
    torch.distributed.all_gather_object(all_pp_rank_ids, pp_global_rank_ids)

    # Build a mapping on each PP rank from a computed global key to the raw state dict key.
    # The global key is computed by replacing the local layer number (after "layers.")
    # with its corresponding global layer number (if applicable).
    local_map = {}
    state_dict = model.state_dict()
    for local_key in keys:
        if local_key not in state_dict:
            continue
        local_layer = model_converters.get_local_layer_num(local_key)
        if local_layer is not None:
            global_layer = model_converters.get_global_layer_num(local_key, model_cfg)
            # Replace the first occurrence of the digits after "layers." with the global layer number.
            global_key = re.sub(
                r"(?<=layers\.)\d+", str(global_layer), local_key, count=1
            )
        else:
            global_key = local_key
        local_expert = model_converters.get_local_expert_num(global_key)
        if local_expert is not None:
            global_expert = model_converters.get_global_expert_num(global_key, model_cfg)
            # Replace the last occurrence of the digits after "weight" with the global expert number.
            global_key = re.sub(r"(?<=weight)\d+", str(global_expert), global_key)
        local_map[global_key] = local_key

    # Gather the local maps from all PP ranks (only lightweight key info is gathered).
    all_maps_pp = [None] * pp_world_size
    torch.distributed.all_gather_object(all_maps_pp, local_map, group=pp_group)

    # then gather across EP Ranks
    all_maps = [None] * ep_world_size
    torch.distributed.all_gather_object(all_maps, all_maps_pp, group=ep_group)

    # Build the union over global keys and assign an owner (the rank with the smallest PP rank).
    union_global_map = {}
    for ep_rank, item in enumerate(all_maps):
        for pp_rank, omap in enumerate(item):
            for gk, raw_key in omap.items():
                #if (
                #    gk not in union_global_map
                #    or pp_global_rank_ids[pp_rank] < union_global_map[gk][0]
                #):
                if gk not in union_global_map:
                    global_rank, pp_or_ep_gather = get_global_rank_helper(
                        ep_rank,
                        pp_rank,
                        ep_global_rank_ids,
                        pp_global_rank_ids,
                        all_ep_rank_ids,
                        all_pp_rank_ids,
                    )

                    union_global_map[gk] = (global_rank, raw_key, pp_or_ep_gather)
                else:
                    print(
                        f"WARNING: {gk} already in union_global_map when gathering keys",
                        flush=True,
                    )
    return union_global_map


@torch.no_grad()
def gather_params(
    model,
    param_name_to_rank_and_key,
):
    import time
    st = time.time()
    
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()
    pp_global_rank_ids = torch.distributed.get_process_group_ranks(pp_group)
    ep_global_rank_ids = torch.distributed.get_process_group_ranks(ep_group)

    # Process each parameter (by its unique global key) one at a time.
    gathered_params = {}
    named_modules_dict = dict(model.named_modules())
    state_dict = model.state_dict()
    for gk in sorted(param_name_to_rank_and_key.keys()):
        owner_global_rank, owner_raw_key, ep_or_pp_gather = param_name_to_rank_and_key[gk]

        # Only the owner rank has the parameter locally.
        if torch.distributed.get_rank() == owner_global_rank:
            param = state_dict[owner_raw_key]

            # Check if param is TP-sharded
            tp_dim = get_tp_dim(model, owner_raw_key, named_modules_dict)

            # If the parameter is TP-sharded, gather its slices on GPU.
            if tp_dim is not None:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                tp_world_size = torch.distributed.get_world_size(tp_group)
                gathered_slices = [
                    torch.empty_like(param) for _ in range(tp_world_size)
                ]
                torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                full_param = torch.cat(gathered_slices, dim=tp_dim).to(torch.bfloat16)
            else:
                full_param = torch.clone(param).to(torch.bfloat16)

            # Use the original parameter key without conversion
            param_mapping = {gk: full_param}
        else:
            param_mapping = None  # Non-owner ranks will receive the tensors.

        # Broadcast the list of target parameter keys from the owner.
        if torch.distributed.get_rank() == owner_global_rank:
            target_keys = [list(param_mapping.keys())]
        else:
            target_keys = [None]  # Placeholder to be filled by broadcast.

        if ep_or_pp_gather == "pp":
            rank_to_broadcast = owner_global_rank
        else:
            ## we are getting the param from the ep gather, so just do a dummy broadcast here
            rank_to_broadcast = pp_global_rank_ids[0]
        torch.distributed.broadcast_object_list(
            target_keys, src=rank_to_broadcast, group=pp_group
        )

        ## now do the ep broadcast
        ## TODO: are there any cases where this will cause a hang?
        ## like times where we would try to broadcast from two different ranks
        ## in the same ep group?
        if ep_or_pp_gather == "ep":
            rank_to_broadcast = owner_global_rank
        else:
            ## the current rank already has the param from the pp gather
            rank_to_broadcast = torch.distributed.get_rank()
        torch.distributed.broadcast_object_list(
            target_keys, src=rank_to_broadcast, group=ep_group
        )

        ## TODO
        if "None" in target_keys[0]:
            continue

        # For each tensor, broadcast it individually.
        for target_key in target_keys[0]:
            if torch.distributed.get_rank() == owner_global_rank:
                tensor_to_send = param_mapping[target_key]
            else:
                tensor_to_send = None
            # Broadcast tensor metadata (shape and dtype) to allocate GPU buffer on receiving ranks.
            meta = [None]
            if torch.distributed.get_rank() == owner_global_rank:
                meta[0] = (tensor_to_send.shape, str(tensor_to_send.dtype))

            if ep_or_pp_gather == "pp":
                rank_to_broadcast = owner_global_rank
            else:
                ## we are getting the param from the ep gather, so just do a dummy broadcast here
                rank_to_broadcast = pp_global_rank_ids[0]

            torch.distributed.broadcast_object_list(
                meta, src=rank_to_broadcast, group=pp_group
            )

            ## now do the ep broadcast
            if ep_or_pp_gather == "ep":
                rank_to_broadcast = owner_global_rank
            else:
                ## the current rank already has the param from the pp gather
                rank_to_broadcast = torch.distributed.get_rank()

            torch.distributed.broadcast_object_list(
                meta, src=rank_to_broadcast, group=ep_group
            )

            shape, dtype_str = meta[0]
            dtype = getattr(torch, dtype_str.split(".")[-1])
            if torch.distributed.get_rank() != owner_global_rank:
                tensor_to_send = torch.empty(
                    *shape, dtype=dtype, device=torch.cuda.current_device()
                )
            if ep_or_pp_gather == "pp":
                rank_to_broadcast = owner_global_rank
            else:
                ## we are getting the param from the ep gather, so just do a dummy broadcast here
                rank_to_broadcast = pp_global_rank_ids[0]
            torch.distributed.broadcast(
                tensor_to_send, src=rank_to_broadcast, group=pp_group
            )

            ## now do the ep broadcast
            if ep_or_pp_gather == "ep":
                rank_to_broadcast = owner_global_rank
            else:
                ## the current rank already has the param from the pp gather
                rank_to_broadcast = torch.distributed.get_rank()

            torch.distributed.broadcast(
                tensor_to_send, src=rank_to_broadcast, group=ep_group
            )

            gathered_params[target_key] = tensor_to_send

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Time taken to gather params: {time.time() - st}")
    return gathered_params
