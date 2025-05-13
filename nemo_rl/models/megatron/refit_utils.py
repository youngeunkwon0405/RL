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
from nemo.collections.llm.gpt.model.base import GPTConfig

import nemo_rl.models.megatron.converters as model_converters


def get_param_conversion_recipe_dict(
    name, converter_type: model_converters.ModelType, model_cfg: GPTConfig
):
    converter_dict = model_converters.REGISTRY[converter_type]

    local_layer = model_converters.get_local_layer_num(name)
    global_layer = (
        model_converters.get_global_layer_num(name, model_cfg)
        if local_layer is not None
        else None
    )
    format_dict = model_converters.SafeDict(l=local_layer, gl=global_layer)

    formatted_mapping = {
        k.format_map(format_dict): rec for k, rec in converter_dict.items()
    }
    return formatted_mapping, format_dict


@torch.no_grad()
def get_global_param_key_to_local_key_map(
    model, model_cfg: GPTConfig, keys: List[Tuple[str, str]]
) -> Dict[str, Tuple[int, str]]:
    """Get a mapping from global parameter keys to local parameter keys.

    Args:
        model: The model to get the mapping for.
        model_cfg: The model configuration.
        keys: The keys to get the mapping for. Tuple of (local_key, global_hf_key)

    Returns:
        A dictionary mapping global parameter keys to a tuple of (rank, local parameter key).
    """
    # Initialize pipeline parallel group information.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_global_rank_ids = model_converters.get_all_rank_ids_in_group(pp_group)

    # Build a mapping on each PP rank from a computed global key to the raw state dict key.
    # The global key is computed by replacing the local layer number (after "layers.")
    # with its corresponding global layer number (if applicable).
    local_map = {}
    for local_key, _ in keys:
        if local_key not in model.state_dict():
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
        local_map[global_key] = local_key

    # Gather the local maps from all PP ranks (only lightweight key info is gathered).
    all_maps = [None] * pp_world_size
    torch.distributed.all_gather_object(all_maps, local_map, group=pp_group)

    # Build the union over global keys and assign an owner (the rank with the smallest PP rank).
    union_global_map = {}
    for pp_rank, omap in enumerate(all_maps):
        for gk, raw_key in omap.items():
            if (
                gk not in union_global_map
                or pp_global_rank_ids[pp_rank] < union_global_map[gk][0]
            ):
                union_global_map[gk] = (pp_global_rank_ids[pp_rank], raw_key)
            else:
                print(
                    f"WARNING: {gk} already in union_global_map when gathering keys",
                    flush=True,
                )

    return union_global_map


def _modify_model_state_dict(model_state_dict, param_name_to_rank_and_key):
    # Process the model state dict to combine expert weights and biases
    # This combines tensors like "decoder.layers.0.mlp.experts.linear_fc1.weight0" through "weight15"
    # and potentially "decoder.layers.0.mlp.experts.linear_fc1.bias0" through "bias15"
    # into single tensors like "...linear_fc1.weight" and "...linear_fc1.bias".
    # It also modifies param_name_to_rank_and_key similarly.

    combined_params_data = {} # For model_state_dict
    combined_keys_rank_info = {} # For param_name_to_rank_and_key
    keys_to_remove_state_dict = []
    keys_to_remove_rank_map = []

    # Regex to match expert weights or biases with their index
    expert_pattern = re.compile(r'(decoder\.layers\.\d+\.mlp\.experts\.linear_fc[12])\.(weight|bias)(\d+)')

    # --- First pass: Identify expert params in model_state_dict, group them, mark originals for removal ---
    for key in model_state_dict:
        match = expert_pattern.match(key)
        if match:
            base_key = match.group(1)
            param_type = match.group(2)  # 'weight' or 'bias'
            expert_idx = int(match.group(3))

            # Create the combined key (e.g., "...linear_fc1.weight")
            combined_key = f"{base_key}.{param_type}"
            keys_to_remove_state_dict.append(key) # Mark original expert param for removal

            # Initialize the list for this combined param if first time seeing it
            if combined_key not in combined_params_data:
                combined_params_data[combined_key] = []

            # Ensure the list has enough slots (pad with None if needed)
            while len(combined_params_data[combined_key]) <= expert_idx:
                combined_params_data[combined_key].append(None)

            # Store the expert param tensor at the correct index
            combined_params_data[combined_key][expert_idx] = model_state_dict[key]

    # --- Second pass: Stack the gathered expert params and add to model_state_dict ---
    keys_added_state_dict = []
    for key, expert_list in combined_params_data.items():
        if isinstance(expert_list, list):
            # Check if all expert params were found
            if None in expert_list:
                missing_indices = [i for i, p in enumerate(expert_list) if p is None]
                print(f"WARNING: [State Dict] Missing expert params for {key} at indices: {missing_indices}", flush=True)
                continue # Skip this key

            # Stack the expert params along a new first dimension (num_experts, ...)
            try:
                stacked_param = torch.stack(expert_list, dim=0)
                model_state_dict[key] = stacked_param # Add the combined tensor
                keys_added_state_dict.append(key)
            except Exception as e:
                 print(f"[State Dict] Error stacking experts for key {key}: {e}", flush=True)
                 for i, p in enumerate(expert_list):
                      print(f"  Expert {i} shape: {p.shape if p is not None else 'None'}", flush=True)
                 continue # Skip this key

    # --- Third pass: remove the original individual expert keys from model_state_dict ---
    for key_to_remove in keys_to_remove_state_dict:
        if key_to_remove in model_state_dict:
            del model_state_dict[key_to_remove]

    # --- Fourth pass: Process param_name_to_rank_and_key ---
    for key, rank_and_local_key in param_name_to_rank_and_key.items():
        match = expert_pattern.match(key)
        if match:
            base_key = match.group(1)
            param_type = match.group(2)
            # We only need *one* representative rank/key for the combined entry
            combined_key = f"{base_key}.{param_type}"
            if combined_key not in combined_keys_rank_info:
                 # Store the rank and local key from the first expert instance encountered
                 # Note: Assuming all experts for a given layer/type have the same rank owner
                 # and the local key structure is similar (e.g., differing only by index).
                 # We need the owner_raw_key without the expert index for recipe lookup later.
                 local_key_base = expert_pattern.sub(r'\1.\2', rank_and_local_key[1])
                 combined_keys_rank_info[combined_key] = (rank_and_local_key[0], local_key_base)

            keys_to_remove_rank_map.append(key) # Mark original expert key for removal

    # --- Fifth pass: Add combined keys and remove original keys from param_name_to_rank_and_key ---
    keys_added_rank_map = 0
    for combined_key, rank_info in combined_keys_rank_info.items():
        if combined_key not in param_name_to_rank_and_key: # Avoid overwriting if somehow already exists
            param_name_to_rank_and_key[combined_key] = rank_info
            keys_added_rank_map += 1
        else:
             print(f"WARNING: [Rank Map] Combined key {combined_key} already exists. Skipping addition.", flush=True)


    for key_to_remove in keys_to_remove_rank_map:
        if key_to_remove in param_name_to_rank_and_key:
            del param_name_to_rank_and_key[key_to_remove]

    print(f"Combined {len(keys_added_state_dict)} expert parameter groups in state_dict.", flush=True)
    print(f"Removed {len(keys_to_remove_state_dict)} individual expert parameter keys from state_dict.", flush=True)
    print(f"Added {keys_added_rank_map} combined expert key entries to rank map.", flush=True)
    print(f"Removed {len(keys_to_remove_rank_map)} individual expert key entries from rank map.", flush=True)


@torch.no_grad()
def gather_and_convert_params(
    model,
    converter_type: model_converters.ModelType,
    model_cfg: GPTConfig,
    param_name_to_rank_and_key,
):
    # Process each parameter (by its unique global key) one at a time.
    gathered_params = {}
    model_state_dict = model.state_dict()

    # Modify the model state dict and rank map to combine expert weights/biases
    _modify_model_state_dict(model_state_dict, param_name_to_rank_and_key)

    for gk in sorted(param_name_to_rank_and_key.keys()):
        owner_pp_global_rank, owner_raw_key = param_name_to_rank_and_key[gk]

        # Only the owner PP rank has the parameter locally.
        if torch.distributed.get_rank() == owner_pp_global_rank:
            # Use the potentially modified owner_raw_key (now points to combined param)
            param = model_state_dict[owner_raw_key]

            # Use the conversion dict to get the appropriate recipe for this parameter.
            # Note: owner_raw_key might be the combined key now (e.g., ...linear_fc1.weight)
            # The get_param_conversion_recipe_dict needs to handle this.
            # We pass the *original* key structure (without expert index) if it was combined.
            # The rank map now stores this base key.
            recipe_dict, format_dict = get_param_conversion_recipe_dict(
                owner_raw_key, converter_type, model_cfg
            )
            # We need to look up the recipe using the pattern *without* the expert index,
            # as that's how recipes are defined. Let's try matching the potentially combined key.
            recipe_key_to_lookup = owner_raw_key
            # For combined keys (e.g., "...linear_fc1.weight"), the recipe should match this base form.
            recipe = recipe_dict.get(recipe_key_to_lookup, None)

            if recipe is None and "_extra_state" not in owner_raw_key:
                print(
                    f"WARNING: {owner_raw_key} has no recipe mapping for conversion",
                    flush=True,
                )
                hf_mapping = {"None": None}
            else:
                # If the parameter is TP-sharded, gather its slices on GPU.
                if recipe.get("tp", None) is not None:
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    tp_world_size = torch.distributed.get_world_size(tp_group)
                    gathered_slices = [
                        torch.empty_like(param) for _ in range(tp_world_size)
                    ]
                    torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                    full_param = torch.cat(gathered_slices, dim=recipe["tp"]).to(
                        torch.bfloat16
                    )
                else:
                    full_param = torch.clone(param).to(torch.bfloat16)

                # Convert the parameter using the provided function or mapping.
                if recipe.get("hf_func", None) is not None:
                    hf_mapping = recipe["hf_func"](full_param, model_cfg)
                    hf_mapping = {
                        k.format_map(format_dict): v for k, v in hf_mapping.items()
                    }
                elif recipe.get("hf", None) is not None:
                    hf_mapping = {recipe["hf"].format_map(format_dict): full_param}
                else:
                    raise NotImplementedError(
                        f"No conversion recipe found for {owner_raw_key}"
                    )
        else:
            hf_mapping = None  # Non-owner ranks will receive the converted tensors.

        # Broadcast the list of target HF parameter keys from the owner.
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if torch.distributed.get_rank() == owner_pp_global_rank:
            target_keys = [list(hf_mapping.keys())]
        else:
            target_keys = [None]  # Placeholder to be filled by broadcast.

        torch.distributed.broadcast_object_list(
            target_keys, src=owner_pp_global_rank, group=pp_group
        )
        if "None" in target_keys[0]:
            continue

        # For each converted tensor (could be more than one per original parameter), broadcast it individually.
        for target_key in target_keys[0]:
            if torch.distributed.get_rank() == owner_pp_global_rank:
                tensor_to_send = hf_mapping[target_key]
            else:
                tensor_to_send = None
            # Broadcast tensor metadata (shape and dtype) to allocate GPU buffer on receiving ranks.
            meta = [None]
            if torch.distributed.get_rank() == owner_pp_global_rank:
                meta[0] = (tensor_to_send.shape, str(tensor_to_send.dtype))
            torch.distributed.broadcast_object_list(
                meta, src=owner_pp_global_rank, group=pp_group
            )
            shape, dtype_str = meta[0]
            dtype = getattr(torch, dtype_str.split(".")[-1])
            if torch.distributed.get_rank() != owner_pp_global_rank:
                tensor_to_send = torch.empty(
                    *shape, dtype=dtype, device=torch.cuda.current_device()
                )
            torch.distributed.broadcast(
                tensor_to_send, src=owner_pp_global_rank, group=pp_group
            )
            gathered_params[target_key] = tensor_to_send

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return gathered_params
