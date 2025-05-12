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

import torch
from typing import Any, Dict, List

def combine_metrics(metric_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine a list of metrics dictionaries into a single dictionary.

    Args:
        metrics_list: List of dictionaries containing metrics

    Returns:
        Combined dictionary of metrics
    """
    combined_metrics = {}
    if metric_list:
        # Get all unique keys from the results
        all_keys = set().union(*[r.keys() for r in metric_list])
        for key in all_keys:
            # Concatenate values for each key
            values = [r[key] for r in metric_list if key in r]
            if values:
                if torch.is_tensor(values[0]):
                    combined_metrics[key] = torch.cat(values)  # Concatenate tensors
                elif isinstance(values[0], list):
                    combined_metrics[key] = [item for sublist in values for item in sublist]
                else:
                    combined_metrics[key] = values  # Keep as list for other types

    return combined_metrics
