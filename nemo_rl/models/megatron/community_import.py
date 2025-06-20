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


def import_model_from_hf_name(hf_model_name: str, output_path: str):
    if "llama" in hf_model_name.lower():
        from nemo.tron.converter.llama import HFLlamaImporter

        print(f"Importing model {hf_model_name} to {output_path}...")
        importer = HFLlamaImporter(
            hf_model_name,
            output_path=output_path,
        )
        importer.apply()
    elif "qwen" in hf_model_name.lower():
        from nemo.tron.converter.qwen import HFQwen2Importer

        print(f"Importing model {hf_model_name} to {output_path}...")
        importer = HFQwen2Importer(
            hf_model_name,
            output_path=output_path,
        )
        importer.apply()
    elif "nemotron-h" in hf_model_name.lower():
        from nemo.tron.converter.ssm import HFNemotronHImporter

        print(f"Importing model {hf_model_name} to {output_path}...")
        importer = HFNemotronHImporter(
            hf_model_name,
            output_path=output_path,
        )
        # Cast to float32 since the HF checkpoint is homogeneous in dtype, but
        # the mcore state_dict is bfloat16 for most params, but float32 for
        # the ssm params (A_log, D). So the importing is done by casting both
        # to float32, which matches the nemo2 implementation.
        #importer.apply(dtype=torch.float32)
        importer.apply(dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown model: {hf_model_name}")
    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
