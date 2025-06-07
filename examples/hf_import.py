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

# uv run examples/hf_import.py --hf-model-name=/lustre/fsw/portfolios/coreai/users/yifuw/hf_checkpoints/dsv3/DeepSeek-V3-BF16 --output-path=/opt/checkpoints/tron/tmp/model__lustre_fsw_portfolios_coreai_users_yifuw_hf_checkpoints_dsv3_DeepSeek-V3-BF16

import argparse

from nemo_rl.models.megatron.community_import import import_model_from_hf_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Import HF model to megatron checkpoint"
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        default=None,
        help="HF model name or path to HF checkpoint",
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="Path to save megatron checkpoint"
    )
    args = parser.parse_args()

    return args


def main():
    """Main entry point."""
    args = parse_args()
    import_model_from_hf_name(args.hf_model_name, args.output_path)
    print(f"Imported HF model {args.hf_model_name} to {args.output_path}")


if __name__ == "__main__":
    main()
