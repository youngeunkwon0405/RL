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

import argparse

import yaml

from nemo_rl.models.megatron.community_import import export_model_from_megatron


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file in the checkpoint directory",
    )
    parser.add_argument(
        "--megatron-ckpt-path", type=str, default=None, help="Path to Megatron checkpoint"
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    # Parse known args for the script
    args = parser.parse_args()

    return args

def main():
    """Main entry point."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["policy"]["model_name"]
    # TODO: After the following PR gets merged:
    # https://github.com/NVIDIA/NeMo-RL/pull/148/files
    # tokenizer should be copied from policy/tokenizer/* instead of relying on the model name
    # We can expose a arg at the top level --tokenizer_path to plumb that through.
    # This is more stable than relying on the current NeMo-RL get_tokenizer() which can
    # change release to release.
    #tokenizer_name_or_path = config["policy"]["model_name"]

    export_model_from_megatron(
        hf_model_name=model_name,
        input_path=args.megatron_ckpt_path,
        output_path=args.hf_ckpt_path,
        hf_tokenizer_path = config["policy"]["tokenizer"]["name"],
    )


if __name__ == "__main__":
    main()
