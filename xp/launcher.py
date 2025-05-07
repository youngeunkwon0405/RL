#!/usr/bin/env python3
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
import itertools
import os
import subprocess
from typing import Any, Dict, List, Tuple, Optional

import yaml


# Environment variables
LOG_DIR = os.environ["LOG"] + "/nemo-rl"
CONTAINER = os.environ["CON"] + "/nemo_rl_base.sqsh"
MOUNTS = "/lustre:/lustre"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch Slurm experiments with parameter sweeps")
    parser.add_argument("script", type=str, help="Path to the Python script to run")
    parser.add_argument("--config", type=str, help="Path to the base YAML config file")
    parser.add_argument("--sweep", type=str, help="Path to the sweep config YAML file")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--time", type=str, default="4:0:0", help="Time limit for the job")
    parser.add_argument("--account", type=str, required=True, help="Slurm account to use")
    parser.add_argument("--partition", type=str, required=True, help="Slurm partition to use")
    parser.add_argument("--container", type=str, default=CONTAINER, help="Container to use")
    parser.add_argument("--mounts", type=str, default=MOUNTS, help="Mounts to use")
    parser.add_argument("--job-name", type=str, default=None, help="Base name for the job")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def verify_sweep_config(sweep_config: Dict[str, Any], script_path: str) -> None:
    """Verify that the sweep config is valid for the given script."""
    if "script_name" not in sweep_config:
        raise AssertionError(
            f"Sweep config must contain a 'script_name' parameter that matches the target script.\n"
            f"Expected script: {os.path.basename(script_path)}"
        )
    
    expected_script = sweep_config["script_name"]
    actual_script = os.path.basename(script_path)
    
    if expected_script != actual_script:
        raise AssertionError(
            f"Sweep config script_name '{expected_script}' does not match the target script '{actual_script}'.\n"
            f"Please update the sweep config to use the correct script name."
        )


def generate_parameter_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from the sweep config."""
    # Remove script_name from the parameters to sweep over
    sweep_params = {k: v for k, v in sweep_config.items() if k != "script_name"}
    
    # Extract parameter names and their values
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]
    
    # Generate all combinations
    combinations = []
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        combinations.append(param_dict)
    
    return combinations


def parse_extra_args(extra_args: List[str]) -> Dict[str, Any]:
    """Parse extra arguments into a dictionary of parameter overrides."""
    if not extra_args:
        return {}
    
    # Filter out the '--' separator
    filtered_args = [arg for arg in extra_args if arg != '--']
    if not filtered_args:
        return {}
    
    # Parse arguments into a dictionary
    overrides = {}
    for arg in filtered_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Remove leading dashes if present
            key = key.lstrip('-')
            # Try to convert value to appropriate type
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)
            except ValueError:
                pass
            overrides[key] = value
        else:
            # Handle boolean flags
            key = arg.lstrip('-')
            overrides[key] = True
    
    return overrides


def format_parameter_override(param_dict: Dict[str, Any]) -> str:
    """Format parameter dictionary into command line override string."""
    overrides = []
    for key, value in param_dict.items():
        if isinstance(value, str):
            overrides.append(f"{key}='{value}'")
        else:
            overrides.append(f"{key}={value}")
    return " ".join(overrides)


def launch_experiment(
    script: str,
    config: Optional[str],
    param_overrides: str,
    num_nodes: int,
    time: str,
    account: str,
    partition: str,
    container: str,
    mounts: str,
    job_name: str,
    dry_run: bool = False,
    extra_args: List[str] = None,
) -> Tuple[str, Optional[str]]:
    """Launch a single experiment using Slurm."""
    # Construct the command
    command = f"uv run {script}"
    if config:
        command += f" --config {config}"
    
    # Parse extra arguments into overrides
    # Add default args before parsing extra args
    all_args = list(extra_args) if extra_args else []
    default_args = [
        f"logger.log_dir={LOG_DIR}",
        "logger.wandb_enabled=True"
    ]
    all_args.extend(default_args)
    extra_overrides = parse_extra_args(all_args)
    
    # If we have parameter overrides, parse them and merge with extra overrides
    if param_overrides:
        # Split the param_overrides string into key-value pairs
        param_dict = {}
        for param in param_overrides.split():
            if '=' in param:
                key, value = param.split('=', 1)
                # Remove quotes if present
                value = value.strip("'")
                param_dict[key] = value
        
        # Update with extra overrides (they take precedence)
        param_dict.update(extra_overrides)
        command += " " + format_parameter_override(param_dict)
    elif extra_overrides:
        # If no param_overrides but we have extra_overrides, use those
        command += " " + format_parameter_override(extra_overrides)
    
    # Construct the sbatch command
    sbatch_cmd = [
        f"NUM_ACTOR_NODES={num_nodes}",
        f"CONTAINER={container}",
        f"MOUNTS={mounts}",
        f"COMMAND='{command}'",
        "sbatch",
        f"--nodes={num_nodes}",
        f"--account={account}",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--time={time}",
        "--gres=gpu:8",
        "ray.sub",
    ]
    
    # Join the command
    full_cmd = " \\\n".join(sbatch_cmd)
    
    if dry_run:
        return full_cmd, None
    else:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        job_id = result.stdout.strip() if result.stdout else None
        return full_cmd, job_id


def print_experiment_info(
    job_name: str,
    params: Dict[str, Any],
    cmd: str,
    job_id: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Print information about an experiment in a consistent format."""
    print(f"\n[{job_name}]")
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("\nCommand:")
    print(cmd)
    if not dry_run and job_id:
        print(f"\nSubmitted batch job {job_id}")
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    args, extra_args = parse_args()
    
    # Load sweep config if provided
    param_combinations = [{}]  # Default to no parameters
    if args.sweep:
        sweep_config = load_yaml(args.sweep)
        verify_sweep_config(sweep_config, args.script)
        param_combinations = generate_parameter_combinations(sweep_config)
    
    # Print header
    mode = "DRY RUN" if args.dry_run else "SUBMITTING"
    print(f"\n=== {mode} - Experiments ===\n")
    
    # Launch experiments
    for i, params in enumerate(param_combinations):
        # Format parameter overrides
        param_overrides = format_parameter_override(params)
        
        # Generate job name if not provided
        job_name = args.job_name or os.path.splitext(os.path.basename(args.script))[0]
        if len(param_combinations) > 1:
            job_name = f"{job_name}_sweep_{i+1}"
        
        # Launch experiment
        cmd, job_id = launch_experiment(
            script=args.script,
            config=args.config,
            param_overrides=param_overrides,
            num_nodes=args.num_nodes,
            time=args.time,
            account=args.account,
            partition=args.partition,
            container=args.container,
            mounts=args.mounts,
            job_name=job_name,
            dry_run=args.dry_run,
            extra_args=extra_args,
        )
        
        # Print experiment info
        print_experiment_info(
            job_name=job_name,
            params=params,
            cmd=cmd,
            job_id=job_id,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
