#!/usr/bin/env python3
"""Build the Ultra SWE actor environments once on shared storage."""

import os
from pathlib import Path

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.utils.venvs import create_local_venv


COMPONENTS = (
    (
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        PY_EXECUTABLES.VLLM,
    ),
    (
        "nemo_rl.environments.nemo_gym.NemoGym",
        PY_EXECUTABLES.NEMO_GYM,
    ),
    (
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
        PY_EXECUTABLES.MCORE,
    ),
)

SYSTEM_PYTHON = Path("/usr/bin/python3.12")


def main() -> None:
    venv_root = os.environ.get("NEMO_RL_VENV_DIR")
    if not venv_root:
        raise RuntimeError("NEMO_RL_VENV_DIR must point to shared storage")

    if not SYSTEM_PYTHON.is_file():
        raise RuntimeError(f"container Python does not exist: {SYSTEM_PYTHON}")

    Path(venv_root).mkdir(parents=True, exist_ok=True)
    for name, command in COMPONENTS:
        print(f"Building {name} in {venv_root}", flush=True)
        python_path = Path(create_local_venv(command, name, force_rebuild=False))
        if python_path.resolve() != SYSTEM_PYTHON:
            python_path.unlink()
            python_path.symlink_to(SYSTEM_PYTHON)
        print(f"Ready: {python_path}", flush=True)


if __name__ == "__main__":
    main()
