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


def main() -> None:
    venv_root = os.environ.get("NEMO_RL_VENV_DIR")
    if not venv_root:
        raise RuntimeError("NEMO_RL_VENV_DIR must point to shared storage")

    Path(venv_root).mkdir(parents=True, exist_ok=True)
    for name, command in COMPONENTS:
        print(f"Building {name} in {venv_root}", flush=True)
        path = create_local_venv(command, name, force_rebuild=False)
        print(f"Ready: {path}", flush=True)


if __name__ == "__main__":
    main()
