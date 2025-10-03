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

import copy
import gc
import os
import sys
from typing import Any, Optional, cast

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation
from nemo_rl.models.huggingface.common import ModelFlag
from nemo_rl.models.policy.utils import is_vllm_v1_engine_enabled
from nemo_rl.utils.nsys import wrap_with_nvtx_name


# Use a base class to share some functions to avoid code duplication.
class BaseVllmGenerationWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        return f"{self.__class__.__name__}"

    @staticmethod
    def configure_worker(
        num_gpus: int | float, bundle_indices: Optional[tuple[int, list[int]]] = None
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Provides complete worker configuration for vLLM tensor and pipeline parallelism.

        This method configures the worker based on its role in tensor and pipeline parallelism,
        which is determined directly from the bundle_indices parameter.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for parallelism (if applicable)

        Returns:
            tuple with complete worker configuration:
              - 'resources': Resource allocation (e.g., num_gpus)
              - 'env_vars': Environment variables for this worker
              - 'init_kwargs': Parameters to pass to __init__ of the worker
        """
        # Initialize configuration
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        local_bundle_indices = None
        if bundle_indices is not None:
            node_idx = bundle_indices[0]
            local_bundle_indices = bundle_indices[1]
            init_kwargs["bundle_indices"] = local_bundle_indices

            """
            compute a unique seed from the node_idx and bundle_indices:
            node_idx = 0, bundle_indices = [0, 1, 2, 3] -> seed = 0*1024 + 0
            node_idx = 0, bundle_indices = [4, 5, 6, 7] -> seed = 0*1024 + 1
            node_idx = 1, bundle_indices = [0, 1, 2, 3] -> seed = 1*1024 + 0
            node_idx = 1, bundle_indices = [4, 5, 6, 7] -> seed = 1*1024 + 1
            """
            # For single worker groups, use a simpler seed calculation
            if len(local_bundle_indices) == 1:
                seed = node_idx * 1024 + local_bundle_indices[0]
            else:
                # For parallel groups, use the original calculation
                bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
                seed = node_idx * 1024 + bundle_id

            init_kwargs["seed"] = seed
            # Need to give each DP group its own vllm cache to address:
            # https://github.com/vllm-project/vllm/issues/18851
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm_{seed}")

        # Check if this worker is part of a parallel group (TP or TP+PP).
        # A worker is part of a parallel group if it's a secondary member (local_bundle_indices is None)
        # or if it's a primary member of a group with multiple workers.
        is_part_of_parallel_workers = (
            local_bundle_indices is not None and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None

        if is_part_of_parallel_workers:
            # Ray + vllm likes to manage GPU assignment internally for parallel groups
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            init_kwargs["fraction_of_gpus"] = num_gpus

        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Skip vllm P2P check and rely on driver to report peer to peer capability.
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"

        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: VllmConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize a vLLM worker for distributed inference.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
                          Only needed for the first worker in each tied worker group.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
        """
        self.cfg = config

        self.model_name = self.cfg["model_name"]
        self.tensor_parallel_size = self.cfg["vllm_cfg"]["tensor_parallel_size"]
        self.pipeline_parallel_size = self.cfg["vllm_cfg"]["pipeline_parallel_size"]
        self.expert_parallel_size = self.cfg["vllm_cfg"]["expert_parallel_size"]
        self.enable_expert_parallel = self.expert_parallel_size > 1
        self.gpu_memory_utilization = self.cfg["vllm_cfg"]["gpu_memory_utilization"]
        self.precision = self.cfg["vllm_cfg"]["precision"]
        self.fraction_of_gpus = fraction_of_gpus
        self.is_model_owner = bundle_indices is not None

        # Store the Python executable being used by this worker
        self.py_executable = sys.executable

        # Skip model loading if we're not the model owner
        if not self.is_model_owner:
            self.llm = None
            self.tokenizer = None
            self.rank = 0
            self.world_size = 1
            return

        # In Ray+vLLM setup, each worker process considers itself rank 0
        # vLLM handles the parallelism internally through Ray
        self.rank = 0
        self.world_size = 1

        # Monkey patch for vLLM to ensure RAY_ADDRESS is set in Ray actors.
        try:
            import vllm.utils
            from vllm.logger import init_logger
            from vllm.utils import cuda_is_initialized, is_in_ray_actor

            logger = init_logger("vllm_patch")

            def _patched_maybe_force_spawn():
                """Patched version of vllm.utils._maybe_force_spawn.

                This patch changes an `elif is_in_ray_actor()` to an `if` statement.
                This ensures that `os.environ["RAY_ADDRESS"]` is set when running
                within a Ray actor, even if CUDA has already been initialized.
                This is crucial for vLLM workers to connect back to the Ray cluster.
                """
                if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
                    return

                reason = None
                if cuda_is_initialized():
                    reason = "CUDA is initialized"

                if is_in_ray_actor():
                    # even if we choose to spawn, we need to pass the ray address
                    # to the subprocess so that it knows how to connect to the ray cluster.
                    # env vars are inherited by subprocesses, even if we use spawn.
                    import ray

                    os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
                    if reason is None:
                        reason = "In a Ray actor and can only be spawned"

                if reason is not None:
                    logger.warning(
                        "We must use the `spawn` multiprocessing start method. "
                        "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                        "See https://docs.vllm.ai/en/latest/getting_started/"
                        "troubleshooting.html#python-multiprocessing "
                        "for more information. Reason: %s",
                        reason,
                    )
                    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            vllm.utils._maybe_force_spawn = _patched_maybe_force_spawn
            logger.info("Successfully patched vllm.utils._maybe_force_spawn.")

            def _patch_vllm_init_workers_ray():
                """Patch the vLLM ray_distributed_executor.py file.

                1. Pass custom runtime_env in _init_workers_ray call.
                    - This allows passing custom py_executable to worker initialization.
                2. Add NCCL_CUMEM_ENABLE and NCCL_NVLS_ENABLE to vLLM ADDITIONAL_ENV_VARS.
                    - This is a workaround to fix async vllm in some scenarios.
                    - See https://github.com/NVIDIA-NeMo/RL/pull/898 for more details.
                """
                try:
                    import vllm.executor.ray_distributed_executor as ray_executor_module

                    file_to_patch = ray_executor_module.__file__

                    with open(file_to_patch, "r") as f:
                        content = f.read()

                    old_lines = [
                        "self._init_workers_ray(placement_group)",
                        'ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}',
                    ]

                    new_lines = [
                        f'self._init_workers_ray(placement_group, runtime_env={{"py_executable": "{self.py_executable}"}})',
                        'ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE"}',
                    ]

                    need_replace = False
                    for old_line, new_line in zip(old_lines, new_lines):
                        if new_line in content or old_line not in content:
                            continue
                        content = content.replace(old_line, new_line)
                        need_replace = True

                    if not need_replace:
                        return

                    # Write back the patched content
                    with open(file_to_patch, "w") as f:
                        f.write(content)

                except (ImportError, FileNotFoundError, PermissionError):
                    # Allow failures gracefully
                    pass

            _patch_vllm_init_workers_ray()
            logger.info("Successfully patched vllm _init_workers_ray.")

            # Patch the vLLM sampler.py file to modify logprobs computation wrt temperature.
            # This replaces raw_logprobs = self.compute_logprobs(logits) with custom temperature-applied logprobs.
            # TODO(zhanda): This is only a temporary fix to address the issue of incorrect logprobs returned by vllm
            # and should be removed or improved after vllm's new logprobs option is released. And currently, other
            # sampling parameters like top_p, top_k, etc. are not supported.
            # See https://github.com/NVIDIA-NeMo/RL/issues/69 for more details.
            def _patch_vllm_sampler():
                try:
                    import vllm.v1.sample.sampler as sampler_module

                    file_to_patch = sampler_module.__file__

                    with open(file_to_patch, "r") as f:
                        content = f.read()

                    old_line = "raw_logprobs = self.compute_logprobs(logits)"
                    new_lines = "raw_logprobs = self.compute_logprobs(self.apply_temperature(logits.to(torch.float32), sampling_metadata.temperature) if sampling_metadata.temperature is not None else logits)"

                    if new_lines in content:
                        return

                    if old_line not in content:
                        return

                    # Replace all instances of the old line with the new lines
                    patched_content = content.replace(old_line, new_lines)

                    # Write back the patched content
                    with open(file_to_patch, "w") as f:
                        f.write(patched_content)

                except (ImportError, FileNotFoundError, PermissionError):
                    # Allow failures gracefully
                    pass

            _patch_vllm_sampler()

        except (ImportError, AttributeError):
            # vllm not installed or has a different structure, skipping patch.
            pass

        try:
            import vllm

            self.SamplingParams = vllm.SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
                "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
                "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
                "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
            )
        vllm_kwargs: dict[str, Any] = copy.deepcopy(self.cfg.get("vllm_kwargs", {}))

        # Calculate total parallel size (TP * PP)
        model_parallel_size = self.tensor_parallel_size * self.pipeline_parallel_size

        # Special handling for parallel case (either TP or PP or both)
        if model_parallel_size > 1:
            # Configure vLLM for tensor/pipeline parallelism within Ray
            # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(
                self.fraction_of_gpus / model_parallel_size
            )

            # Set bundle indices for parallel workers
            bundle_indices_str = ",".join(map(str, bundle_indices))
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = bundle_indices_str
            print(
                f"VLLM_RAY_BUNDLE_INDICES environment variable set to: {os.environ.get('VLLM_RAY_BUNDLE_INDICES')}"
            )

            # Use Ray for distributed execution in parallel mode
            vllm_kwargs["distributed_executor_backend"] = "ray"
        else:
            # For non-parallel mode, explicitly set executor to None to avoid Ray issues
            vllm_kwargs["distributed_executor_backend"] = None

        os.environ["VLLM_USE_V1"] = "1" if is_vllm_v1_engine_enabled() else "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # We should use vLLM DP if ep_size > tp_size since EP_SIZE = DP_SIZE * TP_SIZE in vLLM.
        # See details in https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py
        if self.expert_parallel_size > self.tensor_parallel_size:
            # set vLLM DP rank
            world_size = int(os.environ["VLLM_DP_SIZE"]) * model_parallel_size
            rank = int(os.environ["RANK"]) % world_size
            os.environ["VLLM_DP_RANK"] = str(rank // model_parallel_size)
            os.environ["VLLM_DP_RANK_LOCAL"] = str((rank % 8) // model_parallel_size)
            # set vLLM DP address and port
            leader_rank = int(os.environ["RANK"]) // world_size * world_size
            addr_list = eval(os.environ["AVAILABLE_ADDR_LIST"])
            port_list = eval(os.environ["AVAILABLE_PORT_LIST"])
            os.environ["VLLM_DP_MASTER_IP"] = addr_list[leader_rank]
            os.environ["VLLM_DP_MASTER_PORT"] = str(port_list[leader_rank])

        load_format = self.cfg["vllm_cfg"]["load_format"]
        if ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(self.model_name):
            load_format = "auto"

        if (
            len(get_nsight_config_if_pattern_matches("vllm_generation_worker")) > 0
            and vllm_kwargs["distributed_executor_backend"] == "ray"
        ):
            logger.warning(
                "Nsight profiling is enabled for vllm generation worker through the vllm ray distributed executor. "
                "The nsight command-line args and output file names are automatically picked by the ray distributed "
                "executor. Refer to https://github.com/vllm-project/vllm/blob/7e3a8dc90670fd312ce1e0d4eba9bf11c571e3ad/vllm/executor/ray_distributed_executor.py#L136 "
                "for more information."
            )
            vllm_kwargs["ray_workers_use_nsight"] = True

        if self.cfg["vllm_cfg"]["precision"] == "fp8":
            from nemo_rl.models.generation.fp8 import init_fp8

            fp8_kwargs = init_fp8(
                self.cfg["vllm_cfg"], self.model_name, model_parallel_size
            )
            vllm_kwargs.update(fp8_kwargs)
            # overriden by quant config, however vllm complains if this not passed
            self.precision = "bfloat16"

        llm_kwargs = dict(
            model=self.model_name,
            load_format=load_format,
            # vllm==0.10.0 breaks skip_tokenizer_init=True.
            # This will be reverted to `self.cfg["vllm_cfg"]["skip_tokenizer_init"]` once https://github.com/NVIDIA-NeMo/RL/issues/818 is resolved.
            skip_tokenizer_init=False,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=torch.cuda.get_device_capability()[0] >= 8,
            dtype=self.precision,
            seed=seed,
            enforce_eager=self.cfg["vllm_cfg"]["enforce_eager"],
            max_model_len=self.cfg["vllm_cfg"]["max_model_len"],
            trust_remote_code=True,
            worker_extension_cls="nemo_rl.models.generation.vllm.vllm_backend.VllmInternalWorkerExtension",
            enable_sleep_mode=True,
            disable_log_stats=True,
            logprobs_mode="raw_logprobs",
            **vllm_kwargs,
        )

        self._create_engine(llm_kwargs)

        # will be initialized in post_init
        # used in update_weights_from_ipc_handles
        self.vllm_device_ids = None

    def llm(self):
        return self.llm

    def is_alive(self):
        """Check if the worker is alive."""
        return True

    def _merge_stop_strings(self, batch_stop_strings):
        stop_set: set[str] = set()

        if self.cfg.get("stop_strings"):
            stop_set.update(self.cfg["stop_strings"])

        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)

        return list(stop_set) if stop_set else None

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings,
        max_new_tokens: Optional[int] = None,
    ):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        temperature = 0.0 if greedy else self.cfg["temperature"]

        max_tokens = (
            max_new_tokens if max_new_tokens is not None else self.cfg["max_new_tokens"]
        )

        return self.SamplingParams(
            temperature=temperature,
            top_p=self.cfg["top_p"],
            top_k=top_k_val,
            max_tokens=max_tokens,
            logprobs=0,
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,
        )

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()
        if self.llm is not None:
            self.llm.collective_rpc("start_gpu_profiling", args=tuple())

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
        if self.llm is not None:
            self.llm.collective_rpc("stop_gpu_profiling", args=tuple())


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)  # pragma: no cover
class VllmGenerationWorker(BaseVllmGenerationWorker):
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        import vllm

        self.llm = vllm.LLM(**llm_kwargs)

    def post_init(self):
        self.vllm_device_ids = self.report_device_id()

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        self.llm.collective_rpc(
            "init_collective",
            args=(
                rank_prefix,
                ip,
                port,
                world_size,
                train_world_size,
            ),
        )

    @wrap_with_nvtx_name("vllm_genertion_worker/generate")
    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            # Return empty BatchedDataDict with all required fields
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                }
            )

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        sampling_params = self._build_sampling_params(
            greedy=greedy,
            stop_strings=stop_strings,
        )

        # verify inputs have correct padding
        verify_right_padding(data, pad_value=self.cfg["pad_token_id"])

        # Original input length with padding
        padded_input_length = input_ids.size(1)

        # Convert inputs to vLLM format
        prompts = format_prompt_for_vllm_generation(data)

        # Generate outputs
        assert self.llm is not None, (
            "Attempting to generate with either an uninitialized vLLM or non-model-owner"
        )
        outputs = self.llm.generate(prompts, sampling_params)

        # Process the outputs - but preserve the original input padding structure
        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []
        max_length = 0
        for output in outputs:
            max_length = max(max_length, len(output.outputs[0].token_ids))

        for i, output in enumerate(outputs):
            # Extract generated tokens
            sequence_length = input_lengths[i]
            generation = output.outputs[0]
            generated_tokens = list(generation.token_ids)

            # Calculate total sequence length (original input length + generated tokens)
            total_length = padded_input_length + max_length

            # Create a new tensor with the right size and fill with padding token
            full_output = torch.full(
                (total_length,), self.cfg["pad_token_id"], dtype=input_ids.dtype
            )

            # Copy original input (with padding) into the beginning
            full_output[:sequence_length] = input_ids[i][:sequence_length]

            # Add generated tokens after the original input
            full_output[sequence_length : sequence_length + len(generated_tokens)] = (
                torch.tensor(generated_tokens)
            )

            output_ids_list.append(full_output)
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if hasattr(generation, "logprobs") and generation.logprobs:
                try:
                    for idx, logprob_dict in enumerate(generation.logprobs):
                        if logprob_dict:
                            position = sequence_length + idx
                            full_logprobs[position] = next(iter(logprob_dict.items()))[
                                1
                            ].logprob
                except Exception:
                    import traceback

                    traceback.print_exc()

            logprobs_list.append(full_logprobs)

            response_length = sequence_length + len(generated_tokens)
            generation_lengths.append(len(generated_tokens))
            unpadded_sequence_lengths.append(response_length)
            assert response_length <= self.llm.llm_engine.model_config.max_model_len, (
                f"response_length={response_length} > max_model_len={self.llm.llm_engine.model_config.max_model_len}, which should not happen. Please check this behavior in isolation by running `uv run --extra vllm tools/model_diagnostics/1.max_model_len_respected.py {self.llm.llm_engine.model_config.model}` and raise this issue with the vllm team."
            )

        # Create return data conforming to GenerationOutputSpec
        output_ids = torch.stack(output_ids_list)
        logprobs = torch.stack(logprobs_list)

        return_data = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "logprobs": logprobs,
                "generation_lengths": torch.tensor(
                    generation_lengths, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths, dtype=torch.long
                ),
            }
        )

        return return_data

    @wrap_with_nvtx_name("vllm_genertion_worker/generate_text")
    def generate_text(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate text responses using vLLM generation.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict containing:
                - texts: List of generated text responses
        """
        # Check if async engine is enabled
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text cannot be used with async_engine=True. Use generate_text_async instead."
            )

        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * len(data["prompts"])
        )

        # This function requires all generations have the same stop strings, so we collect all here
        stop_strings: set[str] = set()
        for sample_stop_strings in batch_stop_strings:
            if sample_stop_strings:
                stop_strings.update(sample_stop_strings)

        # Add default stop strings from config
        if self.cfg.get("stop_strings", None):
            stop_strings.update(self.cfg["stop_strings"])

        stop_strings = list(stop_strings) if len(stop_strings) > 0 else None

        # Read generation parameters from config
        top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
        sampling_params = self.SamplingParams(
            temperature=self.cfg["temperature"] if not greedy else 0,
            top_p=self.cfg["top_p"],
            top_k=top_k if not greedy else 1,
            max_tokens=self.cfg["max_new_tokens"],
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,  # returning stop strings like hf
        )

        # Generate outputs
        assert self.llm is not None, (
            "Attempting to generate with either an uninitialized vLLM or non-model-owner"
        )
        outputs = self.llm.generate(data["prompts"], sampling_params)
        texts = [output.outputs[0].text for output in outputs]

        # Convert to BatchedDataDict
        return_data: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict(
            {"texts": texts}
        )
        return return_data

    def report_device_id(self) -> list[str]:
        """Report device ID from the vLLM worker."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id cannot be used with async_engine=True. Use report_device_id_async instead."
            )

        list_of_worker_results = self.llm.collective_rpc(
            "report_device_id", args=tuple()
        )
        return cast(list[str], list_of_worker_results)

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare the info for refit."""
        self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    @wrap_with_nvtx_name("vllm_genertion_worker/update_weights_from_ipc_handles")
    def update_weights_from_ipc_handles(self, ipc_handles: dict[str, Any]) -> bool:
        """Update weights from IPC handles by delegating to the vLLM Worker implementation.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_ipc_handles cannot be used with async_engine=True. Use update_weights_from_ipc_handles_async instead."
                )

            if self.tensor_parallel_size == 1:
                # UniProcExecutor
                assert len(self.vllm_device_ids) == 1
                result_or_coro = self.llm.collective_rpc(
                    "update_weights_from_local_ipc_handles",
                    args=(ipc_handles[self.vllm_device_ids[0]],),
                )
            else:
                """
                DO NOT USE VLLM's collective_rpc: This code causes duplicate IPC data transfer across Ray workers,
                leading to unnecessary network serialization overhead and potential performance degradation.

                result_or_coro = self.llm.collective_rpc(
                    "update_weights_from_global_ipc_handles", args=(ipc_handles,)
                )
                """
                ray_worker_outputs = []
                # MultiProcExecutor
                for worker, device_id in zip(
                    self.llm.llm_engine.model_executor.workers, self.vllm_device_ids
                ):
                    ray_worker_outputs.append(
                        worker.execute_method.remote(
                            "update_weights_from_local_ipc_handles",
                            ipc_handles[device_id],
                        )
                    )

                # Gather the results
                result_or_coro = ray.get(ray_worker_outputs)

            worker_result = result_or_coro[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    @wrap_with_nvtx_name("vllm_genertion_worker/update_weights_from_collective")
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective can only be used with async_engine=False. Use update_weights_from_collective_async instead."
                )

            result_or_coro = self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
            )
            worker_result = result_or_coro[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    def reset_prefix_cache(self):
        """Reset the prefix cache of vLLM engine."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache can only be used with async_engine=False. Use reset_prefix_cache_async instead."
            )

        self.llm.llm_engine.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def sleep(self):
        """Put the vLLM engine to sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep cannot be used with async_engine=True. Use sleep_async instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        self.llm.llm_engine.reset_prefix_cache()
        self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, **kwargs):
        """Wake up the vLLM engine."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up cannot be used with async_engine=True. Use wake_up_async instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        self.llm.wake_up(**wake_up_args)

    def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            if self.llm is not None:
                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
