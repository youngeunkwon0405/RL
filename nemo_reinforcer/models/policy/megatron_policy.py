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
import gc
import warnings
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Iterable
from functools import partial
import time

import ray
from ray.util.queue import Queue
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.utils import import_class_from_path
from nemo_reinforcer.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)
from nemo_reinforcer.distributed.named_sharding import NamedSharding

# from reinforcer.examples.setup import setup


@ray.remote
class MegatronPolicyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.DEFAULT_VENV

    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__name__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__name__}"

    def __init__(
        self,
        config: PolicyConfig,
        checkpoint_dir: str,
        pre_init_queue: Queue,
        sharding_annotations: NamedSharding,
    ):
        from nemo.tron.init import initialize_megatron
        from nemo.tron.config import (
            ConfigContainer,
            RNGConfig,
            TrainingConfig,
            RerunStateMachineConfig,
            LoggerConfig,
            GPTConfig,
            OptimizerConfig,
            SchedulerConfig,
            CheckpointConfig,
            DistributedInitConfig,
            DistributedDataParallelConfig,
        )
        from megatron.training.global_vars import set_tokenizer
        from nemo.tron.utils.common_utils import get_rank_safe
        from nemo.tron.config import TokenizerConfig
        from nemo.tron.model import get_model_from_config
        from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint

        self.cfg = config
        self.checkpoint_dir = checkpoint_dir

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        hf_model_name = self.cfg["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # If no pad token is defined, you might need:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if the checkpoint already exists
        output_path = f"/opt/checkpoints/tron/{hf_model_name}"
        pt_checkpoint_exists = os.path.exists(output_path) and os.path.exists(
            os.path.join(output_path, "iter_0000000")
        )

        if get_rank_safe() == 0:
            if pt_checkpoint_exists:
                print(f"Checkpoint already exists at {output_path}. Skipping import.")
            else:
                from nemo.tron.converter.llama import HFLlamaImporter

                print(f"Importing model {hf_model_name} to {output_path}...")
                importer = HFLlamaImporter(
                    hf_model_name,
                    output_path=f"/opt/checkpoints/tron/{hf_model_name}",
                )
                importer.apply()
                import megatron.core.rerun_state_machine

                megatron.core.rerun_state_machine.destroy_rerun_state_machine()
            pre_init_queue.put(True)
        else:
            pre_init_queue.get()
            pre_init_queue.put(True)

        pretrained_ckpt = f"/opt/checkpoints/tron/{hf_model_name}"
        pretrained_run_config = os.path.join(
            pretrained_ckpt, "iter_0000000/run_config.yaml"
        )
        cfg_from_pretrained = ConfigContainer.from_yaml(pretrained_run_config)
        model_cfg = cfg_from_pretrained.model_config
        cfg_from_pretrained.logger_config = LoggerConfig()
        cfg_from_pretrained.checkpoint_config = CheckpointConfig(
            save_interval=100,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            pretrained_checkpoint=pretrained_ckpt,  # This is the path to the pretrained ckpt for the SFT case
            async_save=True,
            fully_parallel_save=True,
        )

        model_cfg.tensor_model_parallel_size = self.cfg["tensor_model_parallel_size"]
        model_cfg.pipeline_model_parallel_size = self.cfg[
            "pipeline_model_parallel_size"
        ]
        model_cfg.context_parallel_size = self.cfg["context_parallel_size"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        model_cfg.params_dtype = torch.float32  # self.dtype
        model_cfg.parallel_output = self.cfg["parallel_output"]
        # model_cfg.autocast_enabled = True

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            pretrained_checkpoint=pretrained_ckpt,  # This is the path to the pretrained ckpt for the SFT case
            async_save=True,
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_ckpt,  # This is the path to the pretrained ckpt for the SFT case
            fully_parallel_load=True,  # Enable fully parallel load
        )
        self.megatron_cfg = ConfigContainer(
            model_config=model_cfg,
            checkpoint_config=checkpoint_config,
            logger_config=LoggerConfig(logging_level=0),
            train_config=TrainingConfig(
                micro_batch_size=self.cfg["train_micro_batch_size"],  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=1000,  # Default value for inference
            ),
            optimizer_config=OptimizerConfig(
                optimizer="adam",
                bf16=False,
                fp16=False,
                adam_beta1=0.9,
                adam_beta2=0.999,
                use_distributed_optimizer=True,
                clip_grad=0.0,
                lr=self.cfg["optimizer"]["kwargs"]["lr"],
                weight_decay=self.cfg["optimizer"]["kwargs"]["weight_decay"],
                adam_eps=self.cfg["optimizer"]["kwargs"]["eps"],
                min_lr=self.cfg["optimizer"]["kwargs"]["lr"] * 0.01,
            ),
            ddp_config=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=True,
                use_distributed_optimizer=True,
            ),
            scheduler_config=SchedulerConfig(
                start_weight_decay=self.cfg["optimizer"]["kwargs"]["weight_decay"],
                end_weight_decay=self.cfg["optimizer"]["kwargs"]["weight_decay"],
                weight_decay_incr_style="constant",
                lr_decay_style="constant",  # Default value
                lr_decay_iters=None,  # Default value
                lr_warmup_iters=50,  # No warmup needed for inference
                lr_warmup_init=5e-7,  # Start from 0
            ),
            dataset_config=None,
            tokenizer_config=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        self.megatron_cfg.validate()
        import nemo.tron.setup as setup

        print(f"cfg: {self.megatron_cfg}")
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            _,
            _,
            _,
            self.checkpointing_context,
        ) = setup.setup(self.megatron_cfg)
        self.model = self.model[0]  # Get the first model from the list
        for name, item in self.model.state_dict().items():
            if isinstance(item, torch.Tensor):
                item = item.detach().to(device="cpu", non_blocking=True, copy=True)
            self.model.state_dict()[name] = item

        ref_ckpt_context = setup._init_checkpointing_context(ref_checkpoint_config)
        reference_model = get_model_from_config(
            self.megatron_cfg.model_config,
            self.megatron_cfg.ddp_config,
            use_torch_fsdp2=self.megatron_cfg.dist_config.use_torch_fsdp2,
            overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
            data_parallel_random_init=self.megatron_cfg.rng_config.data_parallel_random_init,
        )
        if (
            ref_checkpoint_config.pretrained_checkpoint is not None
            and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
        ):
            load_checkpoint(
                self.mcore_state,
                reference_model,
                None,  # no optimizer
                None,  # no scheduler
                checkpointing_context=ref_ckpt_context,
                skip_load_to_model_and_opt=setup.HAVE_FSDP2
                and self.megatron_cfg.dist_config.use_torch_fsdp2,
            )
            reference_model = reference_model[0]
            reference_model.eval()
            self.reference_state_dict = {}
            for name, item in reference_model.state_dict().items():
                if isinstance(item, torch.Tensor):
                    item = item.detach().to(device="cpu", non_blocking=True, copy=True)
                self.reference_state_dict[name] = item
            print("Reference model loaded")
        else:
            print("Reference model not loaded")

        for name, item in self.model.state_dict().items():
            if isinstance(item, torch.Tensor):
                item = item.detach().to(device="cuda", non_blocking=True, copy=True)
            self.model.state_dict()[name] = item

        from nemo.tron.tokenizers.tokenizer import build_tokenizer

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=self.cfg["tensor_model_parallel_size"],
        )
        set_tokenizer(self.megatron_tokenizer)
        self.final_padded_vocab_size = tokenizer_config.padded_vocab_size
        self.dp_size = sharding_annotations.get_axis_size("data_parallel")

    def is_alive(self):
        return True

    def get_gpu_info(self):
        """Return information about the GPU being used by this worker."""
        # Basic distributed info
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Device info
        device = torch.cuda.current_device()

        # Get global device ID
        global_device_id = device
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if local_rank < len(cuda_visible_devices):
                global_device_id = int(cuda_visible_devices[local_rank])

        # Get a sample parameter to verify device placement
        param_info = {}
        for _, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param is not None and param.requires_grad:
                    param_info = {
                        "device": str(param.device),
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                    }
                    break
            if param_info:
                break

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device_name": torch.cuda.get_device_name(device),
            "memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
            "global_device_id": global_device_id,
            "parameter_sample": param_info,
            "env_vars": {
                k: v
                for k, v in os.environ.items()
                if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
            },
        }

    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size  # torch.distributed.get_world_size()
        dataset_size = data.size
        self.model.train()

        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.training.utils import get_ltor_masks_and_position_ids
        from megatron.core.models.gpt import GPTModel
        from megatron.core.parallel_state import get_tensor_model_parallel_group
        from nemo_reinforcer.models.megatron.common import forward_step_arbitrary_loss
        from nemo.tron.train import train_step

        forward_step = partial(forward_step_arbitrary_loss, loss_fn=loss_fn)
        all_mb_metrics = []
        for gb_start in range(0, dataset_size, local_gbs):
            (
                loss_metrics,
                skipped_iter,
                should_checkpoint,
                should_exit,
                exit_code,
                grad_norm,
                num_zeros_in_grad,
                curr_lr,
                curr_wd,
            ) = train_step(
                forward_step,
                num_fw_args=3,
                data_iterator=data.slice(
                    gb_start, gb_start + local_gbs
                ).make_microbatch_iterator(mbs),
                model=[self.model],
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                global_state=self.mcore_state,
                num_microbatches_per_dp=local_gbs // mbs,
            )
            loss_metrics["lr"] = curr_lr
            loss_metrics["wd"] = curr_wd

            print(f"Skipped iterations: {skipped_iter}")
            all_mb_metrics.append(loss_metrics)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            loss = torch.tensor(loss_metrics["loss"], device="cuda")
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        metrics = {
            "global_loss": loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "all_mb_metrics": dict(mb_metrics),
        }
        return metrics

    def get_logprobs(self, data: BatchedDataDict) -> BatchedDataDict:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = self.cfg["logprob_batch_size"]
        all_log_probs = []
        self.model.eval()
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.training.utils import get_ltor_masks_and_position_ids
        from megatron.core.models.gpt import GPTModel
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        with torch.no_grad():

            def forward_step_fn(data_iterator: Iterable, model: GPTModel):
                data_dict = next(data_iterator)
                input_ids = data_dict["input_ids"].cuda()
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    input_ids, 0, False, False, False
                )
                output_tensor = model(input_ids, position_ids, attention_mask)

                def collection_fn(output_tensor):
                    tp_grp = get_tensor_model_parallel_group()
                    gathered_output = [
                        torch.zeros_like(output_tensor)
                        for _ in range(torch.distributed.get_world_size(tp_grp))
                    ]
                    torch.distributed.all_gather(
                        gathered_output, output_tensor, group=tp_grp
                    )

                    output_tensor = torch.cat(gathered_output, dim=-1)

                    log_probs = torch.nn.functional.log_softmax(
                        output_tensor.to(torch.float32), dim=-1
                    )
                    token_ids = input_ids
                    next_tokens = token_ids[:, 1:]  # Skip first token
                    log_probs = log_probs[:, :-1]  # Remove last position's logits
                    token_logprobs = log_probs.gather(
                        dim=-1, index=next_tokens.unsqueeze(-1)
                    ).squeeze(-1)

                    # Prepend 0 logprob for first token to maintain same sequence length as input
                    token_logprobs = torch.cat(
                        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                    )

                    return torch.tensor(0.0), {"logprobs": token_logprobs}

                return output_tensor, collection_fn

            forward_backward_func = get_forward_backward_func()
            list_of_logprobs = forward_backward_func(
                forward_step_func=forward_step_fn,
                data_iterator=data.make_microbatch_iterator(logprob_batch_size),
                model=self.model,
                num_microbatches=max(1, data.size // logprob_batch_size),
                seq_length=self.cfg["max_total_sequence_length"],
                micro_batch_size=logprob_batch_size,
                decoder_seq_length=self.cfg["max_total_sequence_length"],
                forward_only=True,
            )
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            logprobs = torch.cat(all_logprobs, dim=0)
            return BatchedDataDict(logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu

        """
        try:
            # Save original references
            model_state_dict = {}
            for name, item in self.model.state_dict().items():
                if isinstance(item, torch.Tensor):
                    item = item.detach().to(device="cpu", non_blocking=True, copy=True)
                model_state_dict[name] = item

            self.model.load_state_dict(self.reference_state_dict, strict=True)
            # for name, item in self.reference_state_dict.items():
            # if isinstance(item, torch.Tensor):
            # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

            gc.collect()
            torch.cuda.empty_cache()

            # - self.model is the original reference_model, now on CUDA
            # - self.reference_model is the original model, now on CPU
            yield

        finally:
            # Restore original references and device placement
            self.model.load_state_dict(model_state_dict, strict=True)
            # for name, item in model_state_dict.items():
            # if isinstance(item, torch.Tensor):
            # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
            # self.model.state_dict()[name] = item

            gc.collect()
            torch.cuda.empty_cache()

    def get_reference_policy_logprobs(self, data: BatchedDataDict) -> BatchedDataDict:
        """Get the logprobs from the reference policy for a batch of data.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(data)

        return_data = BatchedDataDict()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model_config
        from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
            InferenceWrapperConfig,
        )
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )
        from megatron.inference.text_generation.mcore_engine_server import (
            ModelInferenceWrapperServer,
            run_mcore_engine,
        )
        from megatron.core.inference.engines import (
            AbstractEngine,
            StaticInferenceEngine,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=False,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        inference_wrapped_model = ModelInferenceWrapperServer(
            self.model, inference_wrapper_config
        )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )
        inference_engine = StaticInferenceEngine(
            text_generation_controller=text_generation_controller,
            max_batch_size=self.cfg["generation_batch_size"],
        )

        # apply chat template
        out = run_mcore_engine(
            engine=inference_engine,
            prompt_tokens_tensor=data.get("input_ids"),
            prompt_lengths_tensor=data.get("input_lengths"),
            tokens_to_generate=self.cfg["generation"]["max_new_tokens"]
            - data.get("input_ids").size(1),
        )
        # print(out)

        input_lengths = data.get("input_lengths")
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data.get("input_ids").size(0)
        max_seq_len = max([len(tokens) for tokens in out["tokens"]])

        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data.get("input_ids").device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data.get("input_ids").device,
        )

        # Fill in the padded tensors with actual values
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data.get("input_ids").device
            )

            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data.get("input_ids").device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": torch.tensor(
                [len(o) - input_lengths[i] for i, o in enumerate(out["logprobs"])]
            ),
            "unpadded_sequence_lengths": torch.tensor(
                [len(o) for o in out["logprobs"]]
            ),
        }
        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    def zero_out_weights(self):
        """Zero out the weights of the model."""
        pass

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(torch.cuda.current_device())
        return self.device_uuid

    @torch.no_grad()
    def get_weight_ipc_handles(self, offload_model=True):
        pass

    def prepare_for_lp_inference(self):
        self.model.to("cuda")
        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model.to("cuda")
        self.model.train()

        # Move optimizer state to CUDA if it exists
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v) and not v.is_cuda:
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        return
        torch.randn(1).cuda()  # wake up torch allocator
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to("cpu")

        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    @torch.no_grad()
    def offload_after_refit(self):
        # Offload as much as possible on the CPU
        return
        self.model = self.move_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        if self._held_reference_model_params is not None:
            del self._held_reference_model_params
            self._held_reference_model_params = None

        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_to_cpu(self, model):
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        pass

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        pass

    def shutdown(self):
        """Shutdown the policy."""
        #
        pass


class MegatronPolicy(PolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        name_prefix: str = "megatron_policy",
        workers_per_node: Optional[Union[int, List[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        import numpy as np

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                (
                    -1,
                    config["pipeline_model_parallel_size"],
                    config["tensor_model_parallel_size"],
                )
            ),
            names=["data_parallel", "pipeline_model_parallel", "tensor_model_parallel"],
        )

        pre_init_queue = (
            Queue()
        )  # just for communication before torch distributed is set up
        worker_builder = RayWorkerBuilder(
            MegatronPolicyWorker,
            config,
            checkpoint_dir=None,
            pre_init_queue=pre_init_queue,
            sharding_annotations=self.sharding_annotations,
        )

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            workers_per_node=workers_per_node,
            sharding_annotations=self.sharding_annotations,
        )
        self.cfg = config
        self.dp_size = self.sharding_annotations.get_axis_size("data_parallel")

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        return logprobs

    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )
        return logprobs

    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ):
        """Train the policy on a batch of data with a given loss function."""
        # Shard and replicate the batch
        shards = self.dp_size
        sharded_data = data.shard_by_batch_size(
            shards, batch_size=self.cfg["train_global_batch_size"]
        )
        futures = self.worker_group.run_all_workers_sharded_data(
            "train",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "gbs": gbs,
                "mbs": mbs,
            },
        )
        results = self.worker_group.get_all_worker_results(futures)

        # Aggregate the results
        aggregated_results = {}
        aggregated_results["loss"] = results[0]["global_loss"]
        # aggregated_results["lr"] = results[0]["lr"]
        # aggregated_results["wd"] = results[0]["wd"]

        # Aggregate metrics across all workers
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated_results

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        # Verify input data is right-padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "generate",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            common_kwargs={"greedy": greedy},
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
        )
        result = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # Verify the output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def prepare_for_training(self, *args, **kwargs):
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_training", respect_tied_workers=True
        )
        ray.get(futures)
        pass

    def prepare_for_lp_inference(self, *args, **kwargs):
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", respect_tied_workers=True
        )
        ray.get(futures)

    def finish_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def finish_training(self, *args, **kwargs):
        # Placeholder implementation
        pass

    def get_weights_ipc_handles(self):
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles = ray.get(
            [
                worker.get_weight_ipc_handles.remote()
                for worker in self.worker_group.workers
            ]
        )

        # Combine all worker handles into a single dictionary
        all_handles = {}
        for handle in worker_handles:
            all_handles.update(handle)

        return all_handles

    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_before_refit", respect_tied_workers=True
        )
        ray.get(futures)

    def offload_after_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_after_refit", respect_tied_workers=True
        )
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path,
            optimizer_path,
            offload_to_cpu=offload_to_cpu,
            respect_tied_workers=True,
        )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all HF workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self):
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        self.worker_group.shutdown()
