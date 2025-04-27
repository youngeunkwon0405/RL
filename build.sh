PATH=/usr/bin:$PATH
set -eoux pipefail
rm -rf .venv
uv venv
uv pip install torch==2.6.0 setuptools
#uv sync --no-build-isolation
#uv run --extra mcore --no-build-isolation te.py
# commenting out above uv run --extra mcore --no-build-isolation te.py to speed up


# This is just to check everything installed correctly
uv run --extra mcore --no-build-isolation echo good
( cd 3rdparty/Megatron-LM/megatron/core/datasets; uv run --with pybind11 make )

uv run python <<"EOF"
print(0)
from megatron.training.utils import get_ltor_masks_and_position_ids
exit(0)
print(1)
from nemo.tron.init import initialize_megatron
print(2)
from nemo.tron.config import (
    ConfigContainer,
    TrainingConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
    DistributedDataParallelConfig,
)
print(3)
from nemo.tron.utils.common_utils import get_rank_safe
print(4)
from nemo.tron.config import TokenizerConfig
print(5)
from nemo.tron.model import get_model_from_config
print(6)
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint
print(7)
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
print(8)
from nemo.tron.setup import _init_checkpointing_context, _update_model_config_funcs
print(9)
from nemo.tron.state import GlobalState
print(10)
from nemo.tron.optim import setup_optimizer
print(11)
from nemo.tron import fault_tolerance
print(12)
from nemo.tron.tokenizers.tokenizer import build_tokenizer
print(13)
from nemo.tron.utils.train_utils import (
    calc_params_l2_norm,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
print(14)
from nemo.tron.train import train_step
print(15)
from nemo.tron.setup import HAVE_FSDP2
print(16)
EOF
