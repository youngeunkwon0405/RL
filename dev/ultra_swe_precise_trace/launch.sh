#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

EXP_NAME="${EXP_NAME:-ultra-swe-precise-trace-one-step}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/ultra/swe_teacher.yaml}"
MODEL_PATH="${MODEL_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/liding/public/ultra_v3/ultra-green-step42}"
DATA_DIR="${DATA_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/youngeunk/data/ultra-blends}"
TRAIN_PATH="${TRAIN_PATH:-${DATA_DIR}/swe.train.jsonl}"
VAL_PATH="${VAL_PATH:-${DATA_DIR}/swe.val.jsonl}"
SIF_DIR="${SIF_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/nemotron-ultra-swe}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.nightly.sqsh}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/youngeunk/cache/nemotron_ultra_swe_precise_trace}"
HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/youngeunk/hf_home}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/${EXP_NAME}}"

OPENHANDS_REPO="${OPENHANDS_REPO:-https://github.com/youngeunkwon0405/nv-OpenHands.git}"
OPENHANDS_COMMIT="${OPENHANDS_COMMIT:-e806c4b4510639ca7deefb4e8254fc6c61773075}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-short}"
WALLTIME="${WALLTIME:-4:00:00}"
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-64}"
NUM_GEN_NODES="${NUM_GEN_NODES:-64}"
NUM_GYM_NODES="${NUM_GYM_NODES:-0}"
NRL_MAX_STEPS="${NRL_MAX_STEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"

for required in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_PATH}" \
  "${VAL_PATH}" \
  "${CONTAINER}" \
  "${SANDBOX_CONTAINER}" \
  "${REPO_ROOT}/ultra_launch.sh" \
  "${REPO_ROOT}/${CONFIG_PATH}" \
  "${REPO_ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required path does not exist: ${required}" >&2
    exit 1
  fi
done

if [[ $(wc -l < "${TRAIN_PATH}") -ne 7716 ]]; then
  echo "ERROR: expected 7,716 rows in ${TRAIN_PATH}" >&2
  exit 1
fi
if [[ $(wc -l < "${VAL_PATH}") -ne 100 ]]; then
  echo "ERROR: expected 100 rows in ${VAL_PATH}" >&2
  exit 1
fi

GYM_APP="${REPO_ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py"
for marker in generation_start_timestamp evaluation_start_timestamp per_turn_metrics; do
  if ! grep -q "${marker}" "${GYM_APP}"; then
    echo "ERROR: precise Gym instrumentation marker is missing: ${marker}" >&2
    exit 1
  fi
done

while IFS=$'\t' read -r dataset_name instance_id; do
  case "${dataset_name}" in
    *SWE-rebench*) sif="${SIF_DIR}/swerebench/${instance_id}.sif" ;;
    *SWE-Gym*) sif="${SIF_DIR}/swegym/sweb.eval.arm64.${instance_id}.sif" ;;
    *)
      echo "ERROR: unsupported dataset ${dataset_name} for ${instance_id}" >&2
      exit 1
      ;;
  esac
  if [[ ! -f "${sif}" ]]; then
    echo "ERROR: missing SIF for one-step prompt ${instance_id}: ${sif}" >&2
    exit 1
  fi
done < <(
  head -n 32 "${TRAIN_PATH}" |
    jq -r '[.responses_create_params.metadata.dataset_name,.responses_create_params.metadata.instance_id] | @tsv'
)

mkdir -p "${PERSISTENT_CACHE}" "${HF_HOME}" "${RESULTS_DIR}"

export EXP_NAME CONFIG_PATH MODEL_PATH TRAIN_PATH VAL_PATH SIF_DIR CONTAINER SANDBOX_CONTAINER
export PERSISTENT_CACHE HF_HOME RESULTS_DIR
export SLURM_ACCOUNT SLURM_PARTITION SLURM_QOS WALLTIME
export NUM_TRAIN_NODES NUM_GEN_NODES NUM_GYM_NODES NRL_MAX_STEPS DRY_RUN
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-1}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"
export USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${PERSISTENT_CACHE}/component-venvs}"
export PYTHON_RUNNER="${PYTHON_RUNNER:-/opt/nemo_rl_venv/bin/python}"

# This validation uses local/public assets and intentionally avoids forwarding
# login-shell credentials into the generated command or provenance file.
unset WANDB_API_KEY WANDB_ENTITY
export HF_TOKEN=""

cat <<EOF
Ultra SWE precise-trace validation
  NeMo RL:    $(git -C "${REPO_ROOT}" rev-parse HEAD)
  Gym:        $(git -C "${REPO_ROOT}/3rdparty/Gym-workspace/Gym" rev-parse HEAD)
  OpenHands:  ${OPENHANDS_REPO}@${OPENHANDS_COMMIT}
  Config:     ${CONFIG_PATH}
  Model:      ${MODEL_PATH}
  Data:       ${TRAIN_PATH}
  SIF root:   ${SIF_DIR}
  Container:  ${CONTAINER}
  Shape:      ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} generation + ${NUM_GYM_NODES} Gym nodes
  Batch:      32 prompts x 16 generations = 512 rollouts
  Context:    196,608 tokens
  Steps:      ${NRL_MAX_STEPS}
EOF

cd "${REPO_ROOT}"
bash ultra_launch.sh \
  checkpointing.enabled=false \
  logger.monitor_gpus=false \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_framework_repo=${OPENHANDS_REPO}" \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_framework_commit=${OPENHANDS_COMMIT}" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_framework_repo=${OPENHANDS_REPO}" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_framework_commit=${OPENHANDS_COMMIT}"
