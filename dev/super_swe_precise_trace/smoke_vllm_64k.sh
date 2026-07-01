#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/nemo-rl-internal/super_models/swe_pivot/hf}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
PORT="${PORT:-18000}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
LOG_PATH="${LOG_PATH:-/tmp/super-vllm-64k-smoke.log}"

: "${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR must point to the shared actor environments}"
: "${UV_PYTHON_INSTALL_DIR:?UV_PYTHON_INSTALL_DIR must point to the shared uv Python runtime}"
VLLM_PY="${VLLM_PY:-${NEMO_RL_VENV_DIR}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"

for required in "${MODEL_PATH}/config.json" "${MODEL_PATH}/configuration_nemotron_h.py" "${VLLM_PY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required path does not exist: ${required}" >&2
    exit 1
  fi
done

python_header_dir=$(printf '%s\n' "${UV_PYTHON_INSTALL_DIR}"/cpython-3.12.*-linux-aarch64-gnu/include/python3.12 | sort -V | tail -n 1)
if [[ ! -f "${python_header_dir}/Python.h" ]]; then
  echo "ERROR: shared Python headers do not exist under ${UV_PYTHON_INSTALL_DIR}" >&2
  exit 1
fi
if [[ ! -e /usr/include/python3.12 ]]; then
  ln -s "${python_header_dir}" /usr/include/python3.12
elif [[ ! -f /usr/include/python3.12/Python.h ]]; then
  echo "ERROR: /usr/include/python3.12 exists without Python.h" >&2
  exit 1
fi

if [[ -f /opt/nemo-rl/3rdparty/vllm/nemo-rl.env ]]; then
  # shellcheck disable=SC1091
  source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env
fi

export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency

"${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name super-v3-smoke \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 4 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  >"${LOG_PATH}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "VLLM_64K_READY model=${MODEL_PATH} max_model_len=${MAX_MODEL_LEN}"
    exit 0
  fi

  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "ERROR: vLLM exited before becoming healthy" >&2
    tail -n 120 "${LOG_PATH}" >&2
    wait "${server_pid}"
  fi
  sleep 5
done

echo "ERROR: vLLM did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s" >&2
tail -n 120 "${LOG_PATH}" >&2
exit 1
