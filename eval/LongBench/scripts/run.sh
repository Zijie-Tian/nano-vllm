#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/3rdparty/nanovllm:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:-/home/zijie/models}"
BASE_OUTPUT_ROOT="${SCRIPT_DIR}/../benchmark_root"
OUTPUT_ROOT="${BASE_OUTPUT_ROOT}"
OUTPUT_ROOT_EXPLICIT=0
DATA_ROOT="${LONG_BENCH_DATA_ROOT:-$HOME/data/LongBench}"
MODEL_REF=""
TASK_SET="all"
MODEL_PATH=""
MODEL_NAME=""
BACKEND=""
TOKENIZER_PATH=""
TEMPLATE_TYPE=""
DTYPE_OVERRIDE=""
MAX_MODEL_LEN=""
E_FLAG=()
TASK_OVERRIDE=""
NUM_SAMPLES_OVERRIDE=""
NANOVLLM_CPU_OFFLOAD=0
NANOVLLM_NUM_GPU_BLOCKS=""
NANOVLLM_GPU_MEMORY_UTILIZATION=""
NANOVLLM_BLOCK_SIZE=""
NANOVLLM_ENFORCE_EAGER=0
NANOVLLM_SPARSE_POLICY=""
NANOVLLM_SPARSE_STRIDE=""
NANOVLLM_SPARSE_THRESHOLD=""
NANOVLLM_SPARSE_CHUNK_SIZE=""
NANOVLLM_COMPASS_TOP_P=""
NANOVLLM_COMPASS_LAMBDA=""
NANOVLLM_COMPASS_THETA=""
BLASST_LAMBDA=""
EXTRA_ARGS=()

sanitize_path_component() {
  local value="${1:-}"
  value="${value,,}"
  value="${value//,/+}"
  value="$(printf '%s' "${value}" | sed -E 's#[^a-z0-9._+-]+#-#g; s#-+#-#g; s#(^[-._+]+|[-._+]+$)##g')"
  printf '%s' "${value:-default}"
}

append_settings_component() {
  local prefix="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    SETTINGS_INFO+="${prefix}$(sanitize_path_component "${value}")_"
  fi
}

resolve_task_label() {
  local raw_task_label=""
  if [[ -n "${TASK_OVERRIDE}" ]]; then
    raw_task_label="tasks_${TASK_OVERRIDE}"
  else
    raw_task_label="${TASK_SET}"
  fi

  if [[ ${#E_FLAG[@]} -gt 0 ]]; then
    raw_task_label="longbench-e_${raw_task_label}"
  fi

  sanitize_path_component "${raw_task_label}"
}

resolve_results_root() {
  local strategy_label=""
  local policy_upper=""
  SETTINGS_INFO=""

  append_settings_component "" "${BACKEND}"

  if [[ "${BACKEND}" == "torch" ]]; then
    strategy_label="full"
  else
    strategy_label="${NANOVLLM_SPARSE_POLICY:-FULL}"
  fi
  append_settings_component "" "${strategy_label}"

  policy_upper="${strategy_label^^}"
  if [[ "${BACKEND}" == "nanovllm" ]]; then
    case "${policy_upper}" in
      FULL)
        ;;
      COMPASS)
        append_settings_component "topp" "${NANOVLLM_COMPASS_TOP_P}"
        append_settings_component "lambda" "${NANOVLLM_COMPASS_LAMBDA}"
        append_settings_component "theta" "${NANOVLLM_COMPASS_THETA}"
        ;;
      BLASST)
        append_settings_component "lambda" "${BLASST_LAMBDA}"
        ;;
      *)
        append_settings_component "stride" "${NANOVLLM_SPARSE_STRIDE:-8}"
        append_settings_component "thresh" "${NANOVLLM_SPARSE_THRESHOLD:-0.9}"
        append_settings_component "chunk" "${NANOVLLM_SPARSE_CHUNK_SIZE:-16384}"
        ;;
    esac

    if [[ "${NANOVLLM_CPU_OFFLOAD}" == "1" ]]; then
      append_settings_component "" "cpuoffload"
      append_settings_component "gpublocks" "${NANOVLLM_NUM_GPU_BLOCKS:-2}"
    fi
    if [[ "${NANOVLLM_ENFORCE_EAGER}" == "1" ]]; then
      append_settings_component "" "eager"
    fi
    if [[ -n "${NANOVLLM_GPU_MEMORY_UTILIZATION}" ]]; then
      append_settings_component "gpumem" "${NANOVLLM_GPU_MEMORY_UTILIZATION}"
    fi
    if [[ -n "${NANOVLLM_BLOCK_SIZE}" ]]; then
      append_settings_component "block" "${NANOVLLM_BLOCK_SIZE}"
    fi
  fi

  append_settings_component "" "${TASK_LABEL}"
  if [[ -n "${MAX_MODEL_LEN}" ]]; then
    append_settings_component "maxlen" "${MAX_MODEL_LEN}"
  fi
  if [[ -n "${DTYPE_OVERRIDE}" && "${DTYPE_OVERRIDE}" != "bfloat16" ]]; then
    append_settings_component "dtype" "${DTYPE_OVERRIDE}"
  fi

  printf '%s/%s%s' "${BASE_OUTPUT_ROOT}" "${SETTINGS_INFO}" "${MODEL_NAME}"
}

source "${SCRIPT_DIR}/config_models.sh"
source "${SCRIPT_DIR}/config_tasks.sh"

if [[ $# -gt 0 && "$1" != --* ]]; then
  MODEL_REF="$1"
  shift
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  TASK_SET="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      OUTPUT_ROOT_EXPLICIT=1
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --template-type)
      TEMPLATE_TYPE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE_OVERRIDE="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --task)
      TASK_OVERRIDE="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES_OVERRIDE="$2"
      shift 2
      ;;
    --nanovllm-cpu-offload)
      NANOVLLM_CPU_OFFLOAD=1
      shift
      ;;
    --nanovllm-num-gpu-blocks)
      NANOVLLM_NUM_GPU_BLOCKS="$2"
      shift 2
      ;;
    --nanovllm-gpu-memory-utilization)
      NANOVLLM_GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --nanovllm-block-size)
      NANOVLLM_BLOCK_SIZE="$2"
      shift 2
      ;;
    --nanovllm-enforce-eager)
      NANOVLLM_ENFORCE_EAGER=1
      shift
      ;;
    --nanovllm-sparse-policy)
      NANOVLLM_SPARSE_POLICY="$2"
      shift 2
      ;;
    --nanovllm-sparse-stride)
      NANOVLLM_SPARSE_STRIDE="$2"
      shift 2
      ;;
    --nanovllm-sparse-threshold)
      NANOVLLM_SPARSE_THRESHOLD="$2"
      shift 2
      ;;
    --nanovllm-sparse-chunk-size)
      NANOVLLM_SPARSE_CHUNK_SIZE="$2"
      shift 2
      ;;
    --nanovllm-compass-top-p)
      NANOVLLM_COMPASS_TOP_P="$2"
      shift 2
      ;;
    --nanovllm-compass-lambda)
      NANOVLLM_COMPASS_LAMBDA="$2"
      shift 2
      ;;
    --nanovllm-compass-theta)
      NANOVLLM_COMPASS_THETA="$2"
      shift 2
      ;;
    --blasst-lambda)
      BLASST_LAMBDA="$2"
      shift 2
      ;;
    --e)
      E_FLAG=(--e)
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${MODEL_REF}" && -z "${MODEL_PATH}" ]]; then
  echo "Usage: bash run.sh MODEL_NAME_OR_PATH [TASK_SET] [OPTIONS]"
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_CONFIG="$(MODEL_SELECT "${MODEL_REF}" "${MODEL_DIR}")"
  IFS=":" read -r MODEL_PATH MODEL_NAME MODEL_BACKEND TOKENIZER_PATH_DEFAULT TEMPLATE_TYPE_DEFAULT DTYPE_DEFAULT MODEL_MAX_MODEL_LEN <<< "${MODEL_CONFIG}"
  BACKEND="${BACKEND:-${MODEL_BACKEND}}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-${TOKENIZER_PATH_DEFAULT}}"
  TEMPLATE_TYPE="${TEMPLATE_TYPE:-${TEMPLATE_TYPE_DEFAULT}}"
  DTYPE_OVERRIDE="${DTYPE_OVERRIDE:-${DTYPE_DEFAULT}}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MODEL_MAX_MODEL_LEN}}"
fi

if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME="$(basename "${MODEL_PATH}")"
fi

if [[ -z "${BACKEND}" ]]; then
  BACKEND="torch"
fi

if [[ -z "${TEMPLATE_TYPE}" ]]; then
  TEMPLATE_TYPE="auto"
fi

if [[ -z "${DTYPE_OVERRIDE}" ]]; then
  DTYPE_OVERRIDE="${DTYPE}"
fi

if [[ -n "${TASK_OVERRIDE}" ]]; then
  DATASETS_ARG="${TASK_OVERRIDE}"
else
  SELECTED_TASK_SET="${TASK_SET}"
  if [[ ${#E_FLAG[@]} -gt 0 && "${SELECTED_TASK_SET}" != e_* ]]; then
    if TASKS_SELECT "e_${SELECTED_TASK_SET}" >/dev/null 2>&1; then
      SELECTED_TASK_SET="e_${SELECTED_TASK_SET}"
    fi
  fi

  if ! DATASETS_RAW="$(TASKS_SELECT "${SELECTED_TASK_SET}")"; then
    echo "Task set not supported: ${TASK_SET}"
    exit 1
  fi
  DATASETS_ARG="$(echo "${DATASETS_RAW}" | tr ' ' ',')"
fi

if [[ -n "${NUM_SAMPLES_OVERRIDE}" ]]; then
  NUM_SAMPLES="${NUM_SAMPLES_OVERRIDE}"
fi

TASK_LABEL="$(resolve_task_label)"
if [[ "${OUTPUT_ROOT_EXPLICIT}" == "0" ]]; then
  OUTPUT_ROOT="$(resolve_results_root)"
fi

PRED_CMD=(
  python "${SCRIPT_DIR}/pred.py"
  --backend "${BACKEND}"
  --model-path "${MODEL_PATH}"
  --model-name "${MODEL_NAME}"
  --template-type "${TEMPLATE_TYPE}"
  --output-root "${OUTPUT_ROOT}"
  --data-root "${DATA_ROOT}"
  --datasets "${DATASETS_ARG}"
  --num-samples "${NUM_SAMPLES}"
  --dtype "${DTYPE_OVERRIDE}"
)

if [[ -n "${TOKENIZER_PATH}" ]]; then
  PRED_CMD+=(--tokenizer-path "${TOKENIZER_PATH}")
fi

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  PRED_CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ "${NANOVLLM_CPU_OFFLOAD}" == "1" ]]; then
  PRED_CMD+=(--nanovllm-cpu-offload)
fi

if [[ -n "${NANOVLLM_NUM_GPU_BLOCKS}" ]]; then
  PRED_CMD+=(--nanovllm-num-gpu-blocks "${NANOVLLM_NUM_GPU_BLOCKS}")
fi

if [[ -n "${NANOVLLM_GPU_MEMORY_UTILIZATION}" ]]; then
  PRED_CMD+=(--nanovllm-gpu-memory-utilization "${NANOVLLM_GPU_MEMORY_UTILIZATION}")
fi

if [[ -n "${NANOVLLM_BLOCK_SIZE}" ]]; then
  PRED_CMD+=(--nanovllm-block-size "${NANOVLLM_BLOCK_SIZE}")
fi

if [[ "${NANOVLLM_ENFORCE_EAGER}" == "1" ]]; then
  PRED_CMD+=(--nanovllm-enforce-eager)
fi

if [[ -n "${NANOVLLM_SPARSE_POLICY}" ]]; then
  PRED_CMD+=(--nanovllm-sparse-policy "${NANOVLLM_SPARSE_POLICY}")
fi

if [[ -n "${NANOVLLM_SPARSE_STRIDE}" ]]; then
  PRED_CMD+=(--nanovllm-sparse-stride "${NANOVLLM_SPARSE_STRIDE}")
fi

if [[ -n "${NANOVLLM_SPARSE_THRESHOLD}" ]]; then
  PRED_CMD+=(--nanovllm-sparse-threshold "${NANOVLLM_SPARSE_THRESHOLD}")
fi

if [[ -n "${NANOVLLM_SPARSE_CHUNK_SIZE}" ]]; then
  PRED_CMD+=(--nanovllm-sparse-chunk-size "${NANOVLLM_SPARSE_CHUNK_SIZE}")
fi

if [[ -n "${NANOVLLM_COMPASS_TOP_P}" ]]; then
  PRED_CMD+=(--nanovllm-compass-top-p "${NANOVLLM_COMPASS_TOP_P}")
fi

if [[ -n "${NANOVLLM_COMPASS_LAMBDA}" ]]; then
  PRED_CMD+=(--nanovllm-compass-lambda "${NANOVLLM_COMPASS_LAMBDA}")
fi

if [[ -n "${NANOVLLM_COMPASS_THETA}" ]]; then
  PRED_CMD+=(--nanovllm-compass-theta "${NANOVLLM_COMPASS_THETA}")
fi

if [[ -n "${BLASST_LAMBDA}" ]]; then
  PRED_CMD+=(--blasst-lambda "${BLASST_LAMBDA}")
fi

if [[ ${#E_FLAG[@]} -gt 0 ]]; then
  PRED_CMD+=("${E_FLAG[@]}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PRED_CMD+=("${EXTRA_ARGS[@]}")
fi

EVAL_CMD=(
  python "${SCRIPT_DIR}/eval.py"
  --output-root "${OUTPUT_ROOT}"
  --model-name "${MODEL_NAME}"
  --datasets "${DATASETS_ARG}"
)

if [[ ${#E_FLAG[@]} -gt 0 ]]; then
  EVAL_CMD+=("${E_FLAG[@]}")
fi

echo "========================================"
echo "LongBench Script Runner"
echo "========================================"
echo "Model Ref:     ${MODEL_REF:-${MODEL_PATH}}"
echo "Model Path:    ${MODEL_PATH}"
echo "Model Name:    ${MODEL_NAME}"
echo "Task Set:      ${TASK_SET}"
echo "Datasets:      ${DATASETS_ARG}"
echo "Backend:       ${BACKEND}"
echo "Template Type: ${TEMPLATE_TYPE}"
echo "Data Root:     ${DATA_ROOT}"
echo "Task Label:    ${TASK_LABEL}"
echo "Output Root:   ${OUTPUT_ROOT}"
echo "Num Samples:   ${NUM_SAMPLES}"
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  echo "Max Model Len: ${MAX_MODEL_LEN}"
fi
echo "========================================"

"${PRED_CMD[@]}"
"${EVAL_CMD[@]}"
