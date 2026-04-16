#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/3rdparty/nanovllm:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:-/home/zijie/models}"
OUTPUT_ROOT="${SCRIPT_DIR}/../benchmark_root"
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
COMPRESSION_METHOD=""
TRIATTENTION_STATS_PATH="${TRIATTENTION_STATS_PATH:-}"
TRIATTENTION_BUDGET="${TRIATTENTION_BUDGET:-}"
E_FLAG=()
TASK_OVERRIDE=""
NUM_SAMPLES_OVERRIDE=""
EXTRA_ARGS=()

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
    --compression-method)
      COMPRESSION_METHOD="$2"
      shift 2
      ;;
    --triattention-stats-path)
      TRIATTENTION_STATS_PATH="$2"
      shift 2
      ;;
    --triattention-budget)
      TRIATTENTION_BUDGET="$2"
      shift 2
      ;;
    --triattention-frequency-window)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --triattention-score-aggregation)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --triattention-divide-length)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --triattention-disable-mlr|--triattention-disable-trig)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --task)
      TASK_OVERRIDE="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES_OVERRIDE="$2"
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
  IFS=":" read -r MODEL_PATH MODEL_NAME MODEL_BACKEND TOKENIZER_PATH_DEFAULT TEMPLATE_TYPE_DEFAULT DTYPE_DEFAULT MODEL_MAX_MODEL_LEN MODEL_COMPRESSION_METHOD MODEL_TRIATTENTION_STATS MODEL_TRIATTENTION_BUDGET <<< "${MODEL_CONFIG}"
  BACKEND="${BACKEND:-${MODEL_BACKEND}}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-${TOKENIZER_PATH_DEFAULT}}"
  TEMPLATE_TYPE="${TEMPLATE_TYPE:-${TEMPLATE_TYPE_DEFAULT}}"
  DTYPE_OVERRIDE="${DTYPE_OVERRIDE:-${DTYPE_DEFAULT}}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MODEL_MAX_MODEL_LEN}}"
  COMPRESSION_METHOD="${COMPRESSION_METHOD:-${MODEL_COMPRESSION_METHOD}}"
  TRIATTENTION_STATS_PATH="${TRIATTENTION_STATS_PATH:-${MODEL_TRIATTENTION_STATS}}"
  TRIATTENTION_BUDGET="${TRIATTENTION_BUDGET:-${MODEL_TRIATTENTION_BUDGET}}"
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

if [[ -n "${COMPRESSION_METHOD}" ]]; then
  PRED_CMD+=(--compression-method "${COMPRESSION_METHOD}")
fi

if [[ -n "${TRIATTENTION_STATS_PATH}" ]]; then
  PRED_CMD+=(--triattention-stats-path "${TRIATTENTION_STATS_PATH}")
fi

if [[ -n "${TRIATTENTION_BUDGET}" ]]; then
  PRED_CMD+=(--triattention-budget "${TRIATTENTION_BUDGET}")
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
if [[ -n "${COMPRESSION_METHOD}" ]]; then
  echo "Compression:   ${COMPRESSION_METHOD}"
fi
echo "Template Type: ${TEMPLATE_TYPE}"
echo "Data Root:     ${DATA_ROOT}"
echo "Output Root:   ${OUTPUT_ROOT}"
echo "Num Samples:   ${NUM_SAMPLES}"
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  echo "Max Model Len: ${MAX_MODEL_LEN}"
fi
echo "========================================"

"${PRED_CMD[@]}"
"${EVAL_CMD[@]}"
