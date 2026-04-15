#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/3rdparty/nanovllm:${PYTHONPATH:-}"

ORIG_ARGS=("$@")
MODEL_PATH=""
MODEL_NAME=""
OUTPUT_ROOT="${SCRIPT_DIR}/../benchmark_root"
E_FLAG=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --e)
      E_FLAG=(--e)
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Missing required --model-path"
  exit 1
fi
if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME="$(basename "${MODEL_PATH}")"
fi

python "${SCRIPT_DIR}/pred.py" "${ORIG_ARGS[@]}" --output-root "${OUTPUT_ROOT}"
python "${SCRIPT_DIR}/eval.py" --output-root "${OUTPUT_ROOT}" --model-name "${MODEL_NAME}" "${E_FLAG[@]}"
