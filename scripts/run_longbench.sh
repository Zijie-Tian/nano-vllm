#!/bin/bash
# LongBench Benchmark Runner
# Usage: ./scripts/run_longbench.sh [MODEL_NAME_OR_PATH] [TASK_SET] [BACKEND] [OPTIONS]
#
# Examples:
#   ./scripts/run_longbench.sh llama3.1-8b-instruct all torch
#   ./scripts/run_longbench.sh llama3.1-8b-instruct triattention torch --num-samples 1
#   ./scripts/run_longbench.sh /path/to/model triattention torch --data-root ~/data/LongBench
#
# Notes:
#   - Defaults to backend=torch
#   - Defaults to task set=all
#   - Defaults to CUDA_VISIBLE_DEVICES=0 unless explicitly overridden
#   - Additional options are passed through to eval/LongBench/scripts/run.sh
#   - --data-root lets you choose the LongBench dataset location explicitly

set -e

#############################################
# Configuration
#############################################

MODEL_REF="${1:-${MODEL_PATH:-}}"
TASK_SET="all"
BACKEND="torch"  # Options: torch, nanovllm
DATA_ROOT="${LONG_BENCH_DATA_ROOT:-$HOME/data/LongBench}"

if [ -z "${MODEL_REF}" ]; then
    echo "Usage: ./scripts/run_longbench.sh [MODEL_NAME_OR_PATH] [TASK_SET] [BACKEND] [OPTIONS]"
    echo
    echo "Examples:"
    echo "  ./scripts/run_longbench.sh llama3.1-8b-instruct all torch"
    echo "  ./scripts/run_longbench.sh llama3.1-8b-instruct triattention torch --num-samples 1"
    echo "  ./scripts/run_longbench.sh /path/to/model triattention torch --data-root ~/data/LongBench"
    exit 1
fi

# Backward compatibility:
# - if arg2 is a backend, keep old [MODEL_PATH] [BACKEND] shape and default task set to all
# - otherwise treat arg2 as task set and arg3 as optional backend
EXTRA_ARGS=()
if [ $# -ge 2 ]; then
    if [[ "$2" == "torch" || "$2" == "nanovllm" ]]; then
        BACKEND="$2"
        shift 2
    else
        TASK_SET="$2"
        if [ $# -ge 3 ] && [[ "$3" == "torch" || "$3" == "nanovllm" ]]; then
            BACKEND="$3"
            shift 3
        else
            shift 2
        fi
    fi
else
    shift 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# GPU configuration: user explicitly requested GPU0-only by default
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LONG_BENCH_DATA_ROOT="${DATA_ROOT}"

# Set PYTHONPATH (nanovllm from 3rdparty takes priority over system version)
export PYTHONPATH="${PROJECT_DIR}/3rdparty/nanovllm:${PROJECT_DIR}:${PYTHONPATH}"

#############################################
# Run benchmark
#############################################

echo "========================================"
echo "LongBench Benchmark"
echo "========================================"
echo "Project:      $PROJECT_DIR"
echo "Model Ref:    $MODEL_REF"
echo "Task Set:     $TASK_SET"
echo "Backend:      $BACKEND"
echo "CUDA Visible: $CUDA_VISIBLE_DEVICES"
echo "Data Root:    $LONG_BENCH_DATA_ROOT"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra Args:   ${EXTRA_ARGS[*]}"
fi
echo "========================================"

cd "${PROJECT_DIR}/eval/LongBench/scripts"
bash ./run.sh "$MODEL_REF" "$TASK_SET" --backend "$BACKEND" --data-root "$LONG_BENCH_DATA_ROOT" "${EXTRA_ARGS[@]}"
