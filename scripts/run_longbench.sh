#!/bin/bash
# LongBench Benchmark Runner
# Usage: ./scripts/run_longbench.sh [MODEL_PATH] [BACKEND] [OPTIONS]
#
# Examples:
#   ./scripts/run_longbench.sh /path/to/model torch --template-type auto
#   ./scripts/run_longbench.sh /path/to/model nanovllm --datasets passage_count --num-samples 1
#   ./scripts/run_longbench.sh /path/to/model torch --data-root ~/data/LongBench --e
#
# Notes:
#   - Defaults to backend=torch
#   - Defaults to CUDA_VISIBLE_DEVICES=0 unless explicitly overridden
#   - Additional options are passed through to eval/LongBench/scripts/run.sh
#   - --data-root lets you choose the LongBench dataset location explicitly

set -e

#############################################
# Configuration
#############################################

MODEL_PATH="${1:-${MODEL_PATH:-}}"
BACKEND="${2:-torch}"  # Options: torch, nanovllm
DATA_ROOT="${LONG_BENCH_DATA_ROOT:-$HOME/data/LongBench}"

if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: ./scripts/run_longbench.sh [MODEL_PATH] [BACKEND] [OPTIONS]"
    echo
    echo "Examples:"
    echo "  ./scripts/run_longbench.sh /path/to/model torch --template-type auto"
    echo "  ./scripts/run_longbench.sh /path/to/model nanovllm --datasets passage_count --num-samples 1"
    echo "  ./scripts/run_longbench.sh /path/to/model torch --data-root ~/data/LongBench --e"
    exit 1
fi

# Parse additional arguments
EXTRA_ARGS=()
if [ $# -gt 2 ]; then
    shift 2
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
fi

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
echo "Model Path:   $MODEL_PATH"
echo "Backend:      $BACKEND"
echo "CUDA Visible: $CUDA_VISIBLE_DEVICES"
echo "Data Root:    $LONG_BENCH_DATA_ROOT"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra Args:   ${EXTRA_ARGS[*]}"
fi
echo "========================================"

cd "${PROJECT_DIR}/eval/LongBench/scripts"
bash ./run.sh --model-path "$MODEL_PATH" --backend "$BACKEND" --data-root "$LONG_BENCH_DATA_ROOT" "${EXTRA_ARGS[@]}"
