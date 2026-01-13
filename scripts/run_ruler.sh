#!/bin/bash
# RULER Benchmark Runner
# Usage: ./scripts/run_ruler.sh [MODEL_NAME] [BENCHMARK] [METRIC] [--task TASK]
#
# Examples:
#   ./scripts/run_ruler.sh llama3.1-8b-chat synthetic full
#   ./scripts/run_ruler.sh llama3.1-8b-chat synthetic full --task niah_single_1
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_ruler.sh llama3.1-8b-chat synthetic xattn

set -e

#############################################
# Configuration
#############################################

# Model settings
MODEL_NAME="${1:-llama3.1-8b-chat}"
BENCHMARK="${2:-synthetic}"
METRIC="${3:-full}"  # Options: full, xattn, avgpool, compass, minfer, flex

# Parse additional arguments (--task)
EXTRA_ARGS=""
if [ $# -gt 3 ]; then
    shift 3
    EXTRA_ARGS="$@"
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
MODEL_DIR="${MODEL_DIR:-/home/zijie/models}"

# Set PYTHONPATH (nanovllm from 3rdparty takes priority over system version)
export PYTHONPATH="${PROJECT_DIR}/3rdparty/nanovllm:${PROJECT_DIR}:${PYTHONPATH}"

#############################################
# Run benchmark (use current environment)
#############################################

echo "========================================"
echo "RULER Benchmark"
echo "========================================"
echo "Project:      $PROJECT_DIR"
echo "Model Dir:    $MODEL_DIR"
echo "Model:        $MODEL_NAME"
echo "Benchmark:    $BENCHMARK"
echo "Metric:       $METRIC"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra Args:   $EXTRA_ARGS"
fi
echo "========================================"

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# Download datasets if needed (auto-skip if exist)
cd "${PROJECT_DIR}/eval/RULER"
bash setup.sh

# Run RULER benchmark
cd "${PROJECT_DIR}/eval/RULER/scripts"
./run.sh "$MODEL_NAME" "$BENCHMARK" --metric "$METRIC" $EXTRA_ARGS
