#!/bin/bash
# RULER Benchmark Runner
# Usage: ./scripts/run_ruler.sh [MODEL_NAME] [BENCHMARK] [METRIC] [OPTIONS]
#
# Examples:
#   ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic xattn --stride 8
#   ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic xattn --stride 16 --task niah_single_1
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic xattn --stride 8
#
# Options:
#   --stride N     Set XAttention stride (for nanovllm xattn)
#   --task TASK    Run specific task(s) only
#   --threshold N  Set XAttention threshold
#
# Environment Variables for nanovllm:
#   NANOVLLM_SPARSE_STRIDE         XAttention stride (default: 8)
#   NANOVLLM_SPARSE_THRESHOLD      XAttention threshold (default: 0.9)
#   NANOVLLM_SPARSE_CHUNK_SIZE     XAttention chunk size (default: 16384)
#   NANOVLLM_BLASST_A              BLASST inverse formula numerator (default: 16384)
#   NANOVLLM_BLASST_FIXED_LAMBDA   BLASST fixed threshold (default: 0.5)
#   NANOVLLM_BLASST_GRANULARITY    BLASST token granularity (default: 128)
#   NANOVLLM_CPU_OFFLOAD           Enable CPU offload: true/false (default: true)
#   NANOVLLM_NUM_GPU_BLOCKS        Number of GPU blocks for offload (default: 2)
#   NANOVLLM_BLOCK_SIZE            KV cache block size (default: 4096)
#   NANOVLLM_DTYPE                 Data type: float16, bfloat16, float32 (default: bfloat16)
#
# Examples with different metrics:
#   ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic full
#   ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic xattn
#   ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic blasst
#   NANOVLLM_BLASST_FIXED_LAMBDA=0.3 ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic blasst

set -e

#############################################
# Configuration
#############################################

# Model settings
MODEL_NAME="${1:-llama3.1-8b-chat}"
BENCHMARK="${2:-synthetic}"
METRIC="${3:-full}"  # Options: full, xattn, avgpool, compass, minfer, flex, blasst

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

# GPU configuration (support both GPULIST and GPU_LIST)
export GPU_LIST="${GPULIST:-${GPU_LIST:-0,1,2,3}}"

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
echo "GPU List:     $GPU_LIST"
# Extract stride from EXTRA_ARGS for display
if [[ "$EXTRA_ARGS" == *"--stride"* ]]; then
    STRIDE_VALUE=$(echo "$EXTRA_ARGS" | grep -o -- '--stride [0-9]*' | awk '{print $2}')
    echo "Stride:       $STRIDE_VALUE"
fi
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
