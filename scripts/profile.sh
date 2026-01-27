#!/bin/bash

# Profile bench.py using NVIDIA Nsight Systems (GPU-only mode)
#
# Usage:
#   bash scripts/profile.sh [options]
#
# Options:
#   --max-len LENGTH     Max sequence length (default: 32768)
#   --policy POLICY      Sparse policy: full, xattn (default: xattn)
#   --gpu GPU_ID         GPU to use (default: 0)
#   --gpu-util UTIL      GPU memory utilization (default: 0.9)
#   --input-len LENGTH   Input length (default: max-len - 1)
#   --bench-decode       Run decode benchmark instead of prefill
#
# Output:
#   results/nsys/bench_<policy>_<max_len>_<timestamp>.nsys-rep
#
# Examples:
#   bash scripts/profile.sh
#   bash scripts/profile.sh --max-len 65536 --gpu-util 0.7
#   bash scripts/profile.sh --policy full --max-len 32768
#   bash scripts/profile.sh --bench-decode

set -e

# Default configuration
MAX_LEN="32768"
POLICY="xattn"
GPU_ID="0"
GPU_UTIL="0.9"
INPUT_LEN=""
BENCH_MODE="prefill"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --gpu-util)
            GPU_UTIL="$2"
            shift 2
            ;;
        --input-len)
            INPUT_LEN="$2"
            shift 2
            ;;
        --bench-decode)
            BENCH_MODE="decode"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --max-len LENGTH     Max sequence length (default: 32768)"
            echo "  --policy POLICY      Sparse policy: full, xattn (default: xattn)"
            echo "  --gpu GPU_ID         GPU to use (default: 0)"
            echo "  --gpu-util UTIL      GPU memory utilization (default: 0.9)"
            echo "  --input-len LENGTH   Input length (default: max-len - 1)"
            echo "  --bench-decode       Run decode benchmark instead of prefill"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/results/nsys"
BENCH_SCRIPT="$PROJECT_ROOT/bench.py"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for unique filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Convert max_len to human-readable format (e.g., 32768 -> 32k)
if [ "$MAX_LEN" -ge 1024 ]; then
    MAX_LEN_SUFFIX="$((MAX_LEN / 1024))k"
else
    MAX_LEN_SUFFIX="${MAX_LEN}"
fi

OUTPUT_FILE="$OUTPUT_DIR/bench_${POLICY}_${MAX_LEN_SUFFIX}_${BENCH_MODE}_${TIMESTAMP}"

# Build bench.py arguments
BENCH_ARGS="--max-len $MAX_LEN --gpu-util $GPU_UTIL"

if [ -n "$POLICY" ]; then
    BENCH_ARGS="$BENCH_ARGS --policy $POLICY"
fi

if [ -n "$INPUT_LEN" ]; then
    BENCH_ARGS="$BENCH_ARGS --input-len $INPUT_LEN"
fi

if [ "$BENCH_MODE" = "decode" ]; then
    BENCH_ARGS="$BENCH_ARGS --bench-decode"
fi

echo "============================================================"
echo "NVIDIA Nsight Systems Profiling (GPU-only)"
echo "============================================================"
echo "Bench script: $BENCH_SCRIPT"
echo "Policy:       $POLICY"
echo "Max length:   $MAX_LEN"
echo "GPU:          $GPU_ID"
echo "GPU util:     $GPU_UTIL"
echo "Bench mode:   $BENCH_MODE"
echo "Output file:  $OUTPUT_FILE.nsys-rep"
echo ""

# nsys profile options:
# --trace=cuda,nvtx : Trace CUDA API and NVTX markers
# --force-overwrite=true : Overwrite existing output file
# --output=<path> : Output file path (without .nsys-rep extension)

echo "Running nsys profile..."
echo "Command: python bench.py $BENCH_ARGS"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --output="$OUTPUT_FILE" \
    python "$BENCH_SCRIPT" $BENCH_ARGS

echo ""
echo "============================================================"
echo "Profiling completed successfully!"
echo "============================================================"
echo "Output file: $OUTPUT_FILE.nsys-rep"
echo ""
echo "To view results in GUI:"
echo "  nsight-sys $OUTPUT_FILE.nsys-rep"
echo ""
echo "To export statistics:"
echo "  nsys stats --report cuda_api_sum $OUTPUT_FILE.nsys-rep"
echo "  nsys stats --report cuda_gpu_kern_sum $OUTPUT_FILE.nsys-rep"
echo "  nsys stats --report cuda_gpu_mem_size_sum $OUTPUT_FILE.nsys-rep"
echo "============================================================"
