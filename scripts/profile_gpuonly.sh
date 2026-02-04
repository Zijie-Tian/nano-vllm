#!/bin/bash

# Profile GPU-only XAttention using NVIDIA Nsight Systems
#
# Usage:
#   bash scripts/profile_gpuonly.sh [options]
#
# Options:
#   --model PATH         Model path (default: ~/models/Llama-3.2-1B-Instruct)
#   --max-len LENGTH     Max sequence length (default: 135000 for 128K)
#   --input-len LENGTH   Input length (default: max-len - 100)
#   --policy POLICY      Sparse policy: full, xattn (default: xattn)
#   --gpu GPU_ID         GPU to use (default: 0)
#   --gpu-util UTIL      GPU memory utilization (default: 0.5)
#   --no-stats           Skip generating stats reports
#
# Output:
#   results/nsys/gpuonly_<policy>_<max_len>_<timestamp>.nsys-rep
#
# NVTX Markers captured:
#   - xattn_estimate: XAttention importance estimation
#   - xattn_bsa_compute: Block sparse attention computation
#
# Examples:
#   bash scripts/profile_gpuonly.sh
#   bash scripts/profile_gpuonly.sh --max-len 65536 --gpu-util 0.7
#   bash scripts/profile_gpuonly.sh --model ~/models/Llama-3.1-8B-Instruct --gpu-util 0.7

set -e

# Default configuration
MODEL="$HOME/models/Llama-3.2-1B-Instruct"
MAX_LEN="135000"
INPUT_LEN=""
POLICY="xattn"
GPU_ID="0"
GPU_UTIL="0.5"
GENERATE_STATS="true"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        --input-len)
            INPUT_LEN="$2"
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
        --no-stats)
            GENERATE_STATS="false"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH         Model path (default: ~/models/Llama-3.2-1B-Instruct)"
            echo "  --max-len LENGTH     Max sequence length (default: 135000)"
            echo "  --input-len LENGTH   Input length (default: max-len - 100)"
            echo "  --policy POLICY      Sparse policy: full, xattn (default: xattn)"
            echo "  --gpu GPU_ID         GPU to use (default: 0)"
            echo "  --gpu-util UTIL      GPU memory utilization (default: 0.5)"
            echo "  --no-stats           Skip generating stats reports"
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

# Convert max_len to human-readable format (e.g., 135000 -> 128k)
if [ "$MAX_LEN" -ge 1024 ]; then
    MAX_LEN_SUFFIX="$((MAX_LEN / 1024))k"
else
    MAX_LEN_SUFFIX="${MAX_LEN}"
fi

# Extract model name from path
MODEL_NAME=$(basename "$MODEL")

OUTPUT_FILE="$OUTPUT_DIR/gpuonly_${POLICY}_${MAX_LEN_SUFFIX}_${MODEL_NAME}_${TIMESTAMP}"

# Calculate input length if not specified
if [ -z "$INPUT_LEN" ]; then
    INPUT_LEN=$((MAX_LEN - 100))
fi

# Build bench.py arguments
BENCH_ARGS="--model $MODEL --max-len $MAX_LEN --input-len $INPUT_LEN --gpu-util $GPU_UTIL --policy $POLICY"

echo "============================================================"
echo "NVIDIA Nsight Systems Profiling (GPU-only XAttention)"
echo "============================================================"
echo "Model:        $MODEL"
echo "Policy:       $POLICY"
echo "Max length:   $MAX_LEN"
echo "Input length: $INPUT_LEN"
echo "GPU:          $GPU_ID"
echo "GPU util:     $GPU_UTIL"
echo "Output file:  $OUTPUT_FILE.nsys-rep"
echo ""
echo "NVTX markers to capture:"
echo "  - xattn_estimate: Importance estimation phase"
echo "  - xattn_bsa_compute: BSA computation phase"
echo ""

# nsys profile options:
# --trace=cuda,nvtx : Trace CUDA API and NVTX markers
# --force-overwrite=true : Overwrite existing output file
# --output=<path> : Output file path (without .nsys-rep extension)
# --capture-range=cudaProfilerApi : Only capture between cudaProfilerStart/Stop

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

# Generate statistics reports
if [ "$GENERATE_STATS" = "true" ]; then
    echo "Generating statistics reports..."
    echo ""

    # NVTX Summary (shows xattn_estimate vs xattn_bsa_compute breakdown)
    echo "=== NVTX Range Summary (Estimate vs BSA Compute) ==="
    nsys stats --report nvtx_pushpop_sum "$OUTPUT_FILE.nsys-rep" 2>/dev/null || echo "(NVTX report not available)"
    echo ""

    # GPU Kernel Summary
    echo "=== GPU Kernel Summary ==="
    nsys stats --report cuda_gpu_kern_sum "$OUTPUT_FILE.nsys-rep" 2>/dev/null | head -50
    echo ""

    # CUDA API Summary
    echo "=== CUDA API Summary ==="
    nsys stats --report cuda_api_sum "$OUTPUT_FILE.nsys-rep" 2>/dev/null | head -30
    echo ""
fi

echo "============================================================"
echo "To view results in GUI:"
echo "  nsight-sys $OUTPUT_FILE.nsys-rep"
echo ""
echo "To export additional statistics:"
echo "  nsys stats --report nvtx_pushpop_sum $OUTPUT_FILE.nsys-rep"
echo "  nsys stats --report cuda_gpu_kern_sum $OUTPUT_FILE.nsys-rep"
echo "  nsys stats --report cuda_api_sum $OUTPUT_FILE.nsys-rep"
echo "============================================================"
