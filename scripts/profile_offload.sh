#!/bin/bash

# Profile test_ruler.py using NVIDIA Nsight Systems
#
# Usage:
#   bash scripts/profile_offload.sh [options]
#
# Options:
#   --dataset DATASET    Task name (default: niah_single_1)
#   --sample INDEX       Sample index (default: 0)
#   --gpu GPU_ID         GPU to use (default: 0)
#   --num-gpu-blocks N   Number of GPU blocks/slots (default: 4)
#   --no-offload         Disable CPU offload
#
# Output:
#   results/nsys/ruler_<dataset>_sample<index>_<timestamp>.nsys-rep
#
# Examples:
#   bash scripts/profile_offload.sh
#   bash scripts/profile_offload.sh --dataset niah_single_1 --sample 5
#   bash scripts/profile_offload.sh --gpu 1 --no-offload
#   bash scripts/profile_offload.sh --num-gpu-blocks 8

set -e

# Default configuration
DATASET="niah_single_1"
SAMPLE_INDEX="0"
GPU_ID="0"
NUM_GPU_BLOCKS="4"
ENABLE_OFFLOAD="--enable-offload"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --sample)
            SAMPLE_INDEX="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --no-offload)
            ENABLE_OFFLOAD=""
            shift
            ;;
        --num-gpu-blocks)
            NUM_GPU_BLOCKS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Task name (default: niah_single_1)"
            echo "  --sample INDEX       Sample index (default: 0)"
            echo "  --gpu GPU_ID         GPU to use (default: 0)"
            echo "  --no-offload         Disable CPU offload"
            echo "  --num-gpu-blocks N   Number of GPU blocks/slots (default: 4)"
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
TEST_SCRIPT="$PROJECT_ROOT/tests/test_ruler.py"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for unique filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OFFLOAD_SUFFIX=""
if [ -n "$ENABLE_OFFLOAD" ]; then
    OFFLOAD_SUFFIX="_offload_${NUM_GPU_BLOCKS}slots"
fi
OUTPUT_FILE="$OUTPUT_DIR/ruler_${DATASET}_sample${SAMPLE_INDEX}${OFFLOAD_SUFFIX}_${TIMESTAMP}"

echo "============================================================"
echo "NVIDIA Nsight Systems Profiling"
echo "============================================================"
echo "Test script: $TEST_SCRIPT"
echo "Dataset:     $DATASET"
echo "Sample:      $SAMPLE_INDEX"
echo "GPU:         $GPU_ID"
echo "GPU Blocks:  $NUM_GPU_BLOCKS"
echo "Offload:     ${ENABLE_OFFLOAD:-disabled}"
echo "Output file: $OUTPUT_FILE.nsys-rep"
echo ""

# nsys profile options:
# --trace=cuda,nvtx,osrt,cudnn,cublas : Trace CUDA API, NVTX markers, OS runtime, cuDNN, cuBLAS
# --cuda-memory-usage=true            : Track CUDA memory allocations
# --stats=true                        : Generate summary statistics
# --force-overwrite=true              : Overwrite existing output file
# --output=<path>                     : Output file path (without .nsys-rep extension)

echo "Running nsys profile..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --output="$OUTPUT_FILE" \
    python "$TEST_SCRIPT" \
        --datasets "$DATASET" \
        --sample-indices "$SAMPLE_INDEX" \
        --num-gpu-blocks "$NUM_GPU_BLOCKS" \
        $ENABLE_OFFLOAD \
        --quiet

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
