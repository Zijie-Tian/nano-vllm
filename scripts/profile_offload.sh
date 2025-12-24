#!/bin/bash

# Profile test_attention_offload.py using NVIDIA Nsight Systems
#
# Usage:
#   bash scripts/profile_offload.sh
#
# Output:
#   results/nsys/attention_offload_<timestamp>.nsys-rep
#
# View results:
#   nsight-sys results/nsys/attention_offload_<timestamp>.nsys-rep

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/results/nsys"
TEST_SCRIPT="$PROJECT_ROOT/tests/test_attention_offload.py"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for unique filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/attention_offload_$TIMESTAMP"

echo "============================================================"
echo "NVIDIA Nsight Systems Profiling"
echo "============================================================"
echo "Test script: $TEST_SCRIPT"
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

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    --output="$OUTPUT_FILE" \
    python "$TEST_SCRIPT"

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
