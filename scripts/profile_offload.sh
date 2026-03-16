#!/bin/bash

# Profile test_ruler.py using NVIDIA Nsight Systems
#
# Usage:
#   bash scripts/profile_offload.sh [options]
#
# Options:
#   --policy POLICY      Sparse policy name (default: full)
#   --ctx-len LENGTH     Context length: 32k, 64k, 128k (default: 64k)
#   --dataset DATASET    Task name (default: niah_single_1)
#   --sample INDEX       Sample index (default: 0)
#   --gpu GPU_ID         GPU to use (default: 0)
#   --num-gpu-blocks N   Number of GPU blocks/slots (default: 4)
#   --block-size SIZE    KV cache block size (default: 4096)
#   --no-offload         Disable CPU offload
#
# Output:
#   results/nsys/<policy>_<gpuonly|offload>_<ctx-len>_blk<size>_<timestamp>.nsys-rep
#
# Examples:
#   bash scripts/profile_offload.sh
#   bash scripts/profile_offload.sh --policy xattn --ctx-len 128k --no-offload
#   bash scripts/profile_offload.sh --policy full --ctx-len 32k --num-gpu-blocks 8

# Default configuration
POLICY="full"
CTX_LEN="64k"
DATASET="niah_single_1"
SAMPLE_INDEX="0"
GPU_ID="0"
NUM_GPU_BLOCKS="4"
BLOCK_SIZE="4096"
GPU_UTIL="0.9"
ENABLE_OFFLOAD="--enable-offload"
MAX_NEW_TOKENS="1"
MODEL=""
DATA_DIR_OVERRIDE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --ctx-len)
            CTX_LEN="$2"
            shift 2
            ;;
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
        --gpu-util)
            GPU_UTIL="$2"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --policy POLICY      Sparse policy name (default: full)"
            echo "  --ctx-len LENGTH     Context length: 32k, 64k, 128k (default: 64k)"
            echo "  --block-size SIZE    KV cache block size (default: 4096)"
            echo "  --dataset DATASET    Task name (default: niah_single_1)"
            echo "  --sample INDEX       Sample index (default: 0)"
            echo "  --gpu GPU_ID         GPU to use (default: 0)"
            echo "  --gpu-util UTIL      GPU memory utilization (default: 0.9)"
            echo "  --max-new-tokens N   Max new tokens to generate (default: 1)"
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
DATA_DIR="$PROJECT_ROOT/tests/data/ruler_${CTX_LEN}"

# Set max-model-len based on context length
case "$CTX_LEN" in
    32k)
        MAX_MODEL_LEN=36000
        ;;
    64k)
        MAX_MODEL_LEN=72000
        ;;
    128k)
        MAX_MODEL_LEN=144000
        ;;
    256k)
        MAX_MODEL_LEN=288000
        ;;
    512k)
        MAX_MODEL_LEN=576000
        ;;
    768k)
        MAX_MODEL_LEN=850000
        ;;
    1m)
        MAX_MODEL_LEN=1100000
        ;;
    *)
        MAX_MODEL_LEN=72000
        ;;
esac

# Override DATA_DIR if specified
if [ -n "$DATA_DIR_OVERRIDE" ]; then
    DATA_DIR="$DATA_DIR_OVERRIDE"
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for unique filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$ENABLE_OFFLOAD" ]; then
    OFFLOAD_TAG="offload"
else
    OFFLOAD_TAG="gpuonly"
fi
OUTPUT_FILE="$OUTPUT_DIR/${POLICY}_${OFFLOAD_TAG}_${CTX_LEN}_blk${BLOCK_SIZE}_${TIMESTAMP}"

echo "============================================================"
echo "NVIDIA Nsight Systems Profiling"
echo "============================================================"
echo "Policy:      $POLICY"
echo "Offload:     $OFFLOAD_TAG"
echo "Context:     $CTX_LEN"
echo "Block Size:  $BLOCK_SIZE"
echo "Dataset:     $DATASET"
echo "Sample:      $SAMPLE_INDEX"
echo "GPU:         $GPU_ID"
echo "GPU Blocks:  $NUM_GPU_BLOCKS"
echo "Data Dir:    $DATA_DIR"
echo "Max Tokens:  $MAX_NEW_TOKENS"
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

# Map policy name to internal enum name
# User-friendly name -> SparsePolicyType enum name
case "$POLICY" in
    xattn)
        POLICY_ENUM="XATTN_BSA"
        ;;
    *)
        POLICY_ENUM="$POLICY"
        ;;
esac

# Build sparse policy argument
SPARSE_POLICY_ARG=""
if [ -n "$POLICY_ENUM" ] && [ "$POLICY_ENUM" != "full" ]; then
    SPARSE_POLICY_ARG="--sparse-policy $POLICY_ENUM"
fi

# Build model argument
MODEL_ARG=""
if [ -n "$MODEL" ]; then
    MODEL_ARG="--model $MODEL"
fi

# Run nsys profile and capture exit code
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    --output="$OUTPUT_FILE" \
    python "$TEST_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --datasets "$DATASET" \
        --sample-indices "$SAMPLE_INDEX" \
        --num-gpu-blocks "$NUM_GPU_BLOCKS" \
        --block-size "$BLOCK_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --gpu-utilization "$GPU_UTIL" \
        $ENABLE_OFFLOAD \
        $SPARSE_POLICY_ARG \
        $MODEL_ARG \
        --quiet
EXIT_CODE=$?

# If test failed, print warning but keep results
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "Test FAILED (Expected if limiting tokens)! Keeping results..."
    echo "============================================================"
fi

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
