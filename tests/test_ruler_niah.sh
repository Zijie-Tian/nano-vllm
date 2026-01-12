#!/bin/bash
#
# RULER NIAH Parallel Test Script
#
# Runs RULER NIAH benchmark across multiple GPUs in parallel.
# Each sample is tested independently (separate Python process per sample).
#
# Usage:
#   ./tests/test_ruler_niah.sh [OPTIONS]
#
# Options:
#   --gpus "0,1,2,3"     GPUs to use (default: "0,1,2,3")
#   --total N            Total samples to test (default: 100)
#   --model PATH         Model path (default: ~/models/Llama-3.1-8B-Instruct)
#   --output FILE        Output log file (default: /tmp/ruler_niah_results.log)
#

# Note: Removed 'set -e' because ((var++)) returns 1 when var=0, which triggers exit

# Default configuration
GPUS="0,1,2,3"
TOTAL_SAMPLES=100
MODEL_PATH="$HOME/models/Llama-3.1-8B-Instruct"
OUTPUT_LOG="/tmp/ruler_niah_results.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --total)
            TOTAL_SAMPLES="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_LOG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert GPU string to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "RULER NIAH Parallel Test"
echo "============================================================"
echo "GPUs: ${GPUS} (${NUM_GPUS} GPUs)"
echo "Total samples: ${TOTAL_SAMPLES}"
echo "Model: ${MODEL_PATH}"
echo "Output log: ${OUTPUT_LOG}"
echo "Project root: ${PROJECT_ROOT}"
echo "============================================================"
echo ""

# Create output directory
mkdir -p "$(dirname "$OUTPUT_LOG")"

# Initialize result tracking
RESULT_DIR="/tmp/ruler_niah_results_$$"
mkdir -p "$RESULT_DIR"

# Function to run a single sample on a specific GPU
run_sample() {
    local gpu=$1
    local sample_idx=$2
    local result_file="$RESULT_DIR/sample_${sample_idx}.result"

    # Run test with unique port based on GPU
    local port=$((2333 + gpu))

    NANOVLLM_DIST_PORT=$port \
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
    python "$SCRIPT_DIR/test_ruler_niah.py" \
        --model "$MODEL_PATH" \
        --enable-offload \
        --sample-indices "$sample_idx" \
        --quiet \
        2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "PASS" > "$result_file"
    else
        echo "FAIL" > "$result_file"
    fi

    return $exit_code
}

# Function to run samples on a specific GPU
run_gpu_worker() {
    local gpu=$1
    local gpu_idx=$2
    local log_file="$RESULT_DIR/gpu_${gpu}.log"

    echo "[GPU $gpu] Starting worker (gpu_idx=$gpu_idx)" | tee -a "$log_file"

    # Calculate which samples this GPU handles
    local sample_idx=$gpu_idx
    local pass_count=0
    local fail_count=0

    while [ $sample_idx -lt $TOTAL_SAMPLES ]; do
        echo "[GPU $gpu] Testing sample $sample_idx..." | tee -a "$log_file"

        local start_time=$(date +%s)

        if run_sample $gpu $sample_idx >> "$log_file" 2>&1; then
            echo "[GPU $gpu] Sample $sample_idx: PASS" | tee -a "$log_file"
            ((pass_count++))
        else
            echo "[GPU $gpu] Sample $sample_idx: FAIL" | tee -a "$log_file"
            ((fail_count++))
        fi

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "[GPU $gpu] Sample $sample_idx completed in ${duration}s" | tee -a "$log_file"

        # Move to next sample for this GPU (stride by number of GPUs)
        sample_idx=$((sample_idx + NUM_GPUS))

        # Small delay to avoid port conflicts
        sleep 2
    done

    echo "[GPU $gpu] Worker finished: $pass_count passed, $fail_count failed" | tee -a "$log_file"
    echo "$pass_count $fail_count" > "$RESULT_DIR/gpu_${gpu}.summary"
}

# Start time
START_TIME=$(date +%s)
echo "Starting parallel test at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Launch workers for each GPU in background
PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    gpu=${GPU_ARRAY[$i]}
    echo "Launching worker on GPU $gpu..."
    run_gpu_worker $gpu $i &
    PIDS+=($!)
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "Monitor progress with: tail -f $RESULT_DIR/gpu_*.log"
echo ""

# Wait for all workers to complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "FINAL RESULTS"
echo "============================================================"

# Aggregate results
TOTAL_PASS=0
TOTAL_FAIL=0

for gpu in "${GPU_ARRAY[@]}"; do
    if [ -f "$RESULT_DIR/gpu_${gpu}.summary" ]; then
        read pass fail < "$RESULT_DIR/gpu_${gpu}.summary"
        TOTAL_PASS=$((TOTAL_PASS + pass))
        TOTAL_FAIL=$((TOTAL_FAIL + fail))
        echo "GPU $gpu: $pass passed, $fail failed"
    fi
done

TOTAL_TESTED=$((TOTAL_PASS + TOTAL_FAIL))
if [ $TOTAL_TESTED -gt 0 ]; then
    ACCURACY=$(echo "scale=1; $TOTAL_PASS * 100 / $TOTAL_TESTED" | bc)
else
    ACCURACY="0.0"
fi

echo ""
echo "------------------------------------------------------------"
echo "Total: $TOTAL_PASS/$TOTAL_TESTED passed ($ACCURACY%)"
echo "Duration: ${DURATION}s ($(echo "scale=1; $DURATION / 60" | bc) minutes)"
echo "Throughput: $(echo "scale=2; $TOTAL_TESTED * 60 / $DURATION" | bc) samples/min"
echo "------------------------------------------------------------"

# Save detailed results
{
    echo "RULER NIAH Parallel Test Results"
    echo "================================"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "GPUs: $GPUS"
    echo "Total samples: $TOTAL_TESTED"
    echo "Passed: $TOTAL_PASS"
    echo "Failed: $TOTAL_FAIL"
    echo "Accuracy: $ACCURACY%"
    echo "Duration: ${DURATION}s"
    echo ""
    echo "Per-sample results:"
    for i in $(seq 0 $((TOTAL_SAMPLES - 1))); do
        if [ -f "$RESULT_DIR/sample_${i}.result" ]; then
            result=$(cat "$RESULT_DIR/sample_${i}.result")
            echo "Sample $i: $result"
        fi
    done
} > "$OUTPUT_LOG"

echo ""
echo "Detailed results saved to: $OUTPUT_LOG"

# Cleanup
# rm -rf "$RESULT_DIR"

# Exit with appropriate code
if [ $TOTAL_FAIL -eq 0 ]; then
    echo ""
    echo "test_ruler_niah.sh: ALL PASSED"
    exit 0
else
    echo ""
    echo "test_ruler_niah.sh: $TOTAL_FAIL FAILED"
    exit 1
fi
