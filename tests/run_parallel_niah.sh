#!/bin/bash
# Run NIAH tests in parallel on 6 GPUs
# This tests the dynamic port allocation fix

set -e

MODEL="${1:-/home/zijie/models/Llama-3.1-8B-Instruct}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=========================================="
echo "Parallel NIAH Test on 6 GPUs"
echo "=========================================="
echo "Model: $MODEL"
echo "Project: $PROJECT_ROOT"
echo ""

# Sample distribution (100 samples total):
# GPU 0: 0-16   (17 samples)
# GPU 1: 17-33  (17 samples)
# GPU 2: 34-50  (17 samples)
# GPU 3: 51-67  (17 samples)
# GPU 4: 68-83  (16 samples)
# GPU 5: 84-99  (16 samples)

declare -a RANGES=("0-16" "17-33" "34-50" "51-67" "68-83" "84-99")
declare -a PIDS=()

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Start all 6 processes
for gpu in {0..5}; do
    range="${RANGES[$gpu]}"
    log_file="$LOG_DIR/gpu${gpu}_${range}.log"

    echo "Starting GPU $gpu: samples $range -> $log_file"

    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
        python "$PROJECT_ROOT/tests/test_ruler_niah.py" \
        --model "$MODEL" \
        --sample-indices "$range" \
        --enable-offload \
        --num-gpu-blocks 4 \
        --quiet \
        > "$log_file" 2>&1 &

    PIDS+=($!)

    # Small delay to stagger starts
    sleep 2
done

echo ""
echo "All 6 processes started. Waiting for completion..."
echo "PIDs: ${PIDS[*]}"
echo ""

# Wait for all processes and collect results
declare -a RESULTS=()
ALL_PASSED=true

for i in {0..5}; do
    pid="${PIDS[$i]}"
    range="${RANGES[$i]}"
    log_file="$LOG_DIR/gpu${i}_${range}.log"

    if wait $pid; then
        RESULTS+=("GPU $i ($range): PASSED")
        echo "GPU $i completed successfully"
    else
        RESULTS+=("GPU $i ($range): FAILED (exit code $?)")
        ALL_PASSED=false
        echo "GPU $i FAILED!"
    fi
done

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
for result in "${RESULTS[@]}"; do
    echo "$result"
done
echo ""

# Show accuracy from each log
echo "Accuracy per GPU:"
for i in {0..5}; do
    range="${RANGES[$i]}"
    log_file="$LOG_DIR/gpu${i}_${range}.log"
    if [ -f "$log_file" ]; then
        accuracy=$(grep -E "Accuracy:|accuracy" "$log_file" | tail -1 || echo "N/A")
        port=$(grep "Auto-assigned distributed port" "$log_file" | head -1 || echo "N/A")
        echo "  GPU $i ($range): $accuracy | $port"
    fi
done

echo ""
if $ALL_PASSED; then
    echo "=========================================="
    echo "ALL 6 TESTS PASSED!"
    echo "Dynamic port allocation works correctly."
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "SOME TESTS FAILED!"
    echo "Check logs in $LOG_DIR"
    echo "=========================================="
    exit 1
fi
