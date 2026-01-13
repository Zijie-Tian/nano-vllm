#!/bin/bash
# RULER Benchmark Runner (Conda Mode)
# Usage: ./scripts/run_ruler.sh [MODEL_NAME] [BENCHMARK] [METRIC]

set -e

#############################################
# Configuration
#############################################

# Conda environment
CONDA_ENV="${CONDA_ENV:-ruler}"

# Model settings
MODEL_NAME="${1:-llama3.1-8b-chat}"
BENCHMARK="${2:-synthetic}"
METRIC="${3:-full}"  # Options: full, xattn, avgpool, compass, minfer, flex

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
MODEL_DIR="${MODEL_DIR:-/home/zijie/models}"

# Set PYTHONPATH (nanovllm from 3rdparty takes priority over system version)
export PYTHONPATH="${PROJECT_DIR}/3rdparty/nanovllm:${PROJECT_DIR}:${PYTHONPATH}"

#############################################
# Activate conda and run
#############################################

echo "========================================"
echo "RULER Benchmark (Conda Mode)"
echo "========================================"
echo "Conda Env:    $CONDA_ENV"
echo "Project:      $PROJECT_DIR"
echo "Model Dir:    $MODEL_DIR"
echo "Model:        $MODEL_NAME"
echo "Benchmark:    $BENCHMARK"
echo "Metric:       $METRIC"
echo "========================================"

# Source conda
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Activate conda environment
conda activate "$CONDA_ENV"

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Download datasets if needed (auto-skip if exist)
cd "${PROJECT_DIR}/eval/RULER"
bash setup.sh

# Run RULER benchmark
cd "${PROJECT_DIR}/eval/RULER/scripts"
./run.sh "$MODEL_NAME" "$BENCHMARK" --metric "$METRIC"
