#!/bin/bash
# RULER Dataset Download Script
# Run this script to download required datasets for RULER benchmark
#
# Environment setup is documented in: docs/CLAUDE_SETUP_TASK.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== RULER Dataset Download ==="

# Install dependencies for download scripts (if not already installed)
pip install -q html2text beautifulsoup4 2>/dev/null || true

# Download datasets (skip if already exist)
cd scripts/data/synthetic/json/

if [ -f "PaulGrahamEssays.json" ] && [ -f "hotpotqa.json" ] && [ -f "squad.json" ]; then
    echo "All datasets already exist, skipping download."
    echo "  - PaulGrahamEssays.json"
    echo "  - hotpotqa.json"
    echo "  - squad.json"
else
    echo "Downloading datasets..."

    if [ ! -f "PaulGrahamEssays.json" ]; then
        echo "  Downloading Paul Graham Essays..."
        python download_paulgraham_essay.py
    fi

    if [ ! -f "hotpotqa.json" ] || [ ! -f "squad.json" ]; then
        echo "  Downloading QA datasets..."
        bash download_qa_dataset.sh
    fi

    echo "Download complete!"
fi

cd "$SCRIPT_DIR"
echo "=== Done ==="
