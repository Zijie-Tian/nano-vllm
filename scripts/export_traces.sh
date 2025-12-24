#!/bin/bash

# Export detailed profiling traces from nsys report
#
# Usage:
#   bash scripts/export_traces.sh <nsys_report_file>
#
# Example:
#   bash scripts/export_traces.sh results/nsys/attention_offload_20251224_205806.nsys-rep

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <nsys_report_file>"
    echo "Example: $0 results/nsys/attention_offload_20251224_205806.nsys-rep"
    exit 1
fi

NSYS_REPORT="$1"
BASENAME=$(basename "$NSYS_REPORT" .nsys-rep)
OUTPUT_DIR="results/nsys/traces"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Exporting traces from: $NSYS_REPORT"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Export NVTX Push/Pop trace (shows timeline and nesting)
echo ""
echo "[1/4] Exporting NVTX Push/Pop trace..."
nsys stats --report nvtx_pushpop_trace "$NSYS_REPORT" \
    > "$OUTPUT_DIR/${BASENAME}_nvtx_trace.txt" 2>&1
echo "  -> $OUTPUT_DIR/${BASENAME}_nvtx_trace.txt"

# Export CUDA GPU trace (shows kernel execution timeline)
echo ""
echo "[2/4] Exporting CUDA GPU trace..."
nsys stats --report cuda_gpu_trace "$NSYS_REPORT" \
    > "$OUTPUT_DIR/${BASENAME}_cuda_gpu_trace.txt" 2>&1
echo "  -> $OUTPUT_DIR/${BASENAME}_cuda_gpu_trace.txt"

# Export CUDA API trace (shows API calls)
echo ""
echo "[3/4] Exporting CUDA API trace..."
nsys stats --report cuda_api_trace "$NSYS_REPORT" \
    > "$OUTPUT_DIR/${BASENAME}_cuda_api_trace.txt" 2>&1
echo "  -> $OUTPUT_DIR/${BASENAME}_cuda_api_trace.txt"

# Export NVTX kernel summary (shows which kernels ran within NVTX ranges)
echo ""
echo "[4/4] Exporting NVTX kernel summary..."
nsys stats --report nvtx_kern_sum "$NSYS_REPORT" \
    > "$OUTPUT_DIR/${BASENAME}_nvtx_kern_sum.txt" 2>&1
echo "  -> $OUTPUT_DIR/${BASENAME}_nvtx_kern_sum.txt"

echo ""
echo "============================================================"
echo "Traces exported successfully!"
echo "============================================================"
echo ""
echo "Key files:"
echo "  - nvtx_trace.txt:      Timeline with NVTX markers (shows nesting and timing)"
echo "  - cuda_gpu_trace.txt:  GPU kernel execution timeline"
echo "  - cuda_api_trace.txt:  CUDA API call timeline"
echo "  - nvtx_kern_sum.txt:   Kernels grouped by NVTX ranges"
echo ""
echo "For visual analysis, open in Nsight Systems GUI:"
echo "  nsight-sys $NSYS_REPORT"
echo "============================================================"
