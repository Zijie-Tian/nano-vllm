# Specification: Implement COMPASS Sparse Policy with TMAC Prediction (Sequential)

## Overview
Implement the core sparse block selection logic for the `COMPASSPolicy` using the T-MAC quantized GEMM operator on the CPU. The selected blocks will then be used by the GPU to compute block-sparse attention during the chunked prefill phase. For this initial integration, the CPU prediction and GPU computation will run strictly sequentially (no asynchronous pipeline optimization).

## Functional Requirements
1. **CPU Sparse Estimation (`select_blocks`)**:
   - Utilize the offloaded Q chunk and packed K-cache metadata to perform a coarse-grained prediction on the CPU using the TMAC qGEMM operator.
   - The estimation logic should mirror the structure established in `BLASSTPolicy`, adapting it for TMAC's interface and CPU execution.
2. **Thresholding & Block Selection Strategy**:
   - The TMAC estimation will likely operate at a finer granularity (e.g., 128-token blocks).
   - Apply a threshold mechanism to the TMAC prediction scores (using a target `lambda` value around 0.1).
   - **Simplification**: To map the fine-grained estimation back to the coarse 4096-token chunks managed by `select_blocks`, use a simple inclusive logic: **If ANY fine-grained block (e.g., 128 tokens) within a coarse chunk is selected by ANY head, the entire coarse chunk is selected for offloading.** (This is intentionally inefficient for now to prioritize functional correctness).
3. **Sparse Attention Computation (`compute_chunked_prefill`)**:
   - Filter the `available_blocks` based on the mask generated in `select_blocks`.
   - Forward the selected blocks to the GPU attention kernels for the final dense calculation.
4. **Sequential Execution**:
   - Ensure the workflow is straightforward: CPU estimation fully completes, blocks are selected, and then GPU computation begins. Pipeline parallelism is out of scope.

## Acceptance Criteria
- The `COMPASSPolicy` successfully filters blocks based on TMAC CPU predictions.
- The system runs end-to-end without crashing.
- Evaluating a simple sample from the RULER benchmark (e.g., `niah_single_1`) completes and decodes "reasonably normal" text (ideally passing the first sample).

## Out of Scope
- Performance optimization (e.g., hiding CPU latency via asynchronous execution pipelines).
- Advanced/complex threshold tuning or optimal block-to-chunk mapping algorithms.