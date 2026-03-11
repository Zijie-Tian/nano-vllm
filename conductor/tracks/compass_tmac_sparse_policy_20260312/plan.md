# Implementation Plan: Implement COMPASS Sparse Policy with TMAC Prediction

## Phase 1: TMAC Estimator Integration Setup
- [x] Task: Instantiate the TMAC `QGeMMLUTBitsCodegen` engine within `COMPASSPolicy` initialization. Ensure it handles the required dimensions (M, N, K) based on the model's head dimension and typical chunk sizes.
- [x] Task: Conductor - User Manual Verification 'Phase 1: TMAC Estimator Integration Setup' (Protocol in workflow.md)

## Phase 2: CPU Sparse Estimation Logic
- [ ] Task: Implement the TMAC estimation loop inside `COMPASSPolicy.select_blocks`. This involves taking the offloaded `_q_buffer` and `_k_packed_buffer` for historical blocks and running the TMAC operator to compute the coarse attention scores.
- [ ] Task: Implement the simplified thresholding logic (e.g., lambda = 0.1) on the computed scores to identify selected fine-grained blocks.
- [ ] Task: Implement the block-to-chunk mapping logic: If any head selects a fine-grained block within a 4096-token chunk, mark the entire chunk as selected.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: CPU Sparse Estimation Logic' (Protocol in workflow.md)

## Phase 3: GPU Integration and Validation
- [ ] Task: Update `COMPASSPolicy.compute_chunked_prefill` to utilize the filtered `selected_blocks` mask and pass it to the underlying Triton attention kernels (similar to `BLASSTPolicy`).
- [ ] Task: Run the `test_ruler.py` script on a single sample (`niah_single_1`) to verify end-to-end sequential execution, ensuring no crashes and a reasonable output.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: GPU Integration and Validation' (Protocol in workflow.md)