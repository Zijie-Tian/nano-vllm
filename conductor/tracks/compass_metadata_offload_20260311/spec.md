# Specification: COMPASS Metadata Offloading for TMAC Prediction

## Overview
Modify the `COMPASSPolicy` to implement and verify the offloading mechanism for quantized/packed K-cache metadata and Query tensors (Q). This metadata is crucial for TMAC-based coarse prediction. The focus is on the data pipeline and verification of TMAC accuracy.

## Functional Requirements
1. **Metadata Buffer Management**:
   - Implement allocation of pinned CPU memory buffers for COMPASS metadata (packed K-cache and Q tensors) using `alloc_policy_metadata`.
   - **Verification Mode**: Initially, store all Q chunks to facilitate post-inference verification of the TMAC operator.
2. **TMAC Metadata Offloading**:
   - During `offload_prefill_chunk`, perform 2-bit quantization and bit-serial interleaving packing on the K-cache (reusing existing logic).
   - Offload this packed metadata to the pinned CPU buffers.
3. **Query Tensor Offloading**:
   - During `select_blocks`, offload the current chunk's Q tensor to the pinned CPU buffer.
   - During the initial verification phase, ensure all Q chunks are preserved in the metadata buffer.
4. **Policy Integration (Data Pipeline Only)**:
   - Update `nanovllm/kvcache/sparse/compass.py` to trigger these offloading steps. The attention computation flow itself remains unchanged for now.

## Acceptance Criteria
- Pinned CPU buffers are correctly allocated and populated with packed K-cache and Q tensors.
- **Verification Script**: A standalone script demonstrates that the offloaded metadata (tested with a RULER 32K sample) can be used by the TMAC operator to produce prediction results within the expected numerical error range compared to a standard FP16 GEMM baseline.

## Out of Scope
- Modifying the core sparse attention computation logic (deferred).
- Optimizing memory usage of Q tensors (deferred).
- End-to-end model inference benchmarks.
