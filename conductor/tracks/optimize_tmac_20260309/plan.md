# Implementation Plan: Optimize TMAC QGEMM and Implement KVCache Quantization and Packing

## Phase 1: TMAC QGEMM Optimization [checkpoint: 5c8dda7]
- [x] Task: Review current T-MAC QGEMM implementation and identify bottlenecks. [5c8dda7]
- [x] Task: Apply performance optimizations (e.g., memory layout, TVM schedule refinement). [5c8dda7]
- [x] Task: Verify optimized QGEMM correctness and benchmark performance. [5c8dda7]
- [x] Task: Conductor - User Manual Verification 'Phase 1: TMAC QGEMM Optimization' (Protocol in workflow.md) [5c8dda7]

## Phase 2: KV Cache Quantization [checkpoint: 63da706]
- [x] Task: Design and implement the 2-bit quantization algorithm (SRAM-level/Warp-level). [b3de993]
- [x] Task: Write unit tests for quantization correctness. [b3de993]
- [x] Task: Conductor - User Manual Verification 'Phase 2: KV Cache Quantization' (Protocol in workflow.md) [63da706]

## Phase 3: KV Cache Packing [checkpoint: 940effb]
- [x] Task: Implement bit-serial interleaving packing for quantized KV blocks. [ad8ba0a]
- [x] Task: Optimize packing performance using bitwise PyTorch ops and chunking. [ed1edaa]
- [x] Task: Verify packing correctness and long-context (1M) stability. [940effb]
- [x] Task: Conductor - User Manual Verification 'Phase 3: KV Cache Packing' (Protocol in workflow.md) [ed1edaa]
