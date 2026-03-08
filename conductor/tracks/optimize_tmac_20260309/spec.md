# Specification: Optimize TMAC QGEMM and Implement KVCache Quantization and Packing

## 1. Goal
Optimize the existing TMAC QGEMM operator and implement the necessary functions for KV cache quantization and packing to enable efficient CPU-side coarse prediction.

## 2. Scope
- **QGEMM Optimization:** Refine and optimize the TVM-based T-MAC QGEMM implementation for better performance.
- **KV Cache Quantization:** Implement 2-bit quantization for FP16 KV cache blocks.
- **KV Cache Packing:** Implement bit-serial interleaving packing for quantized KV blocks to match T-MAC's requirements.
- **Integration:** Ensure the new operators and functions are compatible with the existing `nanovllm` codebase.

## 3. Technical Requirements
- **Frameworks:** PyTorch, TVM, Triton/CUDA.
- **Performance:** Optimized for AVX-enabled CPUs and minimal GPU overhead during quantization.
