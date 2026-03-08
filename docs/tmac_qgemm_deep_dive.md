# T-MAC QGEMM Architecture & Implementation Deep Dive

This document provides a comprehensive technical overview of the T-MAC (Table-Lookup-based Mixed-precision ACceleration) QGEMM implementation in Nano-vLLM, focusing on its algorithm, data layouts, and the performance optimizations implemented during Phase 1.

---

## 1. Core Algorithm: The Lookup Table Revolution

Traditional GEMM relies on floating-point or integer multiplications ($C = A \times B$). T-MAC replaces these expensive multiplications with high-speed table lookups, making it exceptionally efficient for low-bit quantization (e.g., 2-bit).

### 1.1 The mpGEMM Mechanism
For a 2-bit quantized weight matrix, each weight element can only take one of 4 possible values. T-MAC exploits this:
1.  **Bit-Serial Decomposition**: Weights are decomposed into their constituent bits.
2.  **Lookup Table (LUT) Construction**: For each group of activation values, a small LUT is pre-computed containing all possible partial products for that group's bit patterns.
3.  **Multiplication-Free Accumulation**: The actual "multiplication" becomes a simple memory lookup into the LUT using the quantized weight bits as an index, followed by bit-shifting and accumulation.

### 1.2 System Pipeline (The Big Picture)
In the context of the Nano-vLLM infinite context system:
-   **Preprocessor**: Converts the Query vector into a series of Quantized Lookup Tables (QLUTs).
-   **QGEMM Kernel**: Scans the packed, bit-serial KV Cache, performing table lookups against the QLUTs to generate attention scores or mask predictions.

---

## 2. Data Layout: Hierarchical Tiling for SIMD

To achieve >1000 M-tokens/s on CPU, the data layout must perfectly align with SIMD (AVX2/NEON) vector lanes and cache line boundaries.

### 2.1 Weight Packing (`preprocess_weights`)
Standard row-major or column-major formats are unsuitable for T-MAC. Weights undergo a complex transformation:
1.  **Bit-Serial Packing**: Individual bits of multiple weight elements are interleaved and packed into `uint8` containers.
2.  **Hierarchical Tiling**:
    -   **`bm` (M-tile)**: Typically 256 or 512. Controls the granularity of sequence length processing.
    -   **`kfactor` (K-tile)**: Typically 16. Groups consecutive feature dimensions together for a single SIMD lookup operation.
3.  **SIMD Alignment**: The final layout is optimized for `simd_n_in` (16 for uint8) and `simd_n_out` (8 for float32/float16), ensuring that one vector load fetches exactly the data needed for a parallel lookup.

### 2.2 Preprocessor Output (QLUT)
The activation LUTs are grouped by `act_group_size`. This granularity allows for localized dynamic scaling, improving quantization accuracy while keeping the LUTs small enough to fit in CPU L1/L2 caches.

---

## 3. Optimization Insights (Phase 1 Lessons)

During the initial implementation and benchmarking phase, several critical stability and performance barriers were identified and resolved.

### 3.1 Architecture-Aware Type Stability
**The x86 Segfault Bug**:
-   **Discovery**: Large sequence lengths (>32k) caused segmentation faults on x86_64 but worked on small scales.
-   **Root Cause**: A mismatch between the Python codegen (defaulting to `float16`) and the C++ intrinsics (`tbl.cc`), which use 32-bit `float` for AVX2 `_mm256_fmadd_ps` operations.
-   **Solution**: Implemented automatic architecture detection. On x86, the system now automatically promotes `out_dtype` to `float32` and injects `-mavx2 -mfma` compiler flags.

### 3.2 Global Tuning Cache & Log Reusability
**The Multi-Shape Penalty**:
-   **Issue**: TVM treated different sequence lengths ($M$) as unique operators, requiring hours of tuning for every new context length.
-   **Optimization**: Removed $M$ from the TVM `template_name`. Since $M$ only affects the outer loop, the inner loop's optimal `bm` and `kfactor` are invariant.
-   **Result**: A single tuning session for $M=256$ now provides peak performance for any context up to 1M+ tokens.

### 3.3 Task Registration Integrity
**Process-Level Persistence**:
-   **Issue**: Re-compiling the same operator with different parameters in one session caused `ValueError` due to AutoTVM task re-registration.
-   **Solution**: Implemented a global singleton `_GLOBAL_TEMPLATE_CACHE` in `base.py` to manage and reuse decorated template functions safely.

---

## 4. Performance Baseline (2-bit QGEMM)

Validated on x86_64 (AVX2) with 4 threads:
-   **32k Context**: ~110 ms
-   **256k Context**: ~848 ms
-   **512k Context**: ~1548 ms
-   **Throughput**: Consistently reaches **~1300 to ~1500 M-tokens/s** for the QGEMM operation.

---
*Last Updated: 2026-03-09*
