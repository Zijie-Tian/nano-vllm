# CPU Attention Operators: Design & API Reference

> Fused QK·Rowmax and BLASST Block-Mask Generation with AVX-512 VNNI

## 1. Overview

The `cpu_ops` library provides high-performance CPU operators for sparse attention inference, specifically targeting the **coarse-grained mask prediction** stage of the COMPASS pipeline. These operators run on the CPU side to determine which KV cache blocks can be skipped before committing to expensive GPU computation.

### Architecture Position

```
GPU Fused Epilogue → [CPU cpu_ops: QK·rowmax + block-mask] → PCIe Delta Transfer → GPU Sparse Attention
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      This library
```

### Operators

| Operator | Purpose | Header |
|----------|---------|--------|
| `qk_rowmax` | Fused QK dot-product + per-row max | `cpu_ops/qk_rowmax.h` |
| `qk_blockmask` | BLASST block-level skip mask generation | `cpu_ops/qk_blockmask.h` |

### Hardware Requirements

- **ISA**: AVX-512F, AVX-512BW, AVX-512VNNI (Ice Lake-SP or later)
- **Target**: Xeon Silver/Gold (24+ cores recommended)
- **Memory**: 64-byte aligned buffers for optimal SIMD loads

---

## 2. Template Micro-Kernel Architecture

All kernels are built on a unified template micro-kernel framework:

```cpp
template <int MR, int NR>  // FP32
template <int MR, int NR>  // VNNI (internal)
```

### Template Parameters

| Param | Axis | Meaning | Values | Recommended |
|-------|------|---------|--------|-------------|
| **MR** | Q | Q-rows per micro-kernel call | 1, 4, 8 | **8** |
| **NR** | K | K-tiles (×16 rows) per call | 1, 2 | **2** |
| **GS** | K | INT8 quantization group size | 0, 128, 512, 2048, 4096 | **4096** |

### Micro-Kernel Execution Model

```
One micro-kernel<MR=8, NR=2> call:
  - Processes 8 Q-rows × 2 K-tiles (32 K-rows)
  - D-loop: load 2 K vectors, broadcast 8 Q scalars → 16 FMAs per d-step
  - Output: 8 × __m512i accumulators (INT32 partial dot products)

Register usage (VNNI, MR=8 NR=2):
  - 16 ZMM: 8×2 accumulators
  - 2 ZMM: K vectors
  - 1 ZMM: Q broadcast (reused)
  Total: ~19/32 ZMM registers
```

### VNNI INT8 Quantization Pipeline

```
┌─────────────────────────────────────────────────┐
│ K Packing (offline, once)                       │
│   FP32 K[BK,D] → INT8 K_vnni[tiles, d_groups]  │
│   Per-group scale_k[ng], per-row sum_k[BK]      │
│   Layout: VNNI interleaved (4 INT8 → 1 INT32)   │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Q Quantization (per-row, online)                │
│   FP32 Q[i,:] → UINT8 q_uint8[d_padded]        │
│   Symmetric: zp=128, scale = absmax/127         │
│   Returns scale_q for dequantization            │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Micro-Kernel (VNNI dpbusd)                      │
│   INT32 acc += UINT8(Q) × INT8(K)   // dpbusd   │
│   acc -= 128 × sum_k   // zero-point correction │
│   FP32 result = acc × scale_q × scale_k[group]  │
└─────────────────────────────────────────────────┘
```

---

## 3. qk_rowmax — Fused QK·Rowmax

### Purpose

Computes `rowmax_out[i] = max_j (Q[i,:] · K[j,:])` for all Q rows. Used as the baseline operator and building block.

### API

```cpp
#include "cpu_ops/qk_rowmax.h"

// FP32
template <int MR, int NR>
void qk_rowmax_fp32_omp(
    const float* Q,           // [BQ, D] query matrix
    const float* K_packed,    // packed K (use pack_k())
    float* rowmax_out,        // [BQ] output: max dot product per row
    size_t BQ, size_t BK, size_t D);

// VNNI INT8
template <int MR, int NR, int GS>
void qk_rowmax_vnni_omp(
    const float* Q,           // [BQ, D] query matrix (FP32, quantized online)
    const int8_t* K_vnni,     // VNNI-packed K (use pack_k_vnni())
    const float* scale_k,     // [num_quant_groups(BK,GS)] per-group scales
    const int32_t* sum_k,     // [BK+16] per-row INT8 sums (for zp correction)
    float* rowmax_out,        // [BQ] output
    size_t BQ, size_t BK, size_t D);
```

### Packing Helpers

```cpp
// FP32: pack K into tile-major layout (16-row tiles, zero-padded)
size_t buf_size = pack_k_buffer_size(BK, D);
float* K_packed = alloc_aligned(buf_size);
pack_k(K, K_packed, BK, D);

// VNNI: quantize + pack K into VNNI INT8 layout
size_t buf_size = pack_k_vnni_buffer_size(BK, D);
int8_t* K_vnni = alloc_aligned_i8(buf_size);
float scale_k[num_quant_groups(BK, GS)];
int32_t sum_k[BK + 16] = {};
pack_k_vnni(K, K_vnni, scale_k, sum_k, BK, D, GS);
```

### OpenMP Strategy

- **Prefill (BQ ≥ threads×MR)**: Parallel across Q-rows, each thread processes MR-aligned chunks
- **Decode (BQ small)**: Parallel across K-tiles with critical-section max reduction

### Performance (24-core Xeon Silver 4310, Q=4096, K=32768)

| Config | D=8 | D=16 | D=32 |
|--------|-----|------|------|
| FP32 `<8,2>` | — | ~4.3 ms (1.0 TFLOPS) | — |
| VNNI `<8,2,4096>` | 1.2 ms | 1.9 ms (2.2 TOPS) | — |

---

## 4. qk_blockmask — BLASST Block-Mask Generation

### Purpose

Implements **BLASST Skip-Softmax** (lines 4–9 of the FlashAttention-BLASST algorithm): computes per-block QK rowmax, compares against a running global maximum, and generates a block-level skip mask via AND-reduce.

### Algorithm

```
for each K-tile j (STEP_KV tokens):
    for each Q-row i:
        m̃[i][j] = rowmax(Q[i] · K_tile_j^T)         // local block rowmax
        m[i] = max(m[i], m̃[i][j])                     // running global max
        row_skip[i] = (m̃[i][j] - m[i]) < ln(λ)       // per-row skip decision

    for each Q-block b (BS rows):
        mask[b][j] = AND(row_skip[b*BS .. (b+1)*BS])   // unanimous vote → skip
```

### Template Parameters

```cpp
template <int MR, int NR, int GS, int BS = 128, int STEP_KV = 128>
```

| Param | Axis | Meaning | Default | GPU Analog |
|-------|------|---------|---------|------------|
| **MR** | Q | Q-row register blocking | 8 | — |
| **NR** | K | K-tile SIMD blocking | 2 | — |
| **GS** | K | INT8 quantization group size | 4096 | — |
| **BS** | Q | Voting window for AND-reduce | 128 | Thread Block Q-tile |
| **STEP_KV** | K | Skip tile granularity | 128 | TRT-LLM `STEP_KV` |

### Dimensional Relationships

```
K dimension (BK tokens):
GS=4096:      |<-------- quant group 0 -------->|<-- group 1 -->| ...
STEP_KV=128:  |t0|t1|...|t31|t32|...|t63| ...   (n_k_blocks skip decisions)
NR=2:         Each micro-kernel call = 32 K-rows → 4 calls per STEP_KV block

Q dimension (BQ rows):
BS=128:       |<-- Q-block 0 (128 rows) -->|<-- Q-block 1 -->| ...
MR=8:         Each micro-kernel call = 8 Q-rows → 16 calls per Q-block
```

### API

```cpp
#include "cpu_ops/qk_blockmask.h"

template <int MR, int NR, int GS, int BS = 128, int STEP_KV = 128>
void qk_blockmask_vnni_omp(
    const float* Q,           // [BQ, D]
    const int8_t* K_vnni,     // VNNI-packed K
    const float* scale_k,     // [num_quant_groups(BK,GS)]
    const int32_t* sum_k,     // [BK+16]
    // --- Outputs ---
    float* block_rowmax,      // [BQ × n_k_blocks] per-row per-block local rowmax
    float* running_max,       // [BQ] in/out: streaming global max
    uint8_t* block_mask,      // [n_q_blocks × n_k_blocks] 1=keep, 0=skip
    // --- Parameters ---
    float log_lambda,         // ln(λ) threshold
    size_t BQ, size_t BK, size_t D);
```

Where:
- `n_k_blocks = ceil(BK / STEP_KV)` — number of K skip decisions
- `n_q_blocks = ceil(BQ / BS)` — number of Q voting groups

### Output Layout

```
block_rowmax [BQ × n_k_blocks]:  row-major, block_rowmax[i * n_k_blocks + j]
running_max  [BQ]:               in/out, initialize to -inf for first call
block_mask   [n_q_blocks × n_k_blocks]:  row-major, 1=keep 0=skip
```

### `running_max` Streaming Protocol

`running_max` is **in/out**: this enables splitting K across multiple calls for streaming/chunked processing.

```cpp
// Single call (typical)
std::vector<float> rmax(BQ, -INFINITY);
qk_blockmask_vnni_omp<8,2,4096,128,128>(..., rmax.data(), ..., BQ, BK, D);

// Streaming: split K into two halves
std::vector<float> rmax(BQ, -INFINITY);
qk_blockmask_vnni_omp<8,2,4096,128,128>(..., rmax.data(), ..., BQ, BK/2, D);
// rmax now contains max from first half
qk_blockmask_vnni_omp<8,2,4096,128,128>(..., rmax.data(), ..., BQ, BK/2, D);
// rmax now contains global max across both halves
```

### AND-Reduce: Block-Level Unanimous Vote

A K-block is skipped **only if ALL Q-rows within the BS voting window agree**:

```
block_mask[qblk][kblk] = 1   if ANY row in [qblk*BS, (qblk+1)*BS) wants to keep
block_mask[qblk][kblk] = 0   if ALL rows in the window agree to skip
```

This matches TensorRT-LLM's `atomicAnd(skip_softmax_vote, ...)` implementation, ensuring conservative skip decisions that preserve accuracy.

### Threshold Selection

| λ | ln(λ) | Behavior |
|---|-------|----------|
| 1e-10 | −23.0 | Keep everything (correctness validation) |
| 1e-3 | −6.9 | Aggressive skip |
| 0.1 | −2.3 | Moderate skip |
| 0.9 | −0.11 | Very conservative (almost no skip) |

### OpenMP Strategy (HPC v2)

The OMP implementation applies 5 micro-architectural optimizations:

1. **Loop Inversion**: Outermost parallel on Q-blocks (`#pragma omp parallel for` on `qblk`) — single fork, zero barriers per K-block. Each thread owns a tiny `BS × d_padded` L1-resident Q cache.
2. **Deferred Horizontal Reduction**: Accumulates rowmax in FP32 `__m512` vectors (`_mm512_cvtepi32_ps` + `_mm512_max_ps`). Only one `_mm512_reduce_max_ps` scalar extract per K-block.
3. **Branch-Free Inner Loop**: Pre-computes `tiles_per_group`, chunks the tile loop by quantization group boundaries, and peels the last-tile mask handling out of the hot loop.
4. **NR Template Unrolling**: Main loop calls `microkernel_vnni<MR, NR>` (NR=2 → 32 K-rows/call), increasing ILP and ZMM register utilization.
5. **Dynamic Padding**: Tail Q-rows (< MR) padded by replicating the last valid row pointer, eliminating the `<1,1>` fallback code path entirely.

### Performance (24-core Xeon Silver 4310, Q=4096, `<8,2,4096,128,128>`)

| K | D=8 | D=16 | D=32 |
|---|-----|------|------|
| **32k** | 2.31 ms (929 GOPS) | 3.24 ms (1327 GOPS) | 5.19 ms (1654 GOPS) |
| **128k** | 9.39 ms (915 GOPS) | 12.4 ms (1389 GOPS) | 18.3 ms (1926 GOPS) |
| **1M** | 120 ms (984 GOPS) | 141 ms (973 GOPS) | 187 ms (2160 GOPS) |

**Scaling Analysis**:
- **32k → 128k (4×)**: Time scales ~3.6-4×, near-linear (K data fits in L3 cache).
- **128k → 1M (8×)**: Time scales ~10-12×, super-linear due to L3 eviction (36MB L3 ≪ 1M × D bytes K data → DRAM-bound).
- **Peak throughput**: D=32 reaches **2.16 TOPS** at 1M context, demonstrating good compute density.
- Overhead vs `qk_rowmax_vnni_omp`: ~1.7× at 32k (was 2× before HPC optimizations).

---

## 5. File Structure

```
tests/cpu_ops/
├── CMakeLists.txt
├── include/cpu_ops/
│   ├── qk_rowmax.h                  # Rowmax public API
│   ├── qk_blockmask.h               # Blockmask public API
│   └── internal/
│       └── vnni_kernels.h            # Shared VNNI micro-kernel primitives
├── src/
│   ├── qk_rowmax_avx512.cpp          # FP32 rowmax + K packing
│   ├── qk_rowmax_vnni.cpp            # VNNI rowmax + K quantization
│   ├── qk_blockmask_fp32.cpp         # FP32 blockmask (reference)
│   └── qk_blockmask_vnni.cpp         # VNNI blockmask (production)
└── tests/
    ├── test_qk_rowmax.cpp             # Rowmax tests + benchmarks
    └── test_qk_blockmask.cpp          # Blockmask tests + benchmarks
```

### Shared Primitives (`internal/vnni_kernels.h`)

```cpp
namespace cpu_ops::internal {
    float quantize_q_row(const float* q_fp32, uint8_t* q_uint8, size_t D, size_t d_padded);
    template <int MR, int NR> void microkernel_vnni(...);
}
```

These are header-only (`static inline`) for maximum inlining. Both `qk_rowmax_vnni.cpp` and `qk_blockmask_vnni.cpp` use them via `using internal::microkernel_vnni`.

---

## 6. Build & Test

```bash
cd tests/cpu_ops/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
OMP_NUM_THREADS=24 taskset -c 0,2,...,46 ./test_qk_rowmax
OMP_NUM_THREADS=24 taskset -c 0,2,...,46 ./test_qk_blockmask
```

### Correctness Validation

- VNNI vs FP32 reference: `block_rowmax` relative error < 5%
- Mask comparison: 0 diffs across all tested configs
- λ=1e-10 assertion: all blocks must be kept (100% keep)

---

## 7. Instantiated Configurations

### qk_rowmax

| MR | NR | GS | Notes |
|----|----|----|-------|
| 1,4,8 | 1,2 | 0 | Per-tensor quantization |
| 8 | 2 | 128,512,2048 | Per-group quantization |
| 1,4,8 | 1,2 | 4096 | Recommended production |

### qk_blockmask

| MR | NR | GS | BS | STEP_KV | Notes |
|----|----|----|----|---------| ------|
| 8 | 2 | 4096 | 128 | 128 | **Default production** |
| 8 | 2 | 4096 | 128 | 64 | Finer K granularity |
| 8 | 2 | 4096 | 64 | 128 | Finer Q voting |
| 8 | 2 | 4096 | 64 | 64 | Finest granularity |
| 8 | 2 | 0 | 128 | 128 | Per-tensor quantization |

---

**Author**: Zijie Tian / Antigravity AI
**Version**: 1.0
