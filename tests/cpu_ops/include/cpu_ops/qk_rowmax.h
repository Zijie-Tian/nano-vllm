#pragma once
/**
 * cpu_ops/qk_rowmax.h - Fused QK·rowmax with template micro-kernels
 *
 * Template parameters:
 *   MR: number of Q rows processed per micro-kernel invocation
 *   NR: number of K tiles (×16 rows) processed per micro-kernel invocation
 *   GS: quantization group size along K (VNNI only, 0 = per-tensor)
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace cpu_ops {

// ============================================================
// Aligned memory helpers
// ============================================================
inline float* alloc_aligned(size_t num_floats) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, num_floats * sizeof(float))) return nullptr;
    return static_cast<float*>(ptr);
}
inline int8_t* alloc_aligned_i8(size_t num_bytes) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, num_bytes)) return nullptr;
    return static_cast<int8_t*>(ptr);
}
inline void free_aligned(void* ptr) { free(ptr); }

// ============================================================
// FP32 packing
// ============================================================
inline size_t pack_k_buffer_size(size_t BK, size_t D) {
    return ((BK + 15) / 16) * D * 16;
}
void pack_k(const float* K, float* K_packed, size_t BK, size_t D);

// ============================================================
// VNNI INT8 packing (per-group quantization)
// ============================================================
inline size_t pack_k_vnni_buffer_size(size_t BK, size_t D) {
    return ((BK + 15) / 16) * ((D + 3) / 4) * 64;
}

/**
 * Number of quantization groups for given BK and GS.
 * GS=0 means per-tensor (1 group).
 */
inline size_t num_quant_groups(size_t BK, size_t GS) {
    return (GS == 0) ? 1 : ((BK + GS - 1) / GS);
}

/**
 * Quantize and pack K into VNNI INT8 layout with per-group scales.
 * GS=0: per-tensor (1 scale). GS=4096: every 4096 K rows share one scale.
 * scale_k: array of num_quant_groups(BK, GS) floats
 * sum_k:   array of BK+16 int32 (per-row INT8 sums, padded)
 */
void pack_k_vnni(const float* K, int8_t* K_vnni, float* scale_k,
                 int32_t* sum_k, size_t BK, size_t D, size_t GS);

// ============================================================
// FP32 template kernels
// ============================================================
template <int MR, int NR>
void qk_rowmax_fp32(const float* Q, const float* K_packed,
                    float* rowmax_out,
                    size_t BQ, size_t BK, size_t D);

template <int MR, int NR>
void qk_rowmax_fp32_omp(const float* Q, const float* K_packed,
                        float* rowmax_out,
                        size_t BQ, size_t BK, size_t D);

// ============================================================
// VNNI INT8 template kernels (GS = quantization group size)
// ============================================================
template <int MR, int NR, int GS>
void qk_rowmax_vnni(const float* Q, const int8_t* K_vnni,
                    const float* scale_k, const int32_t* sum_k,
                    float* rowmax_out,
                    size_t BQ, size_t BK, size_t D);

template <int MR, int NR, int GS>
void qk_rowmax_vnni_omp(const float* Q, const int8_t* K_vnni,
                        const float* scale_k, const int32_t* sum_k,
                        float* rowmax_out,
                        size_t BQ, size_t BK, size_t D);

// ============================================================
// Explicit instantiations
// ============================================================
// FP32:  <MR, NR>  = <1,1> <4,1> <4,2> <8,1> <8,2>
// VNNI:  <MR, NR, GS> = {1,4,8} × {1,2} × {0, 4096}

}  // namespace cpu_ops
