#pragma once
/**
 * Internal VNNI micro-kernel primitives shared between qk_rowmax and qk_blockmask.
 * NOT part of the public API.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <limits>

namespace cpu_ops {
namespace internal {

// ============================================================
// Q per-row quantization (symmetric UINT8, zero-point = 128)
// ============================================================
static inline float quantize_q_row(const float* q_fp32, uint8_t* q_uint8,
                                    size_t D, size_t d_padded) {
    float absmax = 0.0f;
    for (size_t d = 0; d < D; ++d)
        absmax = std::max(absmax, std::fabs(q_fp32[d]));

    float scale = absmax / 127.0f;
    float inv_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;

    for (size_t d = 0; d < D; ++d) {
        int val = (int)std::roundf(q_fp32[d] * inv_scale);
        q_uint8[d] = (uint8_t)(std::max(-127, std::min(127, val)) + 128);
    }
    for (size_t d = D; d < d_padded; ++d)
        q_uint8[d] = 128;

    return scale;
}

// ============================================================
// Template VNNI micro-kernel: MR Q-rows × NR K-tiles(×16)
// Uses _mm512_dpbusd_epi32 for INT8→INT32 dot product
// ============================================================
template <int MR, int NR>
static inline __attribute__((always_inline))
void microkernel_vnni(
        const uint8_t* Q_u8[MR],
        size_t d_groups,
        const int8_t* K_vnni,
        size_t tile_stride,
        const int32_t* sum_k,
        __m512i rmax[MR],
        __mmask16 masks[NR],
        const __m512i& neg_inf_i,
        const __m512i& correction_128) {

    __m512i acc[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int n = 0; n < NR; ++n)
            acc[m][n] = _mm512_setzero_si512();

    for (size_t g = 0; g < d_groups; ++g) {
        __m512i k_vals[NR];
        for (int n = 0; n < NR; ++n)
            k_vals[n] = _mm512_load_si512(
                (__m512i*)(K_vnni + n * tile_stride + g * 64));

        for (int m = 0; m < MR; ++m) {
            int32_t qb;
            memcpy(&qb, Q_u8[m] + g * 4, 4);
            __m512i q_vec = _mm512_set1_epi32(qb);
            for (int n = 0; n < NR; ++n)
                acc[m][n] = _mm512_dpbusd_epi32(acc[m][n], q_vec, k_vals[n]);
        }
    }

    for (int n = 0; n < NR; ++n) {
        __m512i sk = _mm512_loadu_si512((__m512i*)(sum_k + n * 16));
        __m512i corr = _mm512_mullo_epi32(correction_128, sk);
        for (int m = 0; m < MR; ++m) {
            acc[m][n] = _mm512_sub_epi32(acc[m][n], corr);
            acc[m][n] = _mm512_mask_blend_epi32(masks[n], neg_inf_i, acc[m][n]);
            rmax[m] = _mm512_max_epi32(rmax[m], acc[m][n]);
        }
    }
}

}  // namespace internal
}  // namespace cpu_ops
