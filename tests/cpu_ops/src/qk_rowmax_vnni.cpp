/**
 * qk_rowmax_vnni.cpp - VNNI INT8 Fused QK·rowmax with template micro-kernels
 *
 * Template parameters:
 *   MR: Q-row blocking
 *   NR: K-tile (×16) blocking
 *   GS: quantization group size (0 = per-tensor, 4096 = per-4096-rows)
 *
 * Per-group quantization:
 *   Each group of GS K-rows shares one scale_k. Within a group, INT32 max
 *   is valid. Across groups, convert to FP32 (multiply by scale_k[g]) and
 *   take FP32 max.
 */

#include "cpu_ops/qk_rowmax.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpu_ops {

// ============================================================
// K quantization + VNNI packing (per-group)
// ============================================================
void pack_k_vnni(const float* K, int8_t* K_vnni, float* scale_k,
                 int32_t* sum_k, size_t BK, size_t D, size_t GS) {
    size_t n_tiles = (BK + 15) / 16;
    size_t d_groups = (D + 3) / 4;
    size_t n_qgroups = (GS == 0) ? 1 : ((BK + GS - 1) / GS);
    size_t actual_gs = (GS == 0) ? BK : GS;

    // Per-group quantization
    std::vector<int8_t> K_int8(BK * D, 0);

    for (size_t g = 0; g < n_qgroups; ++g) {
        size_t j_start = g * actual_gs;
        size_t j_end = std::min(j_start + actual_gs, BK);

        // Find absmax within this group
        float absmax = 0.0f;
        for (size_t j = j_start; j < j_end; ++j)
            for (size_t d = 0; d < D; ++d)
                absmax = std::max(absmax, std::fabs(K[j * D + d]));

        float scale = absmax / 127.0f;
        float inv_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
        scale_k[g] = scale;

        // Quantize rows in this group
        for (size_t j = j_start; j < j_end; ++j) {
            int32_t row_sum = 0;
            for (size_t d = 0; d < D; ++d) {
                int val = (int)std::roundf(K[j * D + d] * inv_scale);
                val = std::max(-128, std::min(127, val));
                K_int8[j * D + d] = (int8_t)val;
                row_sum += val;
            }
            sum_k[j] = row_sum;
        }
    }
    // Zero-pad sum_k tail for aligned loads
    for (size_t j = BK; j < BK + 16; ++j) sum_k[j] = 0;

    // Pack into VNNI interleaved layout
    memset(K_vnni, 0, n_tiles * d_groups * 64);
    for (size_t t = 0; t < n_tiles; ++t) {
        size_t j_base = t * 16;
        size_t j_count = std::min((size_t)16, BK - j_base);
        for (size_t dg = 0; dg < d_groups; ++dg) {
            int8_t* dst = K_vnni + (t * d_groups + dg) * 64;
            for (size_t j = 0; j < j_count; ++j)
                for (size_t k = 0; k < 4 && (dg * 4 + k) < D; ++k)
                    dst[j * 4 + k] = K_int8[(j_base + j) * D + dg * 4 + k];
        }
    }
}

// ============================================================
// Q per-row quantization
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
// Template VNNI micro-kernel (unchanged — operates on NR tiles)
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

// ============================================================
// Process one quantization group: tiles [t_start, t_end)
// Returns INT32 rowmax within this group for MR rows.
// ============================================================
template <int MR, int NR>
static inline void process_group(
        const uint8_t* q_ptrs[MR],
        size_t d_groups,
        const int8_t* K_vnni,
        const int32_t* sum_k,
        size_t tile_stride,
        size_t t_start, size_t t_end, size_t n_tiles_total,
        __mmask16 last_tile_mask,
        int32_t group_max_out[MR]) {

    __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
    __m512i correction_128 = _mm512_set1_epi32(128);

    __m512i rmax[MR];
    for (int m = 0; m < MR; ++m) rmax[m] = neg_inf_i;

    size_t t = t_start;
    for (; t + NR <= t_end; t += NR) {
        __mmask16 masks[NR];
        for (int n = 0; n < NR; ++n)
            masks[n] = ((t + n) == n_tiles_total - 1) ? last_tile_mask : (__mmask16)0xFFFF;
        microkernel_vnni<MR, NR>(
            q_ptrs, d_groups,
            K_vnni + t * tile_stride, tile_stride,
            sum_k + t * 16,
            rmax, masks, neg_inf_i, correction_128);
    }
    for (; t < t_end; ++t) {
        __mmask16 masks[1] = { (t == n_tiles_total - 1) ? last_tile_mask : (__mmask16)0xFFFF };
        microkernel_vnni<MR, 1>(
            q_ptrs, d_groups,
            K_vnni + t * tile_stride, tile_stride,
            sum_k + t * 16,
            rmax, masks, neg_inf_i, correction_128);
    }

    for (int m = 0; m < MR; ++m)
        group_max_out[m] = _mm512_reduce_max_epi32(rmax[m]);
}

// ============================================================
// Outer loop: iterate over quant groups, FP32 max across groups
// ============================================================
template <int MR, int NR, int GS>
static void compute_all_rows_vnni(
        const float* Q, const int8_t* K_vnni,
        const float* scale_k, const int32_t* sum_k,
        float* rowmax_out,
        size_t BQ, size_t BK, size_t D,
        size_t i_start, size_t i_end) {

    size_t n_tiles = (BK + 15) / 16;
    size_t d_groups = (D + 3) / 4;
    size_t d_padded = d_groups * 4;
    size_t tail_count = BK % 16;
    __mmask16 last_tile_mask = (tail_count == 0) ? (__mmask16)0xFFFF : ((__mmask16)((1U << tail_count) - 1));
    size_t tile_stride = d_groups * 64;

    size_t actual_gs = (GS == 0) ? BK : (size_t)GS;
    size_t n_qgroups = (BK + actual_gs - 1) / actual_gs;

    uint8_t q_buf[8][64];

    size_t i = i_start;
    for (; i + MR <= i_end; i += MR) {
        float sq[MR];
        const uint8_t* q_ptrs[MR];
        for (int m = 0; m < MR; ++m) {
            sq[m] = quantize_q_row(Q + (i + m) * D, q_buf[m], D, d_padded);
            q_ptrs[m] = q_buf[m];
        }

        // FP32 running max across quant groups
        float fp_rmax[MR];
        for (int m = 0; m < MR; ++m)
            fp_rmax[m] = -std::numeric_limits<float>::infinity();

        for (size_t g = 0; g < n_qgroups; ++g) {
            size_t t_start_g = (g * actual_gs) / 16;
            size_t t_end_g = std::min(((g + 1) * actual_gs + 15) / 16, n_tiles);

            int32_t group_max[MR];
            process_group<MR, NR>(
                q_ptrs, d_groups, K_vnni, sum_k, tile_stride,
                t_start_g, t_end_g, n_tiles, last_tile_mask,
                group_max);

            // Convert to FP32 with this group's scale and fold into running max
            for (int m = 0; m < MR; ++m) {
                float val = (float)group_max[m] * sq[m] * scale_k[g];
                fp_rmax[m] = std::max(fp_rmax[m], val);
            }
        }

        for (int m = 0; m < MR; ++m)
            rowmax_out[i + m] = fp_rmax[m];
    }
    // Tail rows
    for (; i < i_end; ++i) {
        float sq = quantize_q_row(Q + i * D, q_buf[0], D, d_padded);
        const uint8_t* q_ptrs[1] = { q_buf[0] };

        float fp_rmax = -std::numeric_limits<float>::infinity();
        for (size_t g = 0; g < n_qgroups; ++g) {
            size_t t_start_g = (g * actual_gs) / 16;
            size_t t_end_g = std::min(((g + 1) * actual_gs + 15) / 16, n_tiles);

            int32_t gmax[1];
            process_group<1, 1>(
                q_ptrs, d_groups, K_vnni, sum_k, tile_stride,
                t_start_g, t_end_g, n_tiles, last_tile_mask, gmax);
            fp_rmax = std::max(fp_rmax, (float)gmax[0] * sq * scale_k[g]);
        }
        rowmax_out[i] = fp_rmax;
    }
}

// ============================================================
// Public: single-threaded
// ============================================================
template <int MR, int NR, int GS>
void qk_rowmax_vnni(const float* Q, const int8_t* K_vnni,
                    const float* scale_k, const int32_t* sum_k,
                    float* rowmax_out,
                    size_t BQ, size_t BK, size_t D) {
    compute_all_rows_vnni<MR, NR, GS>(Q, K_vnni, scale_k, sum_k,
                                       rowmax_out, BQ, BK, D, 0, BQ);
}

// ============================================================
// Public: adaptive OpenMP
// ============================================================
template <int MR, int NR, int GS>
void qk_rowmax_vnni_omp(const float* Q, const int8_t* K_vnni,
                        const float* scale_k, const int32_t* sum_k,
                        float* rowmax_out,
                        size_t BQ, size_t BK, size_t D) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    if (BQ >= (size_t)num_threads * MR) {
        // Path A: parallel along Q
        #pragma omp parallel
        {
            int tid = 0, nthreads = 1;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
#endif
            size_t rows_per_thread = ((BQ / MR + nthreads - 1) / nthreads) * MR;
            size_t is = std::min((size_t)tid * rows_per_thread, BQ);
            size_t ie = std::min(is + rows_per_thread, BQ);
            compute_all_rows_vnni<MR, NR, GS>(Q, K_vnni, scale_k, sum_k,
                                               rowmax_out, BQ, BK, D, is, ie);
        }
    } else {
        // Path B: parallel along K tiles (decode)
        size_t n_tiles = (BK + 15) / 16;
        size_t d_groups = (D + 3) / 4;
        size_t d_padded = d_groups * 4;
        size_t tail_count = BK % 16;
        __mmask16 last_tile_mask = (tail_count == 0) ? (__mmask16)0xFFFF : ((__mmask16)((1U << tail_count) - 1));
        __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
        __m512i correction_128 = _mm512_set1_epi32(128);
        size_t tile_stride = d_groups * 64;
        size_t actual_gs = (GS == 0) ? BK : (size_t)GS;
        size_t n_qgroups = (BK + actual_gs - 1) / actual_gs;

        for (size_t i = 0; i < BQ; ++i) {
            uint8_t q_u8[64];
            float sq = quantize_q_row(Q + i * D, q_u8, D, d_padded);

            float fp_rmax = -std::numeric_limits<float>::infinity();
            for (size_t g = 0; g < n_qgroups; ++g) {
                size_t t_start_g = (g * actual_gs) / 16;
                size_t t_end_g = std::min(((g + 1) * actual_gs + 15) / 16, n_tiles);

                int32_t global_max = 0x80000000;
                #pragma omp parallel
                {
                    __m512i local_max = neg_inf_i;
                    #pragma omp for schedule(static) nowait
                    for (size_t t = t_start_g; t < t_end_g; ++t) {
                        __m512i acc = _mm512_setzero_si512();
                        for (size_t dg = 0; dg < d_groups; ++dg) {
                            __m512i k_vec = _mm512_load_si512(
                                (__m512i*)(K_vnni + t * tile_stride + dg * 64));
                            int32_t qb;
                            memcpy(&qb, q_u8 + dg * 4, 4);
                            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(qb), k_vec);
                        }
                        __m512i sk = _mm512_loadu_si512((__m512i*)(sum_k + t * 16));
                        acc = _mm512_sub_epi32(acc, _mm512_mullo_epi32(correction_128, sk));
                        if (t == n_tiles - 1)
                            acc = _mm512_mask_blend_epi32(last_tile_mask, neg_inf_i, acc);
                        local_max = _mm512_max_epi32(local_max, acc);
                    }
                    int32_t tmax = _mm512_reduce_max_epi32(local_max);
                    #pragma omp critical
                    { if (tmax > global_max) global_max = tmax; }
                }
                fp_rmax = std::max(fp_rmax, (float)global_max * sq * scale_k[g]);
            }
            rowmax_out[i] = fp_rmax;
        }
    }
}

// ============================================================
// Explicit template instantiations: {MR} × {NR} × {GS}
// ============================================================
#define INSTANTIATE_VNNI(MR, NR, GS) \
    template void qk_rowmax_vnni<MR, NR, GS>( \
        const float*, const int8_t*, const float*, const int32_t*, \
        float*, size_t, size_t, size_t); \
    template void qk_rowmax_vnni_omp<MR, NR, GS>( \
        const float*, const int8_t*, const float*, const int32_t*, \
        float*, size_t, size_t, size_t);

// GS=0 (per-tensor)
INSTANTIATE_VNNI(1, 1, 0)
INSTANTIATE_VNNI(4, 1, 0)
INSTANTIATE_VNNI(4, 2, 0)
INSTANTIATE_VNNI(8, 1, 0)
INSTANTIATE_VNNI(8, 2, 0)

// GS=128
INSTANTIATE_VNNI(8, 2, 128)

// GS=512
INSTANTIATE_VNNI(8, 2, 512)

// GS=2048
INSTANTIATE_VNNI(8, 2, 2048)

// GS=4096
INSTANTIATE_VNNI(1, 1, 4096)
INSTANTIATE_VNNI(4, 1, 4096)
INSTANTIATE_VNNI(4, 2, 4096)
INSTANTIATE_VNNI(8, 1, 4096)
INSTANTIATE_VNNI(8, 2, 4096)

#undef INSTANTIATE_VNNI

}  // namespace cpu_ops
