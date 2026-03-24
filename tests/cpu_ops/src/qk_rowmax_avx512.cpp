/**
 * qk_rowmax_avx512.cpp - FP32 Fused QK·rowmax with template micro-kernels
 *
 * Micro-kernel template<MR, NR>:
 *   MR Q rows × NR K-tiles(×16), D-dimension FMA loop.
 *   Loads each K vector once, broadcasts MR Q scalars → MR×NR FMAs.
 *   With NR>1: loads NR K vectors per d-step, reuses MR Q broadcasts.
 */

#include "cpu_ops/qk_rowmax.h"

#include <algorithm>
#include <immintrin.h>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpu_ops {

// ============================================================
// K packing (unchanged)
// ============================================================
void pack_k(const float* K, float* K_packed, size_t BK, size_t D) {
    size_t n_tiles = (BK + 15) / 16;
    for (size_t t = 0; t < n_tiles; ++t) {
        size_t j_base = t * 16;
        size_t j_count = std::min((size_t)16, BK - j_base);
        for (size_t d = 0; d < D; ++d) {
            float* dst = K_packed + (t * D + d) * 16;
            for (size_t j = 0; j < j_count; ++j)
                dst[j] = K[(j_base + j) * D + d];
            for (size_t j = j_count; j < 16; ++j)
                dst[j] = 0.0f;
        }
    }
}

// ============================================================
// Template micro-kernel: MR Q-rows × NR K-tiles
// ============================================================
template <int MR, int NR>
static inline __attribute__((always_inline))
void microkernel_fp32(
        const float* Q, size_t D,       // Q[MR rows, D], row stride = D
        const float* K_packed,          // NR consecutive packed tiles
        size_t tile_stride,             // = D * 16 (bytes between tile starts)
        __m512 rmax[MR],               // in/out running rowmax
        __mmask16 masks[NR],           // per-tile mask (0xFFFF for full, tail for last)
        const __m512& neg_inf) {

    // MR×NR accumulators (compiler maps to ZMM registers)
    __m512 acc[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int n = 0; n < NR; ++n)
            acc[m][n] = _mm512_setzero_ps();

    // Main D-loop: load NR K vectors, broadcast MR Q values, MR×NR FMAs
    for (size_t d = 0; d < D; ++d) {
        // Load NR K vectors (one per tile)
        __m512 k_vals[NR];
        for (int n = 0; n < NR; ++n)
            k_vals[n] = _mm512_load_ps(K_packed + n * tile_stride + d * 16);

        // Broadcast MR Q scalars and FMA against all NR K vectors
        for (int m = 0; m < MR; ++m) {
            __m512 q_val = _mm512_set1_ps(Q[m * D + d]);
            for (int n = 0; n < NR; ++n)
                acc[m][n] = _mm512_fmadd_ps(q_val, k_vals[n], acc[m][n]);
        }
    }

    // Apply tail masks and update running max
    for (int n = 0; n < NR; ++n) {
        for (int m = 0; m < MR; ++m) {
            __m512 masked = _mm512_mask_blend_ps(masks[n], neg_inf, acc[m][n]);
            rmax[m] = _mm512_max_ps(rmax[m], masked);
        }
    }
}

// ============================================================
// Template outer loop: dispatches to microkernel<MR, NR>
// Handles BQ/MR blocking with tail, n_tiles/NR blocking with tail
// ============================================================
template <int MR, int NR>
static void compute_all_rows(
        const float* Q, const float* K_packed,
        float* rowmax_out,
        size_t BQ, size_t BK, size_t D,
        size_t i_start, size_t i_end) {

    size_t n_tiles = (BK + 15) / 16;
    size_t tail_count = BK % 16;
    __mmask16 last_tile_mask = (tail_count == 0) ? 0xFFFF : ((__mmask16)((1U << tail_count) - 1));
    __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    size_t tile_stride = D * 16;

    // Process MR rows at a time
    size_t i = i_start;
    for (; i + MR <= i_end; i += MR) {
        __m512 rmax[MR];
        for (int m = 0; m < MR; ++m)
            rmax[m] = neg_inf;

        // Process NR tiles at a time
        size_t t = 0;
        for (; t + NR <= n_tiles; t += NR) {
            __mmask16 masks[NR];
            for (int n = 0; n < NR; ++n) {
                masks[n] = ((t + n) == n_tiles - 1) ? last_tile_mask : 0xFFFF;
            }
            microkernel_fp32<MR, NR>(
                Q + i * D, D,
                K_packed + t * tile_stride,
                tile_stride, rmax, masks, neg_inf);
        }
        // Remaining tiles (< NR) — use microkernel<MR, 1>
        for (; t < n_tiles; ++t) {
            __mmask16 masks[1] = { (t == n_tiles - 1) ? last_tile_mask : 0xFFFF };
            microkernel_fp32<MR, 1>(
                Q + i * D, D,
                K_packed + t * tile_stride,
                tile_stride, rmax, masks, neg_inf);
        }

        for (int m = 0; m < MR; ++m)
            rowmax_out[i + m] = _mm512_reduce_max_ps(rmax[m]);
    }
    // Remaining rows (< MR) — process one at a time
    for (; i < i_end; ++i) {
        __m512 rmax[1] = { neg_inf };
        for (size_t t = 0; t < n_tiles; ++t) {
            __mmask16 masks[1] = { (t == n_tiles - 1) ? last_tile_mask : 0xFFFF };
            microkernel_fp32<1, 1>(
                Q + i * D, D,
                K_packed + t * tile_stride,
                tile_stride, rmax, masks, neg_inf);
        }
        rowmax_out[i] = _mm512_reduce_max_ps(rmax[0]);
    }
}

// ============================================================
// Public API: single-threaded
// ============================================================
template <int MR, int NR>
void qk_rowmax_fp32(const float* Q, const float* K_packed,
                    float* rowmax_out,
                    size_t BQ, size_t BK, size_t D) {
    compute_all_rows<MR, NR>(Q, K_packed, rowmax_out, BQ, BK, D, 0, BQ);
}

// ============================================================
// Public API: adaptive OpenMP
// ============================================================
template <int MR, int NR>
void qk_rowmax_fp32_omp(const float* Q, const float* K_packed,
                        float* rowmax_out,
                        size_t BQ, size_t BK, size_t D) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    if (BQ >= (size_t)num_threads * MR) {
        // Path A: parallel along Q rows
        #pragma omp parallel
        {
            int tid = 0, nthreads = 1;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
#endif
            // Divide BQ into MR-aligned chunks per thread
            size_t rows_per_thread = ((BQ / MR + nthreads - 1) / nthreads) * MR;
            size_t i_start = std::min((size_t)tid * rows_per_thread, BQ);
            size_t i_end = std::min(i_start + rows_per_thread, BQ);

            compute_all_rows<MR, NR>(Q, K_packed, rowmax_out,
                                      BQ, BK, D, i_start, i_end);
        }
    } else {
        // Path B: parallel along K tiles (decode, BQ small)
        size_t n_tiles = (BK + 15) / 16;
        size_t tail_count = BK % 16;
        __mmask16 last_tile_mask = (tail_count == 0) ? 0xFFFF : ((__mmask16)((1U << tail_count) - 1));
        __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        size_t tile_stride = D * 16;

        for (size_t i = 0; i < BQ; ++i) {
            float global_max = -std::numeric_limits<float>::infinity();
            #pragma omp parallel
            {
                __m512 local_max = neg_inf;
                #pragma omp for schedule(static) nowait
                for (size_t t = 0; t < n_tiles; ++t) {
                    __m512 acc = _mm512_setzero_ps();
                    const float* kp = K_packed + t * tile_stride;
                    for (size_t d = 0; d < D; ++d)
                        acc = _mm512_fmadd_ps(
                            _mm512_set1_ps(Q[i * D + d]),
                            _mm512_load_ps(kp + d * 16), acc);
                    if (t == n_tiles - 1)
                        acc = _mm512_mask_blend_ps(last_tile_mask, neg_inf, acc);
                    local_max = _mm512_max_ps(local_max, acc);
                }
                float tmax = _mm512_reduce_max_ps(local_max);
                #pragma omp critical
                { if (tmax > global_max) global_max = tmax; }
            }
            rowmax_out[i] = global_max;
        }
    }
}

// ============================================================
// Explicit template instantiations
// ============================================================
#define INSTANTIATE_FP32(MR, NR) \
    template void qk_rowmax_fp32<MR, NR>( \
        const float*, const float*, float*, size_t, size_t, size_t); \
    template void qk_rowmax_fp32_omp<MR, NR>( \
        const float*, const float*, float*, size_t, size_t, size_t);

INSTANTIATE_FP32(1, 1)
INSTANTIATE_FP32(4, 1)
INSTANTIATE_FP32(4, 2)
INSTANTIATE_FP32(8, 1)
INSTANTIATE_FP32(8, 2)

#undef INSTANTIATE_FP32

}  // namespace cpu_ops

