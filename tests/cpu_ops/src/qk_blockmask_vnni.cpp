/**
 * qk_blockmask_vnni.cpp - VNNI INT8 BLASST block-mask generation
 *
 * Template: <MR, NR, GS, BS, STEP_KV>
 */

#include "cpu_ops/qk_blockmask.h"
#include "cpu_ops/qk_rowmax.h"           // pack_k_vnni, num_quant_groups
#include "cpu_ops/internal/vnni_kernels.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpu_ops {

using internal::quantize_q_row;
using internal::microkernel_vnni;

// ============================================================
// Single-threaded blockmask
// ============================================================
template <int MR, int NR, int GS, int BS, int STEP_KV>
void qk_blockmask_vnni(const float* Q, const int8_t* K_vnni,
                       const float* scale_k, const int32_t* sum_k,
                       float* block_rowmax,
                       float* running_max,
                       uint8_t* block_mask,
                       float log_lambda,
                       size_t BQ, size_t BK, size_t D) {
    size_t n_tiles = (BK + 15) / 16;
    size_t d_groups = (D + 3) / 4;
    size_t d_padded = d_groups * 4;
    size_t tail_count = BK % 16;
    __mmask16 last_tile_mask = (tail_count == 0) ? (__mmask16)0xFFFF
        : (__mmask16)((1U << tail_count) - 1);
    size_t tile_stride = d_groups * 64;
    size_t actual_gs = (GS == 0) ? BK : (size_t)GS;

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;

    uint8_t q_buf[8][64];

    for (size_t kblk = 0; kblk < n_k_blocks; ++kblk) {
        size_t k_start = kblk * STEP_KV;
        size_t k_end = std::min(k_start + (size_t)STEP_KV, BK);
        size_t t_start = k_start / 16;
        size_t t_end = std::min((k_end + 15) / 16, n_tiles);

        // --- Per Q-row: compute local rowmax within this K-block ---
        size_t i = 0;
        for (; i + MR <= BQ; i += MR) {
            float sq[MR];
            const uint8_t* q_ptrs[MR];
            for (int m = 0; m < MR; ++m) {
                sq[m] = quantize_q_row(Q + (i + m) * D, q_buf[m], D, d_padded);
                q_ptrs[m] = q_buf[m];
            }

            __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
            __m512i correction_128 = _mm512_set1_epi32(128);

            size_t cur_group = (t_start * 16) / actual_gs;
            __m512i grp_rmax[MR];
            for (int m = 0; m < MR; ++m) grp_rmax[m] = neg_inf_i;

            float fp_rmax[MR];
            for (int m = 0; m < MR; ++m)
                fp_rmax[m] = -std::numeric_limits<float>::infinity();

            for (size_t t = t_start; t < t_end; ++t) {
                size_t tile_group = (t * 16) / actual_gs;
                if (tile_group != cur_group) {
                    for (int m = 0; m < MR; ++m) {
                        int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[m]);
                        fp_rmax[m] = std::max(fp_rmax[m],
                            (float)gmax * sq[m] * scale_k[cur_group]);
                        grp_rmax[m] = neg_inf_i;
                    }
                    cur_group = tile_group;
                }
                __mmask16 masks[1] = { (t == n_tiles - 1)
                    ? last_tile_mask : (__mmask16)0xFFFF };
                microkernel_vnni<MR, 1>(
                    q_ptrs, d_groups,
                    K_vnni + t * tile_stride, tile_stride,
                    sum_k + t * 16,
                    grp_rmax, masks, neg_inf_i, correction_128);
            }
            for (int m = 0; m < MR; ++m) {
                int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[m]);
                fp_rmax[m] = std::max(fp_rmax[m],
                    (float)gmax * sq[m] * scale_k[cur_group]);
                block_rowmax[(i + m) * n_k_blocks + kblk] = fp_rmax[m];
                running_max[i + m] = std::max(running_max[i + m], fp_rmax[m]);
            }
        }
        // Tail Q rows
        for (; i < BQ; ++i) {
            float sq = quantize_q_row(Q + i * D, q_buf[0], D, d_padded);
            const uint8_t* q_ptrs[1] = { q_buf[0] };
            __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
            __m512i correction_128 = _mm512_set1_epi32(128);
            size_t cur_group = (t_start * 16) / actual_gs;
            __m512i grp_rmax[1] = { neg_inf_i };
            float fp_rmax = -std::numeric_limits<float>::infinity();

            for (size_t t = t_start; t < t_end; ++t) {
                size_t tile_group = (t * 16) / actual_gs;
                if (tile_group != cur_group) {
                    int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[0]);
                    fp_rmax = std::max(fp_rmax,
                        (float)gmax * sq * scale_k[cur_group]);
                    grp_rmax[0] = neg_inf_i;
                    cur_group = tile_group;
                }
                __mmask16 masks[1] = { (t == n_tiles - 1)
                    ? last_tile_mask : (__mmask16)0xFFFF };
                microkernel_vnni<1, 1>(
                    q_ptrs, d_groups,
                    K_vnni + t * tile_stride, tile_stride,
                    sum_k + t * 16,
                    grp_rmax, masks, neg_inf_i, correction_128);
            }
            int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[0]);
            fp_rmax = std::max(fp_rmax, (float)gmax * sq * scale_k[cur_group]);
            block_rowmax[i * n_k_blocks + kblk] = fp_rmax;
            running_max[i] = std::max(running_max[i], fp_rmax);
        }

        // --- AND-reduce across BS Q-rows ---
        for (size_t qblk = 0; qblk < n_q_blocks; ++qblk) {
            size_t q_start = qblk * BS;
            size_t q_end = std::min(q_start + (size_t)BS, BQ);
            bool all_skip = true;
            for (size_t qi = q_start; qi < q_end; ++qi) {
                if ((block_rowmax[qi * n_k_blocks + kblk] - running_max[qi])
                        >= log_lambda) {
                    all_skip = false;
                    break;
                }
            }
            block_mask[qblk * n_k_blocks + kblk] = all_skip ? 0 : 1;
        }
    }
}

// ============================================================
// OMP blockmask: parallel across Q-rows, cached Q quantization
// ============================================================
template <int MR, int NR, int GS, int BS, int STEP_KV>
void qk_blockmask_vnni_omp(const float* Q, const int8_t* K_vnni,
                           const float* scale_k, const int32_t* sum_k,
                           float* block_rowmax,
                           float* running_max,
                           uint8_t* block_mask,
                           float log_lambda,
                           size_t BQ, size_t BK, size_t D) {
    size_t n_tiles = (BK + 15) / 16;
    size_t d_groups = (D + 3) / 4;
    size_t d_padded = d_groups * 4;
    size_t tail_count = BK % 16;
    __mmask16 last_tile_mask = (tail_count == 0) ? (__mmask16)0xFFFF
        : (__mmask16)((1U << tail_count) - 1);
    size_t tile_stride = d_groups * 64;
    size_t actual_gs = (GS == 0) ? BK : (size_t)GS;

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;

    // Pre-quantize all Q rows once (avoid redundant per-K-block quantization)
    std::vector<uint8_t> q_cache(BQ * d_padded);
    std::vector<float> sq_cache(BQ);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < BQ; ++i) {
        sq_cache[i] = quantize_q_row(Q + i * D,
            q_cache.data() + i * d_padded, D, d_padded);
    }

    // Process each K-block (sequential across blocks for running_max consistency)
    for (size_t kblk = 0; kblk < n_k_blocks; ++kblk) {
        size_t k_start = kblk * STEP_KV;
        size_t k_end = std::min(k_start + (size_t)STEP_KV, BK);
        size_t t_start = k_start / 16;
        size_t t_end = std::min((k_end + 15) / 16, n_tiles);

        // Parallel across Q rows — each thread processes MR-aligned chunks
        #pragma omp parallel
        {
            // Thread-local tile computation
            #pragma omp for schedule(static)
            for (size_t i_base = 0; i_base < BQ; i_base += MR) {
                int actual_mr = (int)std::min((size_t)MR, BQ - i_base);

                if (actual_mr == MR) {
                    // Full MR block
                    float sq[MR];
                    const uint8_t* q_ptrs[MR];
                    for (int m = 0; m < MR; ++m) {
                        sq[m] = sq_cache[i_base + m];
                        q_ptrs[m] = q_cache.data() + (i_base + m) * d_padded;
                    }

                    __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
                    __m512i correction_128 = _mm512_set1_epi32(128);
                    size_t cur_group = (t_start * 16) / actual_gs;
                    __m512i grp_rmax[MR];
                    for (int m = 0; m < MR; ++m) grp_rmax[m] = neg_inf_i;
                    float fp_rmax[MR];
                    for (int m = 0; m < MR; ++m)
                        fp_rmax[m] = -std::numeric_limits<float>::infinity();

                    for (size_t t = t_start; t < t_end; ++t) {
                        size_t tile_group = (t * 16) / actual_gs;
                        if (tile_group != cur_group) {
                            for (int m = 0; m < MR; ++m) {
                                int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[m]);
                                fp_rmax[m] = std::max(fp_rmax[m],
                                    (float)gmax * sq[m] * scale_k[cur_group]);
                                grp_rmax[m] = neg_inf_i;
                            }
                            cur_group = tile_group;
                        }
                        __mmask16 masks[1] = { (t == n_tiles - 1)
                            ? last_tile_mask : (__mmask16)0xFFFF };
                        microkernel_vnni<MR, 1>(
                            q_ptrs, d_groups,
                            K_vnni + t * tile_stride, tile_stride,
                            sum_k + t * 16,
                            grp_rmax, masks, neg_inf_i, correction_128);
                    }
                    for (int m = 0; m < MR; ++m) {
                        int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[m]);
                        fp_rmax[m] = std::max(fp_rmax[m],
                            (float)gmax * sq[m] * scale_k[cur_group]);
                        block_rowmax[(i_base + m) * n_k_blocks + kblk] = fp_rmax[m];
                        running_max[i_base + m] = std::max(
                            running_max[i_base + m], fp_rmax[m]);
                    }
                } else {
                    // Tail rows (< MR)
                    for (int m = 0; m < actual_mr; ++m) {
                        size_t i = i_base + m;
                        float sq = sq_cache[i];
                        const uint8_t* q_ptrs[1] = {
                            q_cache.data() + i * d_padded };
                        __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
                        __m512i correction_128 = _mm512_set1_epi32(128);
                        size_t cur_group = (t_start * 16) / actual_gs;
                        __m512i grp_rmax[1] = { neg_inf_i };
                        float fp_rmax = -std::numeric_limits<float>::infinity();

                        for (size_t t = t_start; t < t_end; ++t) {
                            size_t tile_group = (t * 16) / actual_gs;
                            if (tile_group != cur_group) {
                                int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[0]);
                                fp_rmax = std::max(fp_rmax,
                                    (float)gmax * sq * scale_k[cur_group]);
                                grp_rmax[0] = neg_inf_i;
                                cur_group = tile_group;
                            }
                            __mmask16 masks[1] = { (t == n_tiles - 1)
                                ? last_tile_mask : (__mmask16)0xFFFF };
                            microkernel_vnni<1, 1>(
                                q_ptrs, d_groups,
                                K_vnni + t * tile_stride, tile_stride,
                                sum_k + t * 16,
                                grp_rmax, masks, neg_inf_i, correction_128);
                        }
                        int32_t gmax = _mm512_reduce_max_epi32(grp_rmax[0]);
                        fp_rmax = std::max(fp_rmax,
                            (float)gmax * sq * scale_k[cur_group]);
                        block_rowmax[i * n_k_blocks + kblk] = fp_rmax;
                        running_max[i] = std::max(running_max[i], fp_rmax);
                    }
                }
            }
        }  // end omp parallel

        // AND-reduce across BS Q-rows (sequential, cheap)
        for (size_t qblk = 0; qblk < n_q_blocks; ++qblk) {
            size_t q_start = qblk * BS;
            size_t q_end = std::min(q_start + (size_t)BS, BQ);
            bool all_skip = true;
            for (size_t qi = q_start; qi < q_end; ++qi) {
                if ((block_rowmax[qi * n_k_blocks + kblk] - running_max[qi])
                        >= log_lambda) {
                    all_skip = false;
                    break;
                }
            }
            block_mask[qblk * n_k_blocks + kblk] = all_skip ? 0 : 1;
        }
    }
}

// ============================================================
// Explicit instantiations
// ============================================================
#define INSTANTIATE(MR, NR, GS, BS, STEP_KV) \
    template void qk_blockmask_vnni<MR, NR, GS, BS, STEP_KV>( \
        const float*, const int8_t*, const float*, const int32_t*, \
        float*, float*, uint8_t*, float, size_t, size_t, size_t); \
    template void qk_blockmask_vnni_omp<MR, NR, GS, BS, STEP_KV>( \
        const float*, const int8_t*, const float*, const int32_t*, \
        float*, float*, uint8_t*, float, size_t, size_t, size_t);

INSTANTIATE(8, 2, 4096, 128, 128)
INSTANTIATE(8, 2, 4096, 128, 64)
INSTANTIATE(8, 2, 4096, 64, 128)
INSTANTIATE(8, 2, 4096, 64, 64)
INSTANTIATE(8, 2, 0, 128, 128)

#undef INSTANTIATE

}  // namespace cpu_ops
