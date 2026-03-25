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
                       const uint8_t* input_mask,
                       float scale,
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
                fp_rmax[m] *= scale;
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
            fp_rmax *= scale;
            block_rowmax[i * n_k_blocks + kblk] = fp_rmax;
            running_max[i] = std::max(running_max[i], fp_rmax);
        }

        // --- AND-reduce across BS Q-rows ---
        for (size_t qblk = 0; qblk < n_q_blocks; ++qblk) {
            // Check input_mask: skip if masked out
            if (input_mask && input_mask[qblk * n_k_blocks + kblk] == 0) {
                size_t qs = qblk * BS;
                size_t qe = std::min(qs + (size_t)BS, BQ);
                for (size_t qi = qs; qi < qe; ++qi)
                    block_rowmax[qi * n_k_blocks + kblk] = -std::numeric_limits<float>::infinity();
                block_mask[qblk * n_k_blocks + kblk] = 0;
                continue;
            }
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
// OMP blockmask v2: HPC-optimized
//   1. Loop inversion: Q-block-outer parallel (no fork/join per K-block)
//   2. Deferred horizontal reduction (FP32 vector max, single scalar reduce)
//   3. Branch-free inner loop (group chunking + last-tile peeling)
//   4. NR template unrolling (use <MR, NR> microkernel for ILP)
//   5. Dynamic padding (pad Q pointers, eliminate tail fallback)
// ============================================================
template <int MR, int NR, int GS, int BS, int STEP_KV>
void qk_blockmask_vnni_omp(const float* Q, const int8_t* K_vnni,
                           const float* scale_k, const int32_t* sum_k,
                           float* block_rowmax,
                           float* running_max,
                           uint8_t* block_mask,
                           const uint8_t* input_mask,
                           float scale,
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
    // Opt3: precompute tiles-per-group (avoid division in hot loop)
    size_t tiles_per_group = std::max<size_t>(1, actual_gs / 16);

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;

    // Opt1: outermost parallel on Q-blocks — single fork, no barrier per K-block
    #pragma omp parallel for schedule(dynamic)
    for (size_t qblk = 0; qblk < n_q_blocks; ++qblk) {
        size_t q_start = qblk * BS;
        size_t q_end = std::min(q_start + (size_t)BS, BQ);
        size_t cur_bs = q_end - q_start;

        // Opt1: tiny thread-local Q cache — fits in L1
        alignas(64) uint8_t local_q_cache[BS * 64];  // BS * d_padded (max 64)
        float local_sq[BS];
        for (size_t i = 0; i < cur_bs; ++i) {
            local_sq[i] = quantize_q_row(
                Q + (q_start + i) * D,
                local_q_cache + i * d_padded, D, d_padded);
        }

        const __m512i neg_inf_i = _mm512_set1_epi32(0x80000000);
        const __m512i correction_128 = _mm512_set1_epi32(128);
        const __m512 neg_inf_f = _mm512_set1_ps(
            -std::numeric_limits<float>::infinity());

        for (size_t kblk = 0; kblk < n_k_blocks; ++kblk) {
            // Check input_mask: skip if masked out
            if (input_mask && input_mask[qblk * n_k_blocks + kblk] == 0) {
                for (size_t i = q_start; i < q_end; ++i)
                    block_rowmax[i * n_k_blocks + kblk] = -std::numeric_limits<float>::infinity();
                block_mask[qblk * n_k_blocks + kblk] = 0;
                continue;
            }

            size_t k_start = kblk * STEP_KV;
            size_t k_end = std::min(k_start + (size_t)STEP_KV, BK);
            size_t t_start = k_start / 16;
            size_t t_end = std::min((k_end + 15) / 16, n_tiles);

            for (size_t i_base = 0; i_base < cur_bs; i_base += MR) {
                size_t actual_mr = std::min((size_t)MR, cur_bs - i_base);
                size_t global_qi = q_start + i_base;

                float sq[MR];
                const uint8_t* q_ptrs[MR];
                __m512i grp_rmax[MR];
                __m512 v_fp_rmax[MR];  // Opt2: FP32 vector accumulator

                // Opt5: dynamic padding — replicate last valid row
                for (int m = 0; m < MR; ++m) {
                    int safe_idx = std::min(m, (int)actual_mr - 1);
                    sq[m] = local_sq[i_base + safe_idx];
                    q_ptrs[m] = local_q_cache + (i_base + safe_idx) * d_padded;
                    grp_rmax[m] = neg_inf_i;
                    v_fp_rmax[m] = neg_inf_f;
                }

                // Opt3: chunk by quantization group boundaries
                size_t t = t_start;
                while (t < t_end) {
                    size_t cur_group = t / tiles_per_group;
                    size_t next_group_t = (cur_group + 1) * tiles_per_group;
                    size_t chunk_end = std::min(t_end, next_group_t);
                    // Separate safe region (full mask) from potential last-tile
                    size_t chunk_safe = (chunk_end == n_tiles) ? chunk_end - 1
                                                               : chunk_end;

                    __mmask16 full_masks[NR];
                    for (int n = 0; n < NR; ++n) full_masks[n] = 0xFFFF;

                    // Opt4: NR-unrolled main loop (branch-free, full mask)
                    for (; t + NR <= chunk_safe; t += NR) {
                        microkernel_vnni<MR, NR>(
                            q_ptrs, d_groups,
                            K_vnni + t * tile_stride, tile_stride,
                            sum_k + t * 16,
                            grp_rmax, full_masks, neg_inf_i, correction_128);
                    }
                    // NR-remainder (still full mask, no branch)
                    for (; t < chunk_safe; ++t) {
                        microkernel_vnni<MR, 1>(
                            q_ptrs, d_groups,
                            K_vnni + t * tile_stride, tile_stride,
                            sum_k + t * 16,
                            grp_rmax, full_masks, neg_inf_i, correction_128);
                    }

                    // Opt3: peeled last tile with tail mask
                    if (t < chunk_end) {
                        __mmask16 tail_masks[1] = { (t == n_tiles - 1)
                            ? last_tile_mask : (__mmask16)0xFFFF };
                        microkernel_vnni<MR, 1>(
                            q_ptrs, d_groups,
                            K_vnni + t * tile_stride, tile_stride,
                            sum_k + t * 16,
                            grp_rmax, tail_masks, neg_inf_i, correction_128);
                        ++t;
                    }

                    // Opt2: vector-width FP32 accumulation (no scalar reduce)
                    for (int m = 0; m < MR; ++m) {
                        __m512 v_gmax = _mm512_cvtepi32_ps(grp_rmax[m]);
                        __m512 v_scale = _mm512_set1_ps(
                            sq[m] * scale_k[cur_group]);
                        v_fp_rmax[m] = _mm512_max_ps(v_fp_rmax[m],
                            _mm512_mul_ps(v_gmax, v_scale));
                        grp_rmax[m] = neg_inf_i;
                    }
                }  // end group chunking

                // Opt2: single scalar horizontal reduce per K-block (only here)
                for (size_t m = 0; m < actual_mr; ++m) {
                    float fp_max = _mm512_reduce_max_ps(v_fp_rmax[m]) * scale;
                    block_rowmax[(global_qi + m) * n_k_blocks + kblk] = fp_max;
                    running_max[global_qi + m] = std::max(
                        running_max[global_qi + m], fp_max);
                }
            }  // end MR blocks

            // Opt1 bonus: mask computed in-place on hot cache, no extra barrier
            bool all_skip = true;
            for (size_t qi = q_start; qi < q_end; ++qi) {
                if ((block_rowmax[qi * n_k_blocks + kblk] - running_max[qi])
                        >= log_lambda) {
                    all_skip = false;
                    break;
                }
            }
            block_mask[qblk * n_k_blocks + kblk] = all_skip ? 0 : 1;
        }  // end K-blocks
    }  // end OMP parallel for
}

// ============================================================
// Explicit instantiations
// ============================================================
#define INSTANTIATE(MR, NR, GS, BS, STEP_KV) \
    template void qk_blockmask_vnni<MR, NR, GS, BS, STEP_KV>( \
        const float*, const int8_t*, const float*, const int32_t*, \
        float*, float*, uint8_t*, const uint8_t*, float, float, \
        size_t, size_t, size_t); \
    template void qk_blockmask_vnni_omp<MR, NR, GS, BS, STEP_KV>( \
        const float*, const int8_t*, const float*, const int32_t*, \
        float*, float*, uint8_t*, const uint8_t*, float, float, \
        size_t, size_t, size_t);

INSTANTIATE(8, 2, 4096, 128, 128)
INSTANTIATE(8, 2, 4096, 128, 64)
INSTANTIATE(8, 2, 4096, 64, 128)
INSTANTIATE(8, 2, 4096, 64, 64)
INSTANTIATE(8, 2, 0, 128, 128)

#undef INSTANTIATE

}  // namespace cpu_ops
