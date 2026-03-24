/**
 * test_qk_rowmax.cpp - Correctness + Benchmark for qk_rowmax (FP32 + VNNI)
 */

#include "cpu_ops/qk_rowmax.h"
#include <benchmark/benchmark.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void fill_random(float* data, size_t n, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}

static void ref_qk_rowmax(const float* Q, const float* K, float* rowmax,
                           size_t BQ, size_t BK, size_t D) {
    for (size_t i = 0; i < BQ; ++i) {
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < BK; ++j) {
            float dot = 0.0f;
            for (size_t d = 0; d < D; ++d)
                dot += Q[i * D + d] * K[j * D + d];
            mx = std::max(mx, dot);
        }
        rowmax[i] = mx;
    }
}

// ============================================================
// VNNI correctness
// ============================================================
template <int MR, int NR, int GS>
static float verify_vnni(size_t BQ, size_t BK, size_t D) {
    std::vector<float> Q(BQ * D), K(BK * D);
    std::vector<float> rm_ref(BQ), rm_test(BQ);
    fill_random(Q.data(), Q.size(), 42);
    fill_random(K.data(), K.size(), 43);

    size_t ng = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> sk(ng); std::vector<int32_t> sumk(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, sk.data(), sumk.data(), BK, D, GS);

    ref_qk_rowmax(Q.data(), K.data(), rm_ref.data(), BQ, BK, D);
    cpu_ops::qk_rowmax_vnni<MR, NR, GS>(Q.data(), Kv, sk.data(), sumk.data(),
                                          rm_test.data(), BQ, BK, D);
    float max_rel = 0;
    for (size_t i = 0; i < BQ; ++i) {
        float rel = (std::fabs(rm_ref[i]) > 1e-6f)
            ? std::fabs(rm_test[i] - rm_ref[i]) / std::fabs(rm_ref[i])
            : std::fabs(rm_test[i] - rm_ref[i]);
        max_rel = std::max(max_rel, rel);
        assert(rel < 0.05f && "VNNI rowmax error too large");
    }
    cpu_ops::free_aligned(Kv);
    return max_rel;
}

static void verify_correctness() {
    float e;
    e = verify_vnni<8, 2, 4096>(128, 32768, 16);
    printf("[OK] VNNI <8,2,4096> D=16 max_rel=%.3f%%\n", e * 100);
    e = verify_vnni<8, 2, 0>(128, 32768, 8);
    printf("[OK] VNNI <8,2,0>    D=8  max_rel=%.3f%%\n", e * 100);
}

// ============================================================
// Benchmarks
// ============================================================
template <int MR, int NR, int GS>
static void BM_vnni_omp(benchmark::State& state) {
    size_t BQ = state.range(0), BK = state.range(1), D = state.range(2);
    std::vector<float> Q(BQ * D), K(BK * D), rm(BQ);
    fill_random(Q.data(), Q.size(), 42); fill_random(K.data(), K.size(), 43);
    size_t ng = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> sk(ng); std::vector<int32_t> sumk(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, sk.data(), sumk.data(), BK, D, GS);
    for (auto _ : state) {
        cpu_ops::qk_rowmax_vnni_omp<MR, NR, GS>(Q.data(), Kv, sk.data(), sumk.data(),
                                                  rm.data(), BQ, BK, D);
        benchmark::DoNotOptimize(rm.data());
    }
    cpu_ops::free_aligned(Kv);
    state.counters["GOPS"] = benchmark::Counter(
        2.0 * BQ * BK * D / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_vnni_omp<8, 2, 4096>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);

int main(int argc, char** argv) {
    verify_correctness();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
