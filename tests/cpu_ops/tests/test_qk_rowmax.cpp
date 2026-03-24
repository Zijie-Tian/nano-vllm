/**
 * test_qk_rowmax.cpp - Correctness + Benchmark for FP32 and VNNI<MR,NR,GS>
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
// Correctness
// ============================================================
template <int MR, int NR>
static void verify_fp32(size_t BQ, size_t BK, size_t D, const char* tag) {
    std::vector<float> Q(BQ * D), K(BK * D);
    std::vector<float> rm_ref(BQ), rm_test(BQ);
    fill_random(Q.data(), Q.size(), 42);
    fill_random(K.data(), K.size(), 43);
    float* Kp = cpu_ops::alloc_aligned(cpu_ops::pack_k_buffer_size(BK, D));
    cpu_ops::pack_k(K.data(), Kp, BK, D);
    ref_qk_rowmax(Q.data(), K.data(), rm_ref.data(), BQ, BK, D);
    cpu_ops::qk_rowmax_fp32<MR, NR>(Q.data(), Kp, rm_test.data(), BQ, BK, D);
    for (size_t i = 0; i < BQ; ++i) {
        float t = 1e-2f + 1e-3f * std::fabs(rm_ref[i]);
        assert(std::fabs(rm_test[i] - rm_ref[i]) < t && "FP32 mismatch");
    }
    cpu_ops::free_aligned(Kp);
}

template <int MR, int NR, int GS>
static float verify_vnni(size_t BQ, size_t BK, size_t D, const char* tag) {
    std::vector<float> Q(BQ * D), K(BK * D);
    std::vector<float> rm_ref(BQ), rm_test(BQ), rm_omp(BQ);
    fill_random(Q.data(), Q.size(), 42);
    fill_random(K.data(), K.size(), 43);

    size_t n_groups = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> scale_k(n_groups);
    std::vector<int32_t> sum_k(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, scale_k.data(), sum_k.data(), BK, D, GS);

    ref_qk_rowmax(Q.data(), K.data(), rm_ref.data(), BQ, BK, D);
    cpu_ops::qk_rowmax_vnni<MR, NR, GS>(Q.data(), Kv, scale_k.data(), sum_k.data(),
                                          rm_test.data(), BQ, BK, D);
    cpu_ops::qk_rowmax_vnni_omp<MR, NR, GS>(Q.data(), Kv, scale_k.data(), sum_k.data(),
                                              rm_omp.data(), BQ, BK, D);

    float max_rel = 0;
    for (size_t i = 0; i < BQ; ++i) {
        assert(!std::isnan(rm_test[i]) && "VNNI NaN");
        float rel = (std::fabs(rm_ref[i]) > 1e-6f)
            ? std::fabs(rm_test[i] - rm_ref[i]) / std::fabs(rm_ref[i])
            : std::fabs(rm_test[i] - rm_ref[i]);
        max_rel = std::max(max_rel, rel);
        assert(rel < 0.05f && "VNNI excessive error");
        assert(std::fabs(rm_omp[i] - rm_test[i]) < 1e-6f && "OMP mismatch");
    }
    cpu_ops::free_aligned(Kv);
    return max_rel;
}

static void verify_correctness() {
    verify_fp32<4, 1>(128, 200, 16, "fp32");
    verify_fp32<8, 2>(100, 128, 8,  "fp32");
    printf("[OK] FP32 correctness.\n");

    // VNNI GS sweep — all use BK=32768 for meaningful grouping
    struct { int gs; const char* name; } gs_tests[] = {
        {0, "per-tensor"}, {128, "128"}, {512, "512"}, {2048, "2048"}, {4096, "4096"}
    };
    // GS=0
    float e0_16 = verify_vnni<8, 2, 0>(128, 32768, 16, "GS=0");
    float e0_8  = verify_vnni<8, 2, 0>(128, 32768, 8,  "GS=0");
    printf("[OK] VNNI GS=0     D16=%.3f%% D8=%.3f%%\n", e0_16*100, e0_8*100);
    // GS=128
    float e128_16 = verify_vnni<8, 2, 128>(128, 32768, 16, "GS=128");
    float e128_8  = verify_vnni<8, 2, 128>(128, 32768, 8,  "GS=128");
    printf("[OK] VNNI GS=128   D16=%.3f%% D8=%.3f%%\n", e128_16*100, e128_8*100);
    // GS=512
    float e512_16 = verify_vnni<8, 2, 512>(128, 32768, 16, "GS=512");
    float e512_8  = verify_vnni<8, 2, 512>(128, 32768, 8,  "GS=512");
    printf("[OK] VNNI GS=512   D16=%.3f%% D8=%.3f%%\n", e512_16*100, e512_8*100);
    // GS=2048
    float e2k_16 = verify_vnni<8, 2, 2048>(128, 32768, 16, "GS=2048");
    float e2k_8  = verify_vnni<8, 2, 2048>(128, 32768, 8,  "GS=2048");
    printf("[OK] VNNI GS=2048  D16=%.3f%% D8=%.3f%%\n", e2k_16*100, e2k_8*100);
    // GS=4096
    float e4k_16 = verify_vnni<8, 2, 4096>(128, 32768, 16, "GS=4096");
    float e4k_8  = verify_vnni<8, 2, 4096>(128, 32768, 8,  "GS=4096");
    printf("[OK] VNNI GS=4096  D16=%.3f%% D8=%.3f%%\n", e4k_16*100, e4k_8*100);
}

// ============================================================
// Benchmark templates
// ============================================================
template <int MR, int NR>
static void BM_fp32_1t(benchmark::State& state) {
    size_t BQ = state.range(0), BK = state.range(1), D = state.range(2);
    std::vector<float> Q(BQ * D), K(BK * D), rm(BQ);
    fill_random(Q.data(), Q.size(), 42); fill_random(K.data(), K.size(), 43);
    float* Kp = cpu_ops::alloc_aligned(cpu_ops::pack_k_buffer_size(BK, D));
    cpu_ops::pack_k(K.data(), Kp, BK, D);
    for (auto _ : state) {
        cpu_ops::qk_rowmax_fp32<MR, NR>(Q.data(), Kp, rm.data(), BQ, BK, D);
        benchmark::DoNotOptimize(rm.data());
    }
    cpu_ops::free_aligned(Kp);
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * BQ * BK * D / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

template <int MR, int NR>
static void BM_fp32_omp(benchmark::State& state) {
    size_t BQ = state.range(0), BK = state.range(1), D = state.range(2);
    std::vector<float> Q(BQ * D), K(BK * D), rm(BQ);
    fill_random(Q.data(), Q.size(), 42); fill_random(K.data(), K.size(), 43);
    float* Kp = cpu_ops::alloc_aligned(cpu_ops::pack_k_buffer_size(BK, D));
    cpu_ops::pack_k(K.data(), Kp, BK, D);
    for (auto _ : state) {
        cpu_ops::qk_rowmax_fp32_omp<MR, NR>(Q.data(), Kp, rm.data(), BQ, BK, D);
        benchmark::DoNotOptimize(rm.data());
    }
    cpu_ops::free_aligned(Kp);
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * BQ * BK * D / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

template <int MR, int NR, int GS>
static void BM_vnni_1t(benchmark::State& state) {
    size_t BQ = state.range(0), BK = state.range(1), D = state.range(2);
    std::vector<float> Q(BQ * D), K(BK * D), rm(BQ);
    fill_random(Q.data(), Q.size(), 42); fill_random(K.data(), K.size(), 43);
    size_t ng = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> sk(ng); std::vector<int32_t> sumk(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, sk.data(), sumk.data(), BK, D, GS);
    for (auto _ : state) {
        cpu_ops::qk_rowmax_vnni<MR, NR, GS>(Q.data(), Kv, sk.data(), sumk.data(),
                                              rm.data(), BQ, BK, D);
        benchmark::DoNotOptimize(rm.data());
    }
    cpu_ops::free_aligned(Kv);
    state.counters["GOPS"] = benchmark::Counter(
        2.0 * BQ * BK * D / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

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

// ============================================================
// Registrations
// ============================================================

// 1T baselines (best configs)
BENCHMARK(BM_fp32_1t<8, 2>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);
BENCHMARK(BM_vnni_1t<4, 2, 0>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);
BENCHMARK(BM_vnni_1t<4, 2, 4096>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);

// OMP: FP32 best
BENCHMARK(BM_fp32_omp<8, 2>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);

// OMP VNNI: GS sweep with best MR×NR = <8,2>
BENCHMARK(BM_vnni_omp<8, 2, 0>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);
BENCHMARK(BM_vnni_omp<8, 2, 128>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);
BENCHMARK(BM_vnni_omp<8, 2, 512>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);
BENCHMARK(BM_vnni_omp<8, 2, 2048>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);
BENCHMARK(BM_vnni_omp<8, 2, 4096>)->Args({4096, 32768, 16})->Args({4096, 32768, 8})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);

int main(int argc, char** argv) {
    verify_correctness();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
