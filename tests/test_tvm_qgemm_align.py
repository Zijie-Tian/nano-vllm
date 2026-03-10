#!/usr/bin/env python3
"""Verify QGeMMLUTBitsCodegen output against reference implementation.

Tests LUT-based quantized GEMM with:
1. QGeMMLUTBitsPreprocessorCodegen - LUT construction
2. QGeMMLUTBitsCodegen - LUT-based matrix multiplication
"""


import numpy as np
import torch
import tvm

from nanovllm.kvcache.quant import (
    quantize_kcache_per_token,
    pack_kvcache_tmac,
    pack_scales_tmac,
)
from nanovllm.ops.tvm_qgemm.qgemm import (
    QGeMMLUTBitsCodegen,
    QGeMMLUTBitsPreprocessorCodegen,
)
from nanovllm.ops.tvm_qgemm.utils.math_utils import nmse

# =============================================================================
# Configuration
# =============================================================================
target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
cc_opts = ["-O3", "-march=native", "-mllvm", "-inline-threshold=10000"]
out_dtype = "float32"

bits = 2
M = 128 * bits
N = 1
K = 128
g = 4
simd_n_in = 16
simd_n_out = 8
act_group_size = 64  # Must be divisible by 32 for partial_max
group_size = 128
dtype = "int8"
zero_point = True
m_groups = -1

np.random.seed(21)
np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

print(f"Config: bits={bits}, M={M}, N={N}, K={K}, target={target}")
print("=" * 60)

# =============================================================================
# Step 1: First compile codegen to get actual kfactor and bm
# =============================================================================
print("\n[Step 1] Compiling QGeMMLUTBitsCodegen to get config...")

codegen = QGeMMLUTBitsCodegen(
    dtype=dtype,
    target=target,
    name="qgemm_lut",
    tune=False,
    verify=False,
    save_dir="build/test_qgemm",
    num_threads=1,
    cc_opts=cc_opts,
    bits=bits,
    g=g,
    group_size=group_size,
    act_group_size=act_group_size,
    out_dtype=out_dtype,
    m_groups=m_groups,
    zero_point=zero_point,
)

func_qgemm, _ = codegen.compile(M, N, K, return_type="mod")

# Get actual config values from codegen
kfactor = codegen.kfactor
bm = codegen.bm
print(f"  Actual config: bm={bm}, kfactor={kfactor}")


# =============================================================================
# Reference implementations
# =============================================================================
def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4]
    return alphas[:bits]


def preprocessor_reference(B, act_group_size, g, dtype, out_dtype):
    """Generate LUT for activation values."""
    _states = [-1, 1]
    _gamma = 1
    maxv = (1 << 7) - 1

    N, K = B.shape
    b = B.reshape(N, K // g, g)

    codes = np.array([[i] for i in range(1 << g)], dtype=np.uint8)
    codes = np.unpackbits(codes, axis=1, bitorder="little", count=g).T

    def map_states(c):
        return _states[c]

    m = np.vectorize(map_states)(codes).astype(out_dtype)

    lut = b.dot(m)

    lut_biases = lut.reshape(N, K // act_group_size, act_group_size // g, 1 << g)[
        :, :, :, 0
    ]
    lut_biases = np.sum(lut_biases, axis=-1) * _gamma

    qlut = lut.reshape(N, K // act_group_size, act_group_size // g * (1 << g))
    absmax = np.max(np.abs(qlut), axis=-1)
    lut_scales = absmax / maxv

    def recp(s):
        return 1.0 / s if s != 0 else 0

    ils = np.vectorize(recp)(lut_scales).astype(out_dtype)

    qlut = np.rint(
        (
            qlut.transpose(2, 0, 1).reshape(-1, qlut.shape[0] * qlut.shape[1])
            * ils.reshape(1, qlut.shape[0] * qlut.shape[1])
        )
        .reshape(qlut.shape[2], qlut.shape[0], qlut.shape[1])
        .transpose(1, 2, 0)
        .reshape(N, K // g, 1 << g)
    ).astype(dtype)

    return B, lut_scales, lut_biases, qlut


def qgemm_reference(
    A,
    QLUT,
    LUT_Scales,
    LUT_Biases,
    scales,
    bits,
    g,
    group_size,
    m_groups,
    simd_n_in,
    simd_n_out,
    bm,
    kfactor,
    zero_point,
    out_dtype,
    act_group_size,
):
    """Reference implementation for LUT-based quantized GEMM."""
    _ngroups_per_elem = 8 // g
    alphas = get_bits_alphas(bits)

    M_bm, K_g, A_cols = A.shape
    M = M_bm * bm
    K = K_g * g
    N, _, lut_size = QLUT.shape

    cbits = np.zeros((N, M), dtype=out_dtype)

    A = A.reshape(
        M // bm,
        K // g // kfactor,
        bm // _ngroups_per_elem // simd_n_in,
        kfactor,
        simd_n_in,
    )
    A = np.concatenate(
        [(A >> (g * ng)) & ((1 << g) - 1) for ng in range(_ngroups_per_elem)], axis=-1
    )

    if zero_point:
        scales = scales.reshape(
            M // bm, K // group_size, bm // bits // simd_n_out, 2, simd_n_out
        )
    else:
        scales = scales.reshape(
            M // bm, K // group_size, bm // bits // simd_n_out, simd_n_out
        )

    for n in range(N):
        for k in range(K // g):
            for m in range(M):
                mo = m // bm
                ko = k // kfactor
                mi = (m % bm) // _ngroups_per_elem // simd_n_in
                ki = k % kfactor
                e = (m % bm) % (_ngroups_per_elem * simd_n_in)
                a_e = A[mo, ko, mi, ki, e]

                scales_mi = (m % bm) // bits // simd_n_out
                scales_e = (m % bm) % simd_n_out

                if m_groups == -1:
                    if zero_point:
                        s = scales[mo, k * g // group_size, scales_mi, 0, scales_e]
                    else:
                        s = scales[mo, k * g // group_size, scales_mi, scales_e]
                else:
                    m_group_size = M // m_groups
                    s = scales[m // m_group_size]

                cbits[n, m] += (
                    QLUT[n, k, a_e] * LUT_Scales[n, k * g // act_group_size] * s
                )

                if (((k * g) % act_group_size) == 0) and (
                    (((m % bm) // simd_n_out) % bits) == 0
                ):
                    cbits[n, m] += LUT_Biases[n, k * g // act_group_size] * s
                    if zero_point:
                        cbits[n, m] += (
                            LUT_Biases[n, k * g // act_group_size]
                            * (1 / alphas[0])
                            * scales[mo, k * g // group_size, scales_mi, 1, scales_e]
                        )

    c = (
        cbits.reshape((N, M // simd_n_out // bits, bits, simd_n_out))
        .transpose(0, 1, 3, 2)
        .dot(np.array(alphas, dtype=out_dtype))
        .reshape((N, M // bits))
    )

    return c


# =============================================================================
# Prepare test data
# =============================================================================
weight = np.random.randn(M // bits, K).astype(out_dtype)
activation = np.random.randn(N, K).astype(out_dtype)

sym = not zero_point

weight_th = torch.from_numpy(weight)
weight_quant_th, scales_th, zp_th = quantize_kcache_per_token(
    weight_th, bits=bits, sym=sym
)
weight_quant = weight_quant_th.numpy()
scales = scales_th.numpy()
zp = zp_th.numpy() if zp_th is not None else None

Aref = np.round(weight_quant + 2 ** (bits - 1)).astype("uint8")
Sref = (scales * np.ones((M // bits, K // group_size))).astype(out_dtype)
Bref = activation
if zp is not None:
    Zref = (zp * np.ones((M // bits, K // group_size))).astype(out_dtype)
else:
    Zref = None

# Preprocess weights using the new PyTorch interface
A_t_torch = pack_kvcache_tmac(
    weight_quant_th.unsqueeze(0).unsqueeze(0),
    is_key=True,
    bits=bits,
    g=g,
    bm=bm,
    kfactor=kfactor,
).squeeze(0).squeeze(0)

Scales_t_torch = pack_scales_tmac(
    torch.from_numpy(Sref),
    zeros=torch.from_numpy(Zref) if Zref is not None else None,
    bits=bits,
    bm=bm,
    simd_n_out=simd_n_out,
).squeeze(0).squeeze(0)

A_t = A_t_torch.numpy()
Scales_t = Scales_t_torch.numpy()

# Generate reference LUT
Bref, LUT_Scales, LUT_Biases, QLUT = preprocessor_reference(
    Bref, act_group_size, g, dtype, out_dtype
)

# Compute reference result
C_ref = qgemm_reference(
    A_t,
    QLUT,
    LUT_Scales,
    LUT_Biases,
    Scales_t,
    bits,
    g,
    group_size,
    m_groups,
    simd_n_in,
    simd_n_out,
    bm,
    kfactor,
    zero_point,
    out_dtype,
    act_group_size,
)

print(f"Reference output shape: {C_ref.shape}")
print(f"Reference output sample: {C_ref.flatten()[:5]}")

# =============================================================================
# Test 1: QGeMMLUTBitsPreprocessorCodegen
# =============================================================================
print("\n[Test 1] QGeMMLUTBitsPreprocessorCodegen")

preprocessor = QGeMMLUTBitsPreprocessorCodegen(
    dtype=dtype,
    target=target,
    name="qgemm_preproc",
    tune=False,
    verify=False,
    save_dir="build/test_qgemm",
    num_threads=1,
    cc_opts=cc_opts,
    g=g,
    act_group_size=act_group_size,
    out_dtype=out_dtype,
    bits=bits,
    M=M,
)

func_preproc, _ = preprocessor.compile(N, K, return_type="mod")

dev = tvm.device(target, 0)
B_tvm = tvm.nd.array(activation.astype(out_dtype), dev)
LUT_Scales_tvm = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
LUT_Biases_tvm = tvm.nd.array(LUT_Biases.astype(out_dtype), dev)
QLUT_tvm = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype=dtype), dev)

func_preproc(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)

lut_scales_result = LUT_Scales_tvm.numpy()
qlut_result = QLUT_tvm.numpy()

lut_scales_nmse = nmse(LUT_Scales, lut_scales_result)
qlut_nmse = nmse(QLUT.astype(np.float32), qlut_result.astype(np.float32))

print(f"  LUT_Scales NMSE: {lut_scales_nmse:.6f}")
print(f"  QLUT NMSE: {qlut_nmse:.6f}")

if lut_scales_nmse < 1e-4 and qlut_nmse < 1e-2:
    print("  PASSED")
else:
    print("  FAILED")

# =============================================================================
# Test 2: QGeMMLUTBitsCodegen
# =============================================================================
print("\n[Test 2] QGeMMLUTBitsCodegen")

# Use the already compiled func_qgemm from Step 1
A_tvm = tvm.nd.array(A_t, dev)
QLUT_input = tvm.nd.array(QLUT.astype(dtype), dev)
Scales_input = tvm.nd.array(Scales_t.astype(out_dtype), dev)
LUT_Scales_input = tvm.nd.array(LUT_Scales.astype(out_dtype), dev)
LUT_Biases_input = tvm.nd.array(LUT_Biases.astype(out_dtype), dev)
C_tvm = tvm.nd.array(np.zeros((N, M // bits), dtype=out_dtype), dev)

func_qgemm(A_tvm, QLUT_input, Scales_input, LUT_Scales_input, LUT_Biases_input, C_tvm)
C_result = C_tvm.numpy()

print(f"  TVM output sample: {C_result.flatten()[:5]}")
print(f"  Ref output sample: {C_ref.flatten()[:5]}")

qgemm_nmse = nmse(C_ref.astype(np.float32), C_result.astype(np.float32))
print(f"  NMSE vs reference: {qgemm_nmse:.6f}")

try:
    np.testing.assert_allclose(C_result, C_ref, rtol=1e-2, atol=1e-2)
    print("  PASSED")
except AssertionError as e:
    print(f"  FAILED: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary: LUT-based QGeMM test completed")
print(f"  Preprocessor NMSE: lut_scales={lut_scales_nmse:.6f}, qlut={qlut_nmse:.6f}")
print(f"  QGeMM NMSE: {qgemm_nmse:.6f}")
