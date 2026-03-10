import numpy as np
import logging
import sys
import os
from enum import Enum

# Ensure nanovllm is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm.ops.tvm_qgemm.utils.math_utils import (
    compute_sqnr,
    nmse,
)
from nanovllm.ops.tvm_qgemm.utils.quant import (
    quantize_weight_per_tensor,
    dequantize_weight_per_tensor,
    quantize_weight_per_group,
    dequantize_weight_per_group,
)
from nanovllm.ops.tvm_qgemm.utils.model_utils import preprocess_weights

# Setup logging
logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set global float display format
np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

# ============================================================================
# LUT-based GeMV (Generalized Matrix-Vector Multiplication) Implementation
#
# This code demonstrates a lookup table (LUT) based approach for efficient
# quantized matrix-vector multiplication on ARM NEON/SIMD processors.
# ============================================================================


class QuantizationStrategy(Enum):
    """Quantization strategy for weight matrices."""

    PER_TENSOR = "per-tensor"  # Single scale for entire weight matrix
    PER_GROUP = "per-group"  # K dimension split by group_size
    PER_ROW = "per-row"  # Each row has its own scale (group_size=K)


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4]
    return alphas[:bits]


np.random.seed(21)

# NOTE: Current test configuration - small matrix for validation
bits = 2  # > Quantization bit width (2-4 bits typically)
M = 128 * bits  # > Weight matrix rows (expanded by bits)
N = 1  # > Batch size
K = 128  #! Must satisfy: K >= g * kfactor && K >= act_group_size && K >= group_size
g = 4  # > Group size for LUT (4 bits = 16 entries)
bm = 128 * bits  # > M dimension tile size
simd_n_in = 16  # > SIMD input vector size (ARM NEON)
simd_n_out = 8  # > SIMD output vector size
kfactor = 16  # > K dimension tile size
act_group_size = 32  #! Must be >= g for correct grouping
group_size = 128  # > Weight quantization group size
out_dtype = "float16"  # > Output data type
dtype = "int8"  # > LUT storage type
zero_point = True  # > Whether to use zero-point quantization
m_groups = -1  # > Quantization granularity control

# Global control for quantization strategy
QUANT_STRATEGY = QuantizationStrategy.PER_TENSOR  # Change this to switch strategies
COMPARE_ALL_STRATEGIES = True  # Set to False to test only QUANT_STRATEGY

# Use asymmetric distributions for more realistic testing
weight_shape = (M // bits, K)
weight = np.zeros(weight_shape, dtype=out_dtype)
num_rows = M // bits
rows_per_pattern = num_rows // 4

for i in range(num_rows):
    if i < rows_per_pattern:
        weight[i, :] = np.random.normal(loc=0.1, scale=0.5, size=K)
    elif i < 2 * rows_per_pattern:
        weight[i, :] = np.random.normal(loc=3.0, scale=2.0, size=K)
        weight[i, weight[i, :] < 0] *= 0.1
    elif i < 3 * rows_per_pattern:
        weight[i, :] = np.random.normal(loc=-2.5, scale=1.5, size=K)
        weight[i, weight[i, :] > 0] *= 0.1
    else:
        mask = np.random.rand(K) > 0.5
        weight[i, mask] = np.random.normal(loc=4.0, scale=0.5, size=mask.sum())
        weight[i, ~mask] = np.random.normal(loc=-3.0, scale=0.5, size=(~mask).sum())

sparse_rows = np.random.choice(num_rows, size=num_rows // 8, replace=False)
for row in sparse_rows:
    sparse_mask = np.random.rand(K) < 0.7
    weight[row, sparse_mask] = 0

weight = weight.astype(out_dtype)

activation = np.random.normal(loc=-0.2, scale=1.5, size=(N, K))
activation_mask = activation > 0
activation[activation_mask] *= 0.7
activation[~activation_mask] *= 1.2
activation = activation.astype(out_dtype)

print(
    f"Overall weight stats: min={weight.min():.3f}, max={weight.max():.3f}, mean={weight.mean():.3f}, std={weight.std():.3f}"
)
print(
    f"Activation stats: min={activation.min():.3f}, max={activation.max():.3f}, mean={activation.mean():.3f}, std={activation.std():.3f}\n"
)

strategies_to_test = (
    [QUANT_STRATEGY] if not COMPARE_ALL_STRATEGIES else list(QuantizationStrategy)
)

for test_strategy in strategies_to_test:
    print(f"{'=' * 70}")
    print(f" Testing {test_strategy.value.upper()} Strategy")
    print(f"{'=' * 70}")

    group_size = 128
    sym = not zero_point

    if test_strategy == QuantizationStrategy.PER_TENSOR:
        print("Using PER-TENSOR quantization strategy")
        weight_quant, scales, zp = quantize_weight_per_tensor(
            weight, bits=bits, sym=sym
        )
        dequantized_weight = dequantize_weight_per_tensor(
            weight_quant, scales, zero_point=zp
        )
    elif test_strategy == QuantizationStrategy.PER_GROUP:
        print(f"Using PER-GROUP quantization strategy (group_size={group_size})")
        weight_quant, scales, zp = quantize_weight_per_group(
            weight, bits=bits, group_size=group_size, sym=sym
        )
        dequantized_weight = dequantize_weight_per_group(
            weight_quant, scales, group_size=group_size, zero_points=zp
        )
    elif test_strategy == QuantizationStrategy.PER_ROW:
        print("Using PER-ROW quantization strategy (each row has its own scale)")
        import torch
        from nanovllm.kvcache.quant import (
            quantize_kcache_per_token,
            dequantize_kcache_per_token,
        )

        weight_th = torch.from_numpy(weight)
        weight_quant_th, scales_th, zp_th = quantize_kcache_per_token(
            weight_th, bits=bits, sym=sym
        )
        dequantized_weight_th = dequantize_kcache_per_token(
            weight_quant_th, scales_th, zp_th, dtype=torch.float16
        )

        weight_quant = weight_quant_th.numpy()
        scales = scales_th.numpy()
        zp = zp_th.numpy() if zp_th is not None else None
        dequantized_weight = dequantized_weight_th.numpy()
        group_size = K

    print("Weight Quantization NMSE:", nmse(weight, dequantized_weight))
    print("Weight Quantization SQNR:", compute_sqnr(weight, dequantized_weight), "dB")

    qweight = weight_quant
    scale = scales * np.ones((M // bits, K // group_size))

    Aref = np.round(qweight + 2 ** (bits - 1)).astype("uint8")
    Sref = (scale * np.ones((M // bits, K // group_size))).astype(out_dtype)
    Bref = activation

    if zp is not None:
        Zref = zp * np.ones((M // bits, K // group_size)).astype(out_dtype)
    else:
        Zref = None

    if m_groups == -1:
        Adq = Aref.T.reshape(K // group_size, group_size, M // bits).astype(
            "float16"
        ) - (2 ** (bits - 1))
        Adq = Adq.transpose(1, 0, 2) * Sref.T
        if zero_point:
            Adq = Adq - Zref.T
        Adq = Adq.transpose(1, 0, 2).reshape(K, M // bits)
    else:
        Adq = (Aref.T.astype(out_dtype) - (2 ** (bits - 1))) * Sref[0]

    Y_ref = weight.dot(activation.T)
    Cref = Bref.dot(Adq)

    A_t, Scales_t = preprocess_weights(
        Aref,
        Sref,
        zeros=Zref,
        bits=bits,
        g=g,
        bm=bm,
        kfactor=kfactor,
        simd_n_in=simd_n_in,
        simd_n_out=simd_n_out,
    )

    def preprocessor_reference(B, act_group_size, g, dtype, out_dtype):
        _states = [-1, 1]
        maxv = (1 << 7) - 1
        b = B.reshape(N, K // g, g)
        codes = np.array([[i] for i in range(1 << g)], dtype=np.uint8)
        codes = np.unpackbits(codes, axis=1, bitorder="little", count=g).T
        m = np.vectorize(lambda c: _states[c])(codes).astype(out_dtype)
        lut = b.dot(m)
        lut_biases = lut.reshape(N, K // act_group_size, act_group_size // g, 1 << g)[
            :, :, :, 0
        ]
        lut_biases = np.sum(lut_biases, axis=-1)
        qlut = lut.reshape(N, K // act_group_size, act_group_size // g * (1 << g))
        absmax = np.max(np.abs(qlut), axis=-1)
        lut_scales = absmax / maxv
        ils = np.vectorize(lambda s: 1.0 / s if s != 0 else 0)(lut_scales).astype(
            out_dtype
        )
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

    Bref, LUT_Scales, LUT_Biases, QLUT = preprocessor_reference(
        Bref, act_group_size, g, dtype, out_dtype
    )

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
    ):
        _ngroups_per_elem = 8 // g
        M_bm, K_g, _ = A.shape
        M_total = M_bm * bm
        K_total = K_g * g
        alphas = get_bits_alphas(bits)
        cbits = np.zeros((N, M_total), dtype=out_dtype)
        A_unpacked = A.reshape(
            M_total // bm,
            K_total // g // kfactor,
            bm // _ngroups_per_elem // simd_n_in,
            kfactor,
            simd_n_in,
        )
        A_unpacked = np.concatenate(
            [
                (A_unpacked >> (g * ng)) & ((1 << g) - 1)
                for ng in range(_ngroups_per_elem)
            ],
            axis=-1,
        )
        if zero_point:
            scales_p = scales.reshape(
                M_total // bm,
                K_total // group_size,
                bm // bits // simd_n_out,
                2,
                simd_n_out,
            )
        else:
            scales_p = scales.reshape(
                M_total // bm,
                K_total // group_size,
                bm // bits // simd_n_out,
                simd_n_out,
            )
        for n in range(N):
            for k in range(K_total // g):
                for m in range(M_total):
                    mo, ko = m // bm, k // kfactor
                    mi, ki = (m % bm) // _ngroups_per_elem // simd_n_in, k % kfactor
                    e = (m % bm) % (_ngroups_per_elem * simd_n_in)
                    a_e = A_unpacked[mo, ko, mi, ki, e]
                    s_mi, s_e = (m % bm) // bits // simd_n_out, (m % bm) % simd_n_out
                    if m_groups == -1:
                        s = (
                            scales_p[mo, k * g // group_size, s_mi, 0, s_e]
                            if zero_point
                            else scales_p[mo, k * g // group_size, s_mi, s_e]
                        )
                    else:
                        s = scales_p[m // (M_total // m_groups)]
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
                                * scales_p[mo, k * g // group_size, s_mi, 1, s_e]
                            )
        return (
            cbits.reshape((N, M_total // simd_n_out // bits, bits, simd_n_out))
            .transpose(0, 1, 3, 2)
            .dot(np.array(alphas, dtype=out_dtype))
            .reshape((N, M_total // bits))
        )

    C = qgemm_reference(
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
    )

    print("Compute NMSE:", nmse(Cref, C))
    print("Total End-to-End NMSE:", nmse(Y_ref, C))
    print()

print("Summary: All strategies tested successfully.")
