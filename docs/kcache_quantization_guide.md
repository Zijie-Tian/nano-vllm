# KV Cache Quantization Implementation Guide

## 1. Overview
As part of the **Module B (CPU T-MAC Coarse Predictor)** in the long-term vision of Nano-vLLM, we implement high-efficiency KV cache quantization. While model weights are static, the KV cache is dynamic and grows with the sequence length. To enable T-MAC's multiplication-free mpGEMM on the CPU, the K-cache must be quantized and packed into a bit-serial format.

This guide focuses on the **2-bit Per-Token Quantization** for the K-cache, which is specifically designed to match the requirements of the T-MAC QGEMM operator.

## 2. Quantization Design

### 2.1 Per-Token vs. Per-Channel
- **K-Cache (Per-Token)**: The Key cache exhibits extreme outliers along specific channels. Quantizing each token (row) independently along the `head_dim` (features) dimension helps isolate these outliers and maintains high precision for attention score calculation.
- **V-Cache (Per-Channel)**: The Value cache typically has outliers distributed across different tokens. Therefore, quantizing each channel (column) independently across the sequence length is often more effective.
- **Current Implementation**: Since the CPU-side coarse predictor primarily performs **QK GEMM** to generate masks, we prioritize the **K-cache quantization**.

### 2.2 2-Bit Asymmetric Quantization
We use 2-bit quantization ($2^2 = 4$ levels) to achieve a 4x compression ratio compared to FP16.
- **Symmetric**: Maps values to $\{-1, 0, 1\}$. It's simpler but wastes one quantization level.
- **Asymmetric**: Maps values to $\{-2, -1, 0, 1\}$. This utilizes the full range of 2 bits and is generally preferred for the non-uniform distribution of K-cache activations.

**Formula**:
1.  **Scale**: $S = \frac{V_{max} - V_{min}}{Q_{max} - Q_{min}}$
2.  **Zero Point**: $ZP = Q_{min} \times S - V_{min}$
3.  **Quantize**: $Q = \text{round}(\frac{V + ZP}{S})$ clipped to $[Q_{min}, Q_{max}]$
4.  **Dequantize**: $V_{approx} = Q \times S - ZP$

## 3. Implementation Details

The implementation is located in `nanovllm/kvcache/quant/`.

### 3.1 Directory Structure
```text
nanovllm/kvcache/quant/
├── __init__.py           # Exporting core functions
└── kcache_quant.py       # PyTorch implementation of per-token quantization
```

### 3.2 Core Functions
- `quantize_kcache_per_token(k_cache, bits=2, sym=False)`:
    - Takes a float K-cache tensor of shape `[..., seq_len, head_dim]`.
    - Computes min/max along the `head_dim` dimension for each token.
    - Returns quantized `int8` tensor, `scales`, and `zero_points`.
- `dequantize_kcache_per_token(k_q, scales, zero_points, dtype)`:
    - Reconstructs the approximate float tensor for verification and reference computation.

## 4. Integration with T-MAC QGEMM
The T-MAC operator expects the quantized weights (which correspond to the K-cache in our inference scenario) to be provided in a specific format. The `quantize_kcache_per_token` function serves as the front-end for this pipeline:

1.  **Quantization**: `k_q, scales, zp = quantize_kcache_per_token(K_fp16)`
2.  **Packing (Phase 3)**: Convert `k_q` into bit-serial interleaved format.
3.  **T-MAC Execution**: CPU performs LUT-based GEMM using the packed bits and scales.

## 5. Verification
Correctness is verified through two levels of testing:
1.  **Unit Tests**: `tests/test_kcache_quant.py` verifies the mathematical correctness, range clipping, and NMSE (Normalized Mean Squared Error) of the quantization logic.
2.  **Integration Tests**: `tests/test_tvm_qgemm_e2e.py` and `tests/test_tvm_qgemm_align.py` have been updated to use `quantize_kcache_per_token` as the data source for the `PER_ROW` quantization strategy, ensuring end-to-end alignment with the T-MAC QGEMM reference implementation.

---
*Last Updated: 2026-03-09*
