# ao CPU GEMM 算子研究报告

> 研究 `3rdparty/ao` (torchao) 中可用于 COMPASS CPU 端 QK GEMM 加速的量化算子。

## 1. 背景

COMPASS 的 `_estimate_blocks_torch` 在 CPU 上执行 QK score 估计：

```python
# 每层, 每 Q-group (G=32):
for h in range(H):  # H=4 kv_heads
    scores = Q_h @ K_h.T   # [896, D=128] @ [D, N_kv] → [896, N_kv]
```

**参数** (Qwen2-7B): `H=4, hpg=7, F=128, D=128, G=32`, 每层调用 `G×H=128` 次 GEMM。
K cache 由 CPU offload engine 管理，可以在存储前量化以减少内存带宽。

---

## 2. ao 中可用的 CPU GEMM 算子

| # | 算子 | Q dtype | K dtype | 硬件加速 | x86 支持 |
|---|------|---------|---------|---------|---------|
| 1 | `_weight_int4pack_mm_for_cpu` | FP16/BF16/FP32 | **uint4** groupwise | tinygemm | ✅ |
| 2 | `da8w4_linear_cpu` | int8 per-token | **int4** groupwise | AVX512 VNNI | ✅ |
| 3 | `qscaled_dot_product` | int8 (uint8) | int8 (uint8) | AVX512 VNNI brgemm | ✅ |
| 4 | `float8_linear_cpu` | FP8 e4m3fn | FP8 e4m3fn | AMX / BF16 brgemm | ✅ |
| 5 | `safe_int_mm` | int8 | int8 | CPU fallback (→FP32) | ✅ (无加速) |
| 6 | `linear_8bit_act_xbit_weight` | FP32 → int8 | 1-8bit | ARM NEON dot | ❌ ARM only |

**关键结论**: #6 (`linear_8bit_act_xbit_weight`) 在 x86 上的 fallback 实现会 `throw runtime_error`，仅 ARM 可用。

---

## 3. `_weight_int4pack_mm_for_cpu` 详解

### 3.1 量化方法

| 属性 | 值 |
|------|---|
| 位宽 | 4-bit unsigned (`uint4`, 值域 `[0, 15]`) |
| 类型 | Asymmetric (非对称) |
| 粒度 | **Per-group along K 维度**；`block_size = (1, groupsize)` |
| Scale | Float (BF16/FP16)，per-group 一个 |
| Zero Point | **浮点域** (Float ZeroPointDomain) |

量化公式 (tinygemm float zero-point):
```
mid_point = (15 + 0 + 1) / 2 = 8
min_val   = zero_point - scale × mid_point
q_int4    = clamp(round((x - min_val) / scale), 0, 15)

# 反量化 (kernel 内部):
x_fp = (q_int4 - mid_point) × scale + zero_point
```

当 `groupsize = head_dim = 128` 时，退化为 **per-token** 量化。

### 3.2 数据格式

```python
# Step 1: 量化
int_data = quantize(k_fp, scale, zero_point)    # [N, K], int32, ∈[0,15]

# Step 2: Pack
packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(int_data, 1)
# → [N, K/2], uint8

# Step 3: Pack scale + zero_point
scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point)
# → [num_groups, N, 2], bfloat16

# Step 4: GEMM
output = torch.ops.aten._weight_int4pack_mm_for_cpu(
    act,            # [M, K], FP16/BF16/FP32 — 不需要量化
    packed,         # [N, K/2], uint8
    groupsize,      # int
    scale_and_zero, # [num_groups, N, 2]
)  # → [M, N]
```

---

## 4. Benchmark 结果

### 4.1 环境
- **CPU**: 32 threads
- **PyTorch**: 2.9.1+cu128
- **Shape**: `Q=[896, 128] @ K=[N, 128]^T` (模拟单个 kv_head, 单个 Q-group)

### 4.2 单次 GEMM 性能

| N (KV tokens) | FP32 (ms) | Int4 (ms) | Pack (ms) | Int4/FP32 | Int4 GFLOPS |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1,024 | 0.220 | 0.350 | 0.472 | **0.63x (慢)** | 672 |
| 2,048 | 0.385 | 0.676 | 0.526 | **0.57x (慢)** | 695 |
| 4,096 | 0.685 | 1.325 | 0.669 | **0.52x (慢)** | 709 |
| 8,192 | 1.384 | 2.624 | 0.702 | **0.53x (慢)** | 716 |
| 16,384 | 23.217 | 5.221 | 0.943 | **4.45x (快!)** | 720 |
| 32,768 | 29.474 | 21.566 | 2.206 | **1.37x (快)** | 349 |

### 4.3 COMPASS 外推 (Qwen2-7B, 28 层)

每层 cost = `G(32) × H(4) × single_gemm_time`:

| Blocks | KV tokens | FP32/层 | Int4/层 | 加速比 | FP32/模型 | Int4/模型 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 8K | 177 ms | 336 ms | 0.53x | 5.0 s | 9.4 s |
| 4 | 16K | 2972 ms | 668 ms | 4.45x | 83.2 s | 18.7 s |
| 8 | 32K | 3773 ms | 2760 ms | 1.37x | 105.6 s | 77.3 s |
| 16 | 64K | 7545 ms | 5521 ms | 1.37x | 211.3 s | 154.6 s |
| 32 | 128K | 15091 ms | 11042 ms | 1.37x | 422.5 s | 309.2 s |

### 4.4 关键分析

1. **N ≤ 8K: Int4 反而更慢 (~0.5x)**
   - `K=128` 太短，int4 解包/反量化的固定开销占主导
   - FP32 GEMM 在 `[896, 128] @ [128, 8192]` 上是 compute-bound, MKL 已经高效

2. **N = 16K: FP32 突然暴涨到 23ms, Int4 仅 5ms (4.45x)**
   - FP32 的 output `[896, 16384] × 4B ≈ 56 MB` 超出 L3 cache
   - Int4 数据量仅 25%，仍在 cache 内

3. **N = 32K: Int4 有 1.37x 加速**
   - 两者都 memory-bound，Int4 凭带宽优势保持领先

4. **Int4 quantization+pack 开销小 (0.5-2ms)**
   - 可以在 K cache offload 到 CPU 时一次性完成，跨 28 层摊销

---

## 5. 结论与建议

### 当前结论
`_weight_int4pack_mm_for_cpu` **不适合 COMPASS 的短 K 维度 (K=128) 场景**。在 N ≤ 8K 这个最常见的范围内反而更慢。只有在 N 极大(>16K) 且 memory-bound 时才有优势。

### 优化方向

| 方向 | 预期收益 | 复杂度 | 说明 |
|------|---------|--------|------|
| `da8w4_linear_cpu` | 高 | 中 | AVX512 VNNI `dpbusd` 指令，int8×int4→int32 硬件加速 |
| 减少 G (粗粒 Q-group) | 高 | 低 | G=32→8 直接减少 4x 计算量，需验证精度影响 |
| 减少 H (共享 KV 估计) | 中 | 低 | 跨 kv_head 共享分数，需验证选择质量 |
| 降低 fine_grain | 中 | 低 | F=128→256/512 减少 sub-block 数，但降低粒度 |
| C++ 扩展 fused kernel | 高 | 高 | 融合量化+GEMM+amax，避免中间 tensor 分配 |

### 测试脚本
```bash
python tests/test_int4_cpu_gemm.py
```
