# Memory Communication Benchmark

GPU-CPU 通信量测试结果，对比 Full Policy 和 XAttention BSA Policy。

## 测试环境

- **模型**: Llama-3.1-8B-Instruct
- **GPU**: RTX 3090 (24GB)
- **配置**: `num_gpu_blocks=4`, `block_size=1024`, `enable_cpu_offload=True`
- **XAttention 参数**: `threshold=0.95`, `stride=8`

## 32K 上下文测试结果

| 指标 | Full Policy | XAttention | 比率 |
|------|-------------|------------|------|
| **Prefill H2D** | 66.57 GB | 111.12 GB | **1.67x** |
| Prefill D2H | 4.29 GB | 4.29 GB | 1.00x |
| TTFT | 8473 ms | 10367 ms | 1.22x |

### XAttention Block Selection (32K)

| 指标 | 数值 |
|------|------|
| 可用 blocks | 465 |
| 选中 blocks | 374 |
| 选择密度 | 80.4% |

## 64K 上下文测试结果

| 指标 | Full Policy | XAttention | 比率 |
|------|-------------|------------|------|
| **Prefill H2D** | 262.13 GB | 386.62 GB | **1.48x** |
| Prefill D2H | 8.46 GB | 8.46 GB | 1.00x |
| Decode H2D (32 tokens) | 262.13 GB | 262.13 GB | 1.00x |
| TTFT | 27081 ms | 33634 ms | 1.24x |

## 通信量比率对比 (K-only 优化前)

| 上下文长度 | XAttn/Full Prefill H2D 比率 |
|------------|----------------------------|
| 32K | 1.67x |
| 64K | 1.48x |

### 分析 (优化前)

1. **XAttention 通信量增加原因**：
   - Estimate 阶段：加载 **100%** 历史 blocks 的 **K+V**（用于 attention score 估计）
   - Compute 阶段：加载 **选中的** blocks（约 70-80%）
   - 理论比率：`1 + selection_density`

2. **64K 比率更低的原因**：
   - 更长上下文时，attention 分布更稀疏
   - XAttention 的 block 选择更有效（选中比例更低）
   - First/last block 强制包含的影响相对减小

3. **Decode 阶段通信量相同**：
   - XAttention 仅支持 prefill 阶段
   - Decode 阶段 fallback 到 Full Policy

---

## K-only 优化 (2026-01-28)

### 优化原理

XAttention 的 `select_blocks` 估计阶段只需要 K 来计算 attention scores：
```python
# flat_group_gemm_fuse_reshape 只使用 Q 和 K
attn_scores = flat_group_gemm_fuse_reshape(Q, K_chunk, stride, ...)
```

V 在估计阶段完全不使用，但之前代码会同时加载 K 和 V，造成 50% 通信量浪费。

### 优化实现

1. **新增方法**: `OffloadEngine.load_k_only_to_slot_layer()` - 只加载 K
2. **修改 select_blocks**: 使用只加载 K 的新方法

### 优化后测试结果

| 上下文 | Full Policy | XAttn (优化前) | XAttn (优化后) | 优化节省 |
|--------|-------------|---------------|---------------|---------|
| 32K | 66.57 GB | 111.12 GB | **79.76 GB** | **28.2%** |
| 64K | 262.13 GB | 386.62 GB | **258.78 GB** | **33.1%** |

### XAttn/Full 比率变化

| 上下文 | 优化前比率 | 优化后比率 |
|--------|-----------|-----------|
| 32K | 1.67x | **1.20x** |
| 64K | 1.48x | **0.99x** |

### 结论

优化后，64K 上下文的 XAttention 通信量与 Full Policy 基本持平 (0.99x)，
而 32K 也从 1.67x 降到 1.20x。这说明估计阶段的 K-only 优化非常有效

## 测试命令

```bash
# 32K Full Policy
python bench_offload.py --max-len 32768 --input-len 32000

# 32K XAttention
python bench_offload.py --max-len 32768 --input-len 32000 --enable-xattn

# 64K Full Policy
python bench_offload.py --max-len 65536 --input-len 64000

# 64K XAttention
python bench_offload.py --max-len 65536 --input-len 64000 --enable-xattn

# 包含 decode 测试
python bench_offload.py --max-len 65536 --input-len 64000 --bench-decode --output-len 32
```

## 相关文档

- [`observer_architecture.md`](observer_architecture.md) - Observer 架构设计
- [`xattn_bsa_policy_design.md`](xattn_bsa_policy_design.md) - XAttention BSA 算法设计
