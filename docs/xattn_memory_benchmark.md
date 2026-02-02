# XAttention Memory Benchmark

GPU-only 模式下 XAttention 的内存使用分析。

## 测试配置

### 硬件
- **GPU**: NVIDIA A100 80GB (用于基准测试)
- **目标**: 验证在 RTX 3090/4090 (24GB) 上的可行性

### 模型
- **Model**: Qwen3-0.6B (28 layers, 16 heads, 8 KV heads, head_dim=128)
- **Context Length**: 32K (max_model_len=40960)

### XAttention 配置
- **Sparse Policy**: XATTN_BSA
- **Threshold**: 0.9
- **Block Size**: 128 tokens (BSA block)
- **Stride**: 8

---

## 内存使用分析

### 基准测试 (gpu-utilization=0.9)

| 指标 | 数值 |
|------|------|
| KV Cache | 157 blocks × 448 MB = 70.3 GB |
| **峰值内存** | **73,949 MiB (72.2 GB)** |
| GPU 利用率 | 90.2% |

### 24GB 显存可行性测试

| gpu-utilization | KV Cache Blocks | KV Cache Size | 峰值内存 | 测试结果 |
|-----------------|-----------------|---------------|----------|----------|
| 0.25 | 39 blocks | 17.5 GB | **20.6 GB** | ✅ 5/5 PASSED |
| 0.28 | 44 blocks | 19.7 GB | **22.8 GB** | ✅ 5/5 PASSED |

---

## 24GB 显存推荐配置

适用于 **RTX 3090 / RTX 4090 (24GB)**：

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_ruler.py \
    --model ~/models/Qwen3-0.6B \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 5 \
    --max-model-len 40960 \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9 \
    --gpu-utilization 0.28
```

### 配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--gpu-utilization` | 0.28 | 限制 GPU 内存使用到 ~23GB |
| `--max-model-len` | 40960 | 支持 32K+ context |
| `--sparse-policy` | XATTN_BSA | 启用 XAttention 稀疏注意力 |
| `--sparse-threshold` | 0.9 | 选择覆盖 90% attention 的 blocks |

---

## 内存分解

### Qwen3-0.6B @ 32K Context

| 组件 | 计算公式 | 大小 |
|------|----------|------|
| 模型权重 | 0.6B × 2 bytes | ~1.2 GB |
| KV Cache (per-token) | 2 × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes | 112 KB |
| KV Cache (per-block) | 112 KB × 4096 tokens | 448 MB |
| KV Cache (44 blocks) | 448 MB × 44 | 19.7 GB |
| XAttention Buffers | GQA + mask + KV chunking | ~0.3 GB |
| 中间激活 | 运行时分配 | ~1.5 GB |
| **总计** | | **~22.8 GB** |

---

## 性能指标

### RULER niah_single_1 (5 samples)

| 指标 | gpu-util=0.25 | gpu-util=0.28 | gpu-util=0.9 |
|------|---------------|---------------|--------------|
| 准确率 | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| 耗时 | 11.4s | 11.5s | 11.6s |
| Compute Density | 24.77% | 24.77% | 24.77% |
| Min Layer Density | 4.29% (Layer 5) | 4.29% (Layer 5) | 4.29% (Layer 5) |

**结论**: 降低 gpu-utilization 不影响准确率和性能，只影响可支持的最大序列长度。

---

## 不同模型的估算

### KV Cache 公式

```
KV Cache per-token = 2 × num_layers × num_kv_heads × head_dim × dtype_size
KV Cache per-block = per-token × block_size
```

### 常见模型估算 (32K context, block_size=4096)

| 模型 | Layers | KV Heads | Head Dim | Per-Token | 32K Tokens | 24GB 可行? |
|------|--------|----------|----------|-----------|------------|------------|
| Qwen3-0.6B | 28 | 8 | 128 | 112 KB | 3.5 GB | ✅ 是 |
| Qwen3-4B | 36 | 8 | 128 | 144 KB | 4.5 GB | ✅ 是 |
| Llama-3.1-8B | 32 | 8 | 128 | 128 KB | 4.0 GB | ⚠️ 需要 offload |
| Qwen2.5-7B | 28 | 4 | 128 | 56 KB | 1.8 GB | ✅ 是 |

注: 8B 模型权重约 16GB，加上 KV cache 超过 24GB，需要 CPU offload。

---

## 使用建议

### RTX 3090/4090 (24GB)

1. **小模型 (≤4B)**：可直接使用 GPU-only + XAttention
   ```bash
   --gpu-utilization 0.28 --sparse-policy XATTN_BSA
   ```

2. **大模型 (7B-8B)**：需要 CPU offload
   ```bash
   --enable-offload --num-gpu-blocks 4 --num-cpu-blocks 32
   ```

### A100 (40GB/80GB)

1. **所有模型**：可使用 GPU-only 模式
   ```bash
   --gpu-utilization 0.9 --sparse-policy XATTN_BSA
   ```

---

## 相关文件

- `tests/test_ruler.py`: RULER 测试脚本
- `nanovllm/kvcache/sparse/xattn_bsa.py`: XAttention BSA Policy 实现
- `docs/gpuonly_density_alignment_test.md`: Density 对齐验证

---

**Date**: 2026-02-02
**Author**: Zijie Tian
