# 64k 推理内存分析

本文档分析 Llama 3.1 8B 模型在 64k 长度推理时的内存占用，以及 RTX 3090 (24GB) 上的 OOM 问题。

## 模型配置

```python
hidden_size = 4096
intermediate_size = 14336
num_layers = 32
num_heads = 32
num_kv_heads = 8
head_dim = 128
seq_len = 65536
dtype = bfloat16 (2 bytes)
```

## 理论内存占用

### GPU Only 模式

| 组件 | 计算公式 | 内存占用 |
|------|----------|----------|
| 模型权重 | 8.03B × 2 bytes | **16.06 GB** |
| KV Cache | 32 × 65536 × 8 × 128 × 2 × 2 | **8.19 GB** |
| Prefill 激活值峰值 | max(QKV, MLP) | **~2 GB** |
| **总计** | | **~26 GB** |

**结论**：GPU only 模式需要 ~26 GB，**RTX 3090 (24GB) 无法运行**。

### CPU Offload 模式

| 组件 | 计算公式 | 内存占用 |
|------|----------|----------|
| 模型权重 | 8.03B × 2 bytes | **16.06 GB** |
| Ring buffer | num_kv_buffers × seq_len × 128 KB/token | 258-1034 MB |
| GPU KV blocks | num_gpu_blocks × block_size × 128 KB/token | 256 MB (2 blocks) |
| Per-layer decode buffer | 32 layers × 缓冲 | 128 MB |
| 激活值峰值 (chunked) | chunk_size × hidden_size × 2 | ~50 MB |
| PyTorch 开销 | CUDA 上下文 + 碎片 | ~5-6 GB |
| **理论小计** | | **~17.5 GB** |
| **实际需求** | | **~23 GB** |

**配置参数**：
- `num_kv_buffers`: Ring buffer 大小 (1-4)，默认 4
- `num_gpu_blocks`: GPU 上的 KV cache block 数量
- `block_size`: 每个 block 的 token 数

## OOM 问题分析

### 实际观测（RTX 3090, num_kv_buffers=1）

```
PyTorch allocated:     22.49 GB
PyTorch reserved:      429 MB
Free:                  306 MB
Total available:       735 MB
Failed to allocate:    508 MB (torch.cat)
```

### 内存碎片来源

| 来源 | 说明 | 影响 |
|------|------|------|
| Binned 分配器 | PyTorch 使用固定大小的内存池 | 中等 |
| torch.compile 缓存 | 编译后的 kernel 代码和常量 | 高 (~2-3 GB) |
| 频繁分配/释放 | chunked 处理中每个 chunk 的创建销毁 | 高 |
| 不同大小张量 | (128,4096), (65536,6144) 等 | 中等 |

### torch.cat 内存需求

Chunked MLP 处理（chunk_size=128）：
```
65536 / 128 = 512 chunks
每个 chunk 输出: (128, 4096) × 2 bytes = 1 MB
torch.cat 拼接需要: (65536, 4096) × 2 bytes = 508 MB (连续)
```

## 已尝试的优化

| 优化项 | 效果 |
|--------|------|
| 移除 `@torch.compile` | PyTorch: 23.13 → 22.80 GB (-300 MB) |
| 减少 `num_kv_buffers` (4→1) | Ring buffer: 1034 → 258 MB (-776 MB) |
| Chunked QKV/MLP/LayerNorm | 峰值激活: ~2 GB → ~50 MB |
| 降低 GPU 利用率 (0.9→0.75) | 无明显效果 |
| 减小 chunk_size (4096→128) | 峰值降低，但 torch.cat 需要连续内存 |

### 最终状态

```
理论需求:    ~17.5 GB
实际分配:    22.49 GB
剩余空间:    735 MB (306 MB + 429 MB reserved)
分配失败:    508 MB (torch.cat 需要连续内存)
```

## 结论

### 根本原因

**不是绝对内存不足，而是内存碎片导致的分配失败**。

理论需求 17.5 GB < 24 GB，但由于：
- PyTorch 开销（CUDA 上下文、碎片）：~5-6 GB
- torch.compile 缓存：~2-3 GB（已移除）
- 内存碎片导致无法分配 508 MB 连续块

### 硬件限制

| GPU | 显存 | 64k GPU Only | 64k Offload |
|-----|------|--------------|--------------|
| RTX 3090 | 24 GB | ❌ | ⚠️ 碎片问题 |
| RTX 4090 | 24 GB | ❌ | ⚠️ 碎片问题 |
| A100 | 40 GB | ✅ | ✅ |
| A100 | 80 GB | ✅ | ✅ |

### 建议

1. **64k 推理建议使用 40GB+ 显存的 GPU**
2. RTX 3090/4090 适合 32k 或更短的场景
3. 如必须在 24GB GPU 上运行 64k：
   - 使用 RAPIDS RMM 分配器
   - 预分配 torch.cat 需要的内存
   - 或使用流式处理避免 torch.cat

## 参考

- [PyTorch 内存管理文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html)
- [PyTorch 内存碎片讨论](https://discuss.pytorch.org/t/how-to-reduce-memory-fragmentation-when-enable-expandable-segments/221805)
- [STWeaver - 减少 79% 内存碎片](https://arxiv.org/html/2507.16274v1)
