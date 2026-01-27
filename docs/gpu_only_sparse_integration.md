# GPU-only Sparse Policy 整合

本文档记录将 sparse attention 策略整合到 GPU-only 模式的过程和性能对比。

## 背景

当前 sparse policy（Quest、XAttention）仅在 CPU offload 路径中实现。目标是将其扩展到 GPU-only 模式，以提升长上下文场景下的性能。

## 基准性能（优化前）

**测试环境**:
- GPU: NVIDIA A100-SXM4-80GB
- 模型: Llama-3.1-8B-Instruct
- 上下文长度: 32K tokens
- 日期: 2026-01-27

### Prefill Benchmark (32K context)

| 模式 | Throughput | Time | KV Cache 分配 |
|------|------------|------|---------------|
| **GPU-only (Full Attention)** | 4869.67 tok/s | 6.73s | 438 blocks (56GB GPU) |
| CPU Offload (Full Attention) | 1500.29 tok/s | 21.84s | 4 blocks GPU + 32 blocks CPU |

**性能比**: GPU-only 比 CPU Offload 快 **3.2x**

### 配置详情

**GPU-only 模式**:
```bash
CUDA_VISIBLE_DEVICES=0 python bench.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --max-len 32768
```

**CPU Offload 模式**:
```bash
CUDA_VISIBLE_DEVICES=0 python bench_offload.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --max-len 32768
```

### KV Cache 配置

| 参数 | GPU-only | CPU Offload |
|------|----------|-------------|
| block_size | 1024 tokens | 1024 tokens |
| per-token KV | 128 KB | 128 KB |
| per-block KV | 128 MB | 128 MB |
| GPU blocks | 438 | 4 |
| CPU blocks | 0 | 32 |
| Total memory | 56 GB | 4.6 GB |

## 目标

将以下 sparse policy 整合到 GPU-only 模式：

| Policy | 阶段 | 描述 |
|--------|------|------|
| Quest | Decode | Top-K block selection based on query-key scores |
| XAttention BSA | Prefill | Block sparse attention with cumulative threshold |

## 实现进度

- [ ] 分析现有 sparse policy 代码结构
- [ ] 设计 GPU-only sparse policy 接口
- [ ] 实现 GPU-only Quest decode
- [ ] 实现 GPU-only XAttention prefill
- [ ] 性能测试和对比

## 优化后性能

*待测试*

| 模式 | Throughput | Speedup vs Full |
|------|------------|-----------------|
| GPU-only + Quest (decode) | TBD | TBD |
| GPU-only + XAttn (prefill) | TBD | TBD |
