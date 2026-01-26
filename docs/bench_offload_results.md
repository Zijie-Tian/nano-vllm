# CPU Offload Benchmark Results

本文档记录 `bench_offload.py` 在不同配置下的性能测试结果。

## 测试环境

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA A100-SXM4-80GB |
| 模型 | Llama-3.1-8B-Instruct |
| GPU slots | 4 |
| Block size | 1024 tokens |
| Chunk size | 2048 tokens |

## Sparse Policy 配置

| 策略 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| FULL | Full Attention | Full Attention | 基线，加载所有 blocks |
| XATTN_BSA | XAttention (tau=0.95, stride=8) | Full Attention (fallback) | 稀疏 prefill |

## 测试结果

### 32K 上下文

| 策略 | 输入长度 | 耗时 | 吞吐量 | 相对性能 |
|------|----------|------|--------|----------|
| Full Attention | 32767 tok | 20.64s | **1587.74 tok/s** | baseline |
| XAttention BSA | 32767 tok | 27.95s | **1172.33 tok/s** | 0.74x |

### 128K 上下文

| 策略 | 输入长度 | 耗时 | 吞吐量 | 相对性能 |
|------|----------|------|--------|----------|
| Full Attention | 131071 tok | 237.18s | **552.63 tok/s** | baseline |
| XAttention BSA | 131071 tok | 281.17s | **466.17 tok/s** | 0.84x |

### KV Cache 配置

| 上下文 | GPU Memory | CPU Memory | Total |
|--------|------------|------------|-------|
| 32K | 512 MB (4 blocks) | 4096 MB (32 blocks) | 4608 MB |
| 128K | 512 MB (4 blocks) | 16384 MB (128 blocks) | 16896 MB |

## 分析

### XAttention 性能特点

1. **32K 上下文**: XAttention 比 Full 慢 26%
2. **128K 上下文**: XAttention 比 Full 慢 16%

随着上下文增长，XAttention 的相对性能有所提升（74% → 84%），但仍未超过 Full Attention。

### 原因分析

1. **tau=0.95 阈值较高**: 需要覆盖 95% 累积注意力，实际跳过的 block 较少
2. **估计开销**: `xattn_estimate_chunked` 需要对每个 chunk 计算稀疏 mask
3. **BSA kernel overhead**: Block sparse kernel 有额外的 mask 处理和索引开销
4. **Offload 瓶颈**: CPU→GPU 传输是主要瓶颈，稀疏注意力节省的是计算而非传输

### 适用场景

XAttention BSA 更适合以下场景：
- 更长的上下文（256K+），稀疏收益更明显
- 计算密集型任务（非 offload 模式），传输不是瓶颈
- 较低的 tau 阈值（如 0.8），增加稀疏性

## 运行命令

```bash
# Full Attention (32K)
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 32768

# XAttention BSA (32K)
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 32768 --enable-xattn

# Full Attention (128K)
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 131072

# XAttention BSA (128K)
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 131072 --enable-xattn

# 调整 XAttention 参数
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --enable-xattn --xattn-threshold 0.8 --xattn-stride 16
```

## 更新记录

- 2026-01-27: 初始测试，Llama-3.1-8B-Instruct, A100 80GB
