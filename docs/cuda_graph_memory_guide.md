# CUDA Graph 内存机制指南

本文档基于对 Qwen3-4B 模型的实际测试，详细分析 CUDA Graph 在 LLM 推理中的内存行为。

## 概述

CUDA Graph 通过捕获 GPU kernel 执行序列并重放来减少 CPU 开销，从而提升推理性能。本指南重点分析其内存特性。

## 性能提升

| 模式 | Decode 吞吐量 | 说明 |
|------|--------------|------|
| Eager | ~25 tok/s | 每次推理重新调度 kernel |
| CUDA Graph | ~70 tok/s | 重放预录制的 kernel 序列 |
| **加速比** | **2.80x** | |

## 内存阶段分析

基于 Qwen3-4B (bf16) 在 RTX 3090 上的测试结果：

### 各阶段内存变化

| 阶段 | 内存 (MB) | 增量 | 说明 |
|------|-----------|------|------|
| 模型加载 | 7672 | +7672 | 模型权重 |
| StaticCache 分配 | 7816 | +144 | **主要开销** |
| Warmup (3次) | 7825 | +8 | 激活值缓存 |
| Graph 捕获 | 7833 | +8 | 存储 kernel 序列 |
| Graph Replay | 7833 | **0** | 零额外分配 |

### 关键发现

1. **Graph 捕获开销很小**：仅约 8 MB，用于存储 kernel 调用序列

2. **StaticCache 是主要开销**：
   ```
   size = num_layers × 2 × batch_size × num_kv_heads × max_cache_len × head_dim × dtype_size
   ```
   - Qwen3-4B (1024 tokens): 36 × 2 × 1 × 8 × 1024 × 128 × 2 = **144 MB**

3. **Graph Replay 零分配**：所有张量地址在 capture 时已固定，replay 只重放 kernel

## Cache 长度与内存关系

| Cache 长度 | 总开销 | 每 1K tokens |
|------------|--------|--------------|
| 256 | 53 MB | 206 MB |
| 512 | 89 MB | 174 MB |
| 1024 | 161 MB | 157 MB |
| 2048 | 305 MB | 149 MB |
| 4096 | 593 MB | 145 MB |

内存开销与 cache 长度近似线性关系，每 1K tokens 约需 145-160 MB。

## CUDA Graph 工作原理

### 核心要求：固定内存地址

CUDA Graph 要求所有张量在 capture 时地址固定，之后只能通过 `copy_()` 更新值：

```python
# 分配固定地址的张量
static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
static_cache_position = torch.tensor([0], dtype=torch.long, device=device)

# Capture 时使用这些张量
with torch.cuda.graph(graph):
    outputs = model(input_ids=static_input_ids, ...)

# Replay 时通过 copy_() 更新值（地址不变）
static_input_ids.copy_(new_token)       # 更新输入
static_cache_position.fill_(position)   # 更新位置
graph.replay()                          # 重放
```

### StaticCache vs DynamicCache

| 特性 | DynamicCache | StaticCache |
|------|--------------|-------------|
| 内存分配 | 按需增长 | 预分配固定大小 |
| 地址稳定性 | 不稳定 | 稳定 |
| CUDA Graph 兼容 | ❌ | ✅ |
| 内存效率 | 高（按需） | 低（预分配） |

### 典型工作流程

```
1. Prefill (Eager)
   └── 使用 DynamicCache 处理变长输入

2. 创建 StaticCache
   └── 预分配 max_cache_len 大小的缓存

3. 复制 Prefill KV 到 StaticCache
   └── 将 DynamicCache 内容拷贝到固定地址

4. Warmup (3次)
   └── 确保所有 lazy initialization 完成

5. Capture Graph
   └── 录制 decode 的 kernel 序列

6. Decode Loop
   └── 更新输入 → graph.replay() → 读取输出
```

## 多 Batch Size Graph 的内存问题

如果为多个 batch size 分别捕获 graph（如 nanovllm 的设计），内存会快速增长：

| Batch Size | StaticCache (1024 tokens) | 累计 |
|------------|---------------------------|------|
| 1 | 144 MB | 144 MB |
| 2 | 288 MB | 432 MB |
| 4 | 576 MB | 1,008 MB |
| 8 | 1,152 MB | 2,160 MB |
| 16 | 2,304 MB | 4,464 MB |
| ... | ... | ... |

这是因为每个 batch size 需要独立的 StaticCache。实际系统（如 nanovllm）使用 PagedAttention 共享 KV cache 来避免此问题。

## 测试脚本

提供了测试脚本用于验证以上结论：

```bash
# 基本内存分析
CUDA_VISIBLE_DEVICES=0 python tests/test_cudagraph_memory.py

# 指定 cache 长度
CUDA_VISIBLE_DEVICES=0 python tests/test_cudagraph_memory.py --max-cache-len 2048

# 测试 cache 长度缩放
CUDA_VISIBLE_DEVICES=0 python tests/test_cudagraph_memory.py --test-scaling
```

性能对比演示：

```bash
# Eager vs CUDA Graph 性能对比
CUDA_VISIBLE_DEVICES=0 python tests/data/test_cudagraph_demo.py --mode both
```

## 总结

| 项目 | 结论 |
|------|------|
| 性能提升 | ~2.8x decode 吞吐量 |
| Graph 捕获开销 | ~8 MB（很小） |
| 主要内存开销 | StaticCache（与 cache_len 成正比） |
| Replay 内存 | 零额外分配 |
| 核心要求 | 固定张量地址 |
