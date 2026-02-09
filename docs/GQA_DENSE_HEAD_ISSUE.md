# GQA Dense Head Issue: KV Cache Loading Bottleneck in Chunked Prefill + CPU Offload

## 问题概述

在 chunked prefill + CPU offload 策略下，GQA（Grouped Query Attention）中同一 KV group 内存在的 dense head 会迫使整个 group 的 KV cache 几乎全部加载，导致稀疏化收益大幅下降。

## 背景

### Chunked Prefill + CPU Offload 策略

该策略的核心思路是：
1. KV cache 存储在 CPU 内存中
2. 根据 Q 的 attention 分布估计（理论 estimate），选择性地将需要的 K 块加载到 GPU
3. 仅对加载的 K 块计算 attention，跳过不重要的 K 块
4. 以此减少 GPU 显存占用和 CPU→GPU 数据传输量

### GQA 结构约束

在 GQA 中，多个 Q head 共享同一个 KV head。以 GLM-4-9B 为例：
- 32 个 Q heads，4 个 KV heads
- 每 8 个 Q head 共享 1 个 KV head（num_groups = 8）
- **关键约束**：同一 group 内的所有 Q head 使用相同的 K/V 数据

## 问题描述

### 观察现象

从 GLM-4-9B Layer 5（16k, seq=15689）的 post-RoPE attention map 中可以清晰观察到：

**同一 KV group 内，不同 Q head 的 attention 稀疏度差异极大。**

| KV Group | Sparse Heads（对角线为主） | Dense Heads（大面积 attend） |
|----------|---------------------------|------------------------------|
| KV 0 | H0, H1, H2 | H3, H4, H5, H6, H7 |
| KV 1 | H8, H9, H10, H11 | H12, H13, H15 |
| KV 2 | H16, H18, H21 | H17, H19, H20, H22 |
| KV 3 | H24, H25, H27 | H26, H28, H29, H31 |

Dense head 的特征：attention 分布覆盖大量 K token，而非集中在 local 或少量关键位置。

### 根本问题

由于 GQA 的结构约束，**同一 group 中所有 Q head 共享相同的 KV cache**。当决定加载哪些 K 块时：

```
需要加载的 K 块 = Union(所有 Q head 需要的 K 块)
```

即使一个 group 中有 7 个 sparse head 只需要加载 20% 的 K 块，只要有 1 个 dense head 需要加载 90% 的 K 块，那么整个 group 就必须加载 90% 的 K 块。

**Dense head 成为整个 group 的 bottleneck，决定了 KV cache 的最小加载量。**

### 量化影响

假设一个 group 中各 head 的 K 块需求比例为：

```
Sparse heads: [0.15, 0.18, 0.12, 0.20, 0.16, 0.14, 0.19]
Dense head:   [0.88]

Union 后实际加载比例 ≈ 0.88（由 dense head 决定）
```

理论上 7/8 的 head 只需要 ~17% 的 K 块，但实际必须加载 ~88%，稀疏化效率从理想的 ~80% 压缩下降到 ~12%。

## 可视化证据

参考图片：`results/kvcache-rope-ret/glm-4-9b/16k/layer_05_attnmap.png`

- 图片布局：4 行（KV group）× 8 列（group 内的 Q head）
- 色标：log10(attention)，深紫 = 极低 attention，黄色 = 高 attention
- 每个 group 内都可以看到明显的稀疏-密集分化

## 影响范围

此问题可能在以下条件下更为严重：
- **更长的序列**：长序列中 dense head 覆盖范围更广
- **更深的层**：不同层的 dense head 分布可能不同，需要逐层分析
- **不同模型**：GQA ratio 越大（如 Llama-3.1 的 8:1），问题越突出

## 待验证

1. Dense head 在不同层的分布是否一致？（是否总是同样的 head 位置）
2. Dense head 的密度是否随序列长度增长？
3. 不同模型（Llama-3.1-8B, Qwen2.5-7B）是否存在相同问题？
4. 是否可以通过分离 dense head 的 KV 加载策略来缓解？
