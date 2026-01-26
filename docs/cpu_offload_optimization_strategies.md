# CPU Offload 优化策略

本文档记录 CPU Offload 场景下的性能优化策略分析，包括实际可行的方案和前沿研究方向。

## 问题回顾

根据 [CPU 调度延迟分析](cpu_scheduling_latency_analysis.md)，当前 chunked attention pipeline 的主要问题：

| 指标 | 当前值 | 理论值 |
|------|--------|--------|
| Flash kernel 执行时间 | ~138 μs | - |
| Flash kernel 间隔 | ~942 μs | ~211 μs (仅 H2D + merge) |
| GPU 利用率 | **12.8%** | **39.5%** (理论上限) |
| CPU 调度空闲占比 | **77-81%** | 0% |

**瓶颈根源**：每个 block 都经过完整的 Python 循环，导致大量 CPU 调度延迟。

---

## 优化方案一：调大 Chunk Size（推荐）

### 核心洞察

**Merge 多个小 chunk 和直接使用大 chunk 是等效的**：

```
方案 A: Merge 4 个小 chunks
[H2D 2K][H2D 2K][H2D 2K][H2D 2K] → concat → [Flash 8K] → merge

方案 B: 直接用大 chunk
[H2D 8K] → [Flash 8K] → merge

计算结果完全等效！
```

### 收益分析

| 指标 | 小 chunk (2K) × 4 | 大 chunk (8K) × 1 |
|------|-------------------|-------------------|
| H2D 次数 | 4 | 1 |
| Flash kernel 调用 | 4 | 1 |
| Merge 调用 | 4 | 1 |
| Python 循环次数 | 4 | 1 |
| CPU 调度开销 | 4 × ~300μs = 1200μs | 1 × ~300μs = 300μs |

**本质**：CPU 调度延迟问题的根源是循环次数太多，调大 chunk size 直接减少循环次数。

### Trade-off

1. **GPU 内存增加**
   - 2K chunk: 每 slot ~4MB (K+V)
   - 8K chunk: 每 slot ~16MB (K+V)
   - 4 slots = 64MB，对 80GB A100 影响很小

2. **单次 H2D 时间变长**
   - H2D 8K ≈ 350μs
   - Flash 8K ≈ 550μs
   - 因为 Flash > H2D，pipeline 仍然有效

### 配置方法

```bash
# 测试不同 block size
python bench_offload.py --kvcache-block-size 2048   # 基准
python bench_offload.py --kvcache-block-size 4096   # 2x
python bench_offload.py --kvcache-block-size 8192   # 4x
```

---

## 优化方案二：CUDA Graph（适用于非 Attention 部分）

### CUDA Graph 在 Offload 场景的局限性

CUDA Graph 的前提：所有操作在 capture 时确定，数据地址固定。

**Offload 场景的现实**：
1. **H2D 源地址动态** - 每次从不同的 CPU block 加载
2. **加载决策在运行时** - 哪些 block 需要加载是动态的
3. **CPU 必须协调** - H2D 和 Compute 的同步需要 CPU 参与

```
Offload 场景：
┌─────────────────────────────────────────┐
│  数据在 CPU，需要动态加载                 │
│  [H2D_i] → [Compute] → [H2D_{i+n}] → ...│
│  ↑ 动态、CPU 必须参与调度                 │
└─────────────────────────────────────────┘

即使用 Graph：
Python: [wait_h2d] [replay] [launch_h2d] [wait_h2d] [replay] ...
        ↑ CPU 参与           ↑ CPU 参与   ↑ CPU 参与

CPU 调度开销仍然存在，Graph 只优化了中间的 compute 部分。
```

**结论**：CUDA Graph 不是 Offload 场景的银弹。

### 适用场景：MLP 和 Projection 层

LLM 每层的计算流程：

```
┌─────────────────────────────────────────────────────────────┐
│  [LayerNorm] → [QKV Proj] → [Attention] → [O Proj] → [Add]  │
│                                  ↑                          │
│                             KV Offload                      │
│  [LayerNorm] → [MLP: gate + up + down] → [Add]              │
└─────────────────────────────────────────────────────────────┘
```

| 组件 | 涉及 Offload | 能用 CUDA Graph |
|------|-------------|-----------------|
| LayerNorm | ❌ | ✅ |
| QKV Projection | ❌ | ✅ |
| **Attention** | ✅ | ❌ |
| Output Projection | ❌ | ✅ |
| MLP (FFN) | ❌ | ✅ |

**只有 Attention 涉及动态 KV Cache 加载，其余都是"纯计算"，可以用 CUDA Graph。**

### 实现方案

```python
class OptimizedLayer:
    def __init__(self, layer):
        # Graph 1: Attention 之前
        self.graph_pre_attn = capture([
            layer.input_layernorm,
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ])

        # Graph 2: Attention 之后 + MLP
        self.graph_post_attn = capture([
            layer.self_attn.o_proj,
            # residual add
            layer.post_attention_layernorm,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
            # residual add
        ])

    def forward(self, hidden_states, kv_cache):
        # Pre-attention (CUDA Graph)
        self.graph_pre_attn.replay()

        # Attention with offload (动态，不能用 graph)
        attn_output = chunked_attention_with_offload(q, kv_cache)

        # Post-attention + MLP (CUDA Graph)
        self.graph_post_attn.replay()
```

### 收益估算

MLP 每层典型操作 launch 开销：
- `gate_proj`, `up_proj`, `act_fn`, `gate * up`, `down_proj`, `residual add`
- 每个操作 ~30-50μs launch 开销，总计 ~200μs/层
- 用 CUDA Graph：~30μs/层

**32 层 × 170μs 节省 ≈ 5.4ms**

---

## 优化方案三：前沿研究方向

### 1. InfiniGen - 投机预取 (OSDI'24)

**核心思想**：不需要加载所有 KV，只预取"重要"的 token。

```
关键洞察：相邻层的 attention pattern 高度相似
         ↓
用第 L 层的 attention score 预测第 L+1 层需要哪些 token
         ↓
只预取 top-k 重要的 KV entries（而不是全部）
```

**技术实现**：
- 用当前层的 Q 和下一层的部分 K 做"预演"
- 预测下一层的 attention 分布
- 异步预取预测的重要 token
- **减少 PCIe 带宽浪费，而不是加速传输**

**效果**：最高 **3x 加速**

**参考**：[InfiniGen (OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/lee)

### 2. ShadowKV - 低秩压缩 + Sparse Offload (ICML'25 Spotlight)

**核心思想**：Key 压缩存 GPU，Value offload 到 CPU，只加载 1.56% 的 KV。

```
Pre-filling:
┌─────────────────────────────────────────────────┐
│  Key Cache → SVD 低秩压缩 → 保留在 GPU          │
│  Value Cache → Offload 到 CPU                   │
│  计算每个 chunk 的 landmark (均值)               │
│  识别 outlier tokens → 保留在 GPU               │
└─────────────────────────────────────────────────┘

Decoding:
┌─────────────────────────────────────────────────┐
│  用 landmarks 快速估计 attention score          │
│  只加载 top-k 重要的 Value (1.56% sparse)       │
│  结合 GPU 上的 outliers 计算最终结果            │
└─────────────────────────────────────────────────┘
```

**效果**：6x 更大 batch size，**3.04x 吞吐提升**

**参考**：[ShadowKV (ByteDance)](https://github.com/ByteDance-Seed/ShadowKV)

### 3. L2 Cache 异步预取 (2025)

**核心思想**：利用 GPU L2 Cache 做预取，在计算时预取下一批 KV。

```
传统：
Compute:  [Flash_i]        [Flash_{i+1}]
H2D:              [H2D_{i+1}]
                  ↑ 等待

L2 Prefetch：
Compute:  [Flash_i  + Prefetch_{i+1} to L2]  [Flash_{i+1} L2 hit]
          ↑ 计算时利用空闲 memory bandwidth 预取
```

**技术**：
- 在 Flash Attention kernel 内部发起预取指令
- 利用计算时的空闲 memory bandwidth
- 下一次访问直接 L2 hit

**效果**：**2.15x attention kernel 效率**，1.97x 端到端吞吐

**参考**：[Asynchronous KV Cache Prefetching (2025)](https://arxiv.org/abs/2504.06319)

### 4. KVPR - I/O-Aware 调度 (ACL'25)

**核心思想**：计算最优的 recompute vs offload 比例。

```
权衡：
- Recompute: 重新计算 KV（用 GPU 算力换内存）
- Offload: 从 CPU 加载（用 PCIe 带宽换算力）

KVPR: 根据当前负载动态决定最优比例
      + 预取技术重叠数据传输和计算
```

**参考**：[KVPR (ACL'25)](https://aclanthology.org/2025.findings-acl.997.pdf)

---

## 优化策略总结

### 推荐优先级

| 优先级 | 方案 | 核心优化 | 实现复杂度 | 预期收益 |
|--------|------|---------|-----------|---------|
| **P0** | 调大 chunk size | 减少循环次数 | 极低（改配置） | 2-4x |
| **P1** | MLP CUDA Graph | 减少 launch 开销 | 中 | ~5ms/request |
| **P2** | InfiniGen 式预取 | 只加载重要 token | 中高 | 2-3x |
| **P3** | ShadowKV 式压缩 | Key 压缩 + Sparse | 高 | 3x |
| **P3** | C++ Extension | 消除 Python 开销 | 高 | 2-3x |

### 策略分离原则

```
┌─────────────────────────────────────────────────────────────┐
│  Attention + Offload 部分：                                 │
│    - 瓶颈：H2D 传输 + CPU 调度                              │
│    - 优化：调大 chunk size / 投机预取 / Sparse              │
│                                                             │
│  MLP + Proj + Norm 部分：                                   │
│    - 瓶颈：Kernel launch 开销                               │
│    - 优化：CUDA Graph                                       │
└─────────────────────────────────────────────────────────────┘

两部分优化完全正交，可以组合使用。
```

---

## 相关文件

- `nanovllm/kvcache/sparse/full_policy.py`: Chunked attention pipeline
- `nanovllm/kvcache/offload_engine.py`: H2D/D2H 传输管理
- `docs/cpu_scheduling_latency_analysis.md`: 问题分析

## 参考文献

1. [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://www.usenix.org/conference/osdi24/presentation/lee) - OSDI'24
2. [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://github.com/ByteDance-Seed/ShadowKV) - ICML'25 Spotlight
3. [Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching](https://arxiv.org/abs/2504.06319) - 2025
4. [KVPR: Efficient LLM Inference with I/O-Aware KV Cache](https://aclanthology.org/2025.findings-acl.997.pdf) - ACL'25
5. [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://lmcache.ai/tech_report.pdf) - 2025
