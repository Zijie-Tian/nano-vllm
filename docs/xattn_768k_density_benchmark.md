# XAttention 768K Context Density Benchmark

**测试日期**: 2026-02-05
**GPU**: RTX 3090 (GPU 4)
**Context**: 768K tokens (实际 786,086 tokens)
**模式**: CPU Offload + XAttn BSA

---

## 测试配置

| 项目 | 值 |
|------|-----|
| 模型 | GLM-4-9B-Chat-1M |
| 数据集 | tests/data/ruler_768k/niah_single_1 |
| 样本数 | 1 |
| max_model_len | 800,000 |
| sparse_policy | XATTN_BSA |
| sparse_threshold | 0.9 |
| dtype | bfloat16 |
| enable_cpu_offload | true |
| chunk_size (Q) | 4,096 tokens |
| BSA block_size | 128 tokens |
| CPU block_size | 4,096 tokens |

### 测试命令

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_768k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 800000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9 \
    --dtype bfloat16
```

---

## 测试结果

### 整体指标

| 指标 | 值 |
|------|-----|
| **测试时长** | 5267.3 秒 (87.8 分钟) |
| **准确率** | 100% (1/1) |
| **总 chunk 数** | 192 chunks (786086 / 4096) |
| **每 chunk 耗时** | ~27.4 秒 |

### Density 统计

| 指标 | 值 | 说明 |
|------|-----|------|
| **Compute Density** | **56.95%** | BSA block (128 tokens) 粒度下的稀疏度 |
| **Comm Density** | **100%** | CPU block (4096 tokens) 粒度下的稀疏度 |
| **H2D Savings** | **0%** | 无 H2D 传输量节省 |
| **Layer 0 Density** | 56.95% | 单层 density (GLM-4 使用 MQA) |

---

## Per-Chunk Density 详细分析

### Density 变化趋势

768K context 的 prefill 过程分为 192 个 chunk，每个 chunk 处理 4096 个 query tokens。下表展示了不同阶段的 density 变化：

| 阶段 | Chunk 范围 | k_len 范围 | Density 范围 | 趋势 |
|------|-----------|-----------|-------------|------|
| **初期** | 1-25 | 4K-100K | 86% → 62% | 快速下降 |
| **中前期** | 26-75 | 100K-300K | 62% → 57% | 缓慢下降 |
| **中后期** | 76-125 | 300K-500K | 57% → 55.2% | 最低点 |
| **后期** | 126-192 | 500K-786K | 55.2% → 55.6% | 轻微回升并稳定 |

### 关键 Chunk 数据点

| Chunk | k_len | Density | 说明 |
|-------|-------|---------|------|
| 1 | 4,096 | 0.8594 | 初始 density 最高 |
| 25 | 102,400 | 0.6216 | 100K 时 density |
| 50 | 204,800 | 0.5844 | 200K 时 density |
| 75 | 307,200 | 0.5804 | 300K 时 density |
| 100 | 409,600 | 0.5752 | 400K 时 density |
| 125 | 512,000 | 0.5554 | 512K 时 density (最低附近) |
| 133 | 544,768 | **0.5529** | **最低点** |
| 150 | 614,400 | 0.5591 | 600K 时 density (回升) |
| 175 | 716,800 | 0.5555 | 700K 时 density |
| 192 | 786,086 | 0.5559 | 最终 density |

### Density 分布直方图

```
Density Range    | Count | Percentage
-----------------|-------|------------
0.85 - 0.90      |     1 |   0.5%    ████
0.80 - 0.85      |     2 |   1.0%    ████
0.75 - 0.80      |     3 |   1.6%    █████
0.70 - 0.75      |     5 |   2.6%    ██████
0.65 - 0.70      |     8 |   4.2%    ████████
0.60 - 0.65      |    18 |   9.4%    ████████████████
0.55 - 0.60      |   155 |  80.7%    ████████████████████████████████████████████████
```

**观察**: 80% 以上的 chunk 的 density 集中在 55%-60% 区间。

---

## Compute Density vs Comm Density 分析

### 两种 Density 的定义

| 类型 | 粒度 | 计算方式 | 用途 |
|------|------|---------|------|
| **Compute Density** | BSA block (128 tokens) | selected_blocks / total_blocks | 实际 attention 计算量 |
| **Comm Density** | CPU block (4096 tokens) | loaded_cpu_blocks / total_cpu_blocks | H2D 传输量 |

### 聚合效应

Comm density 为 100% 的原因是 **粒度不匹配**：

```
CPU Block (4096 tokens)
├── BSA Block 0 (128 tokens) - selected
├── BSA Block 1 (128 tokens) - NOT selected
├── BSA Block 2 (128 tokens) - selected
├── ...
└── BSA Block 31 (128 tokens) - selected

=> 只要 CPU Block 内有任何一个 BSA Block 被选中，整个 CPU Block 都需要加载
=> 当 compute density = 57% 时，几乎所有 CPU Block 都包含至少一个被选中的 BSA Block
=> 因此 comm density ≈ 100%
```

### 数学分析

假设 BSA blocks 均匀分布，每个 CPU block 包含 32 个 BSA blocks (4096/128)：

```
P(CPU block not selected) = (1 - compute_density)^32
                          = (1 - 0.5695)^32
                          = 0.4305^32
                          ≈ 1.6 × 10^-12
                          ≈ 0

=> 几乎所有 CPU block 都会被选中
=> comm_density ≈ 100%
```

### 临界点分析

要使 comm density 显著降低，需要更低的 compute density：

| Compute Density | P(CPU block not selected) | Expected Comm Density |
|-----------------|---------------------------|----------------------|
| 57% | 1.6 × 10^-12 | ~100% |
| 30% | 0.0000012 | ~100% |
| 10% | 0.035 | ~96.5% |
| 5% | 0.19 | ~81% |
| 1% | 0.72 | ~28% |

**结论**: 当前的 block size 配置下，需要 compute density < 5% 才能有效减少通信量。

---

## 与其他 Context Length 对比

| Context | Compute Density | Comm Density | XAttn 收益 |
|---------|-----------------|--------------|-----------|
| 32K | 49.8% | ~100% | 负收益 (额外开销 > 节省) |
| 64K | ~45% | ~100% | 微小收益或无 |
| 128K | ~42% | ~100% | 待测试 |
| **768K** | **56.95%** | **100%** | **无通信节省** |

**注意**: 768K 的 compute density (56.95%) 比 32K (49.8%) 更高，这可能与 NIAH 任务的注意力模式有关。

---

## 性能特征

### 每 Chunk 耗时分布

基于之前的 32K nsys 分析，XAttn 每 chunk 的主要开销：

| 阶段 | 耗时 | 占比 |
|------|------|------|
| xattn_find_blocks | ~14.6 ms | 54% |
| xattn_estimate_pass1 | ~3.7 ms | 14% |
| xattn_estimate_pass2 | ~3.4 ms | 13% |
| xattn_estimate_merge | ~1.2 ms | 4% |
| xattn_compute_historical | ~3.6 ms | 13% |
| xattn_compute_current | ~0.2 ms | 1% |
| xattn_compute_merge | ~0.2 ms | 1% |

### 768K 特有的性能特征

| 指标 | 32K | 768K | 差异 |
|------|-----|------|------|
| 总 chunk 数 | 8 | 192 | 24x |
| K 长度增长 | 4K→32K | 4K→786K | 24x |
| 每 chunk 耗时 | ~27 ms | ~27.4 s | 1000x |
| find_blocks 复杂度 | O(n) | O(n) | 线性增长 |

**瓶颈**: 768K context 下，单个 chunk 需要处理的 K 长度达到 786K tokens，导致每 chunk 耗时从 ms 级增长到 s 级。

---

## 结论与建议

### 主要结论

1. **Compute density 稳定在 55-57%**
   - 768K context 下，XAttn BSA 可以跳过约 43% 的 attention 计算
   - 但这个节省无法转化为通信量节省

2. **Comm density 为 100%，无通信节省**
   - CPU block 粒度 (4096 tokens) 与 BSA block 粒度 (128 tokens) 不匹配
   - 聚合效应导致几乎所有 CPU block 都被选中

3. **XAttn BSA 在当前配置下对 Offload 模式无正收益**
   - 额外的 estimate 和 find_blocks 开销无法被通信节省抵消
   - 但准确率保持 100%，功能正确性得到验证

### 优化建议

#### 短期优化

1. **减小 CPU block size**
   - 当前: 4096 tokens
   - 建议: 尝试 512 或 1024 tokens
   - 预期: 可能将 comm density 降低到 80-90%

2. **提高稀疏阈值**
   - 当前: sparse_threshold = 0.9
   - 建议: 尝试更高阈值 (0.95, 0.99)
   - 预期: 可能将 compute density 降低到 30-40%

#### 中期优化

3. **动态启用 XAttn BSA**
   - 仅在 compute density < 30% 时启用
   - 其他情况使用 Full Attention

4. **优化 find_blocks 实现**
   - 当前是 Python 层逻辑
   - 考虑 Triton/CUDA 实现减少 CPU-GPU 同步

#### 长期优化

5. **变长 CPU block**
   - 根据 BSA mask 动态确定需要加载的 token 范围
   - 避免加载不必要的 tokens

6. **探索其他稀疏策略**
   - MInference、Quest 等可能有更好的稀疏性

---

## 附录：测试日志摘要

### 启动日志

```
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_768k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 800000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9 \
    --dtype bfloat16
```

### 最终输出

```
============================================================
Density Statistics (XAttention BSA)
============================================================
[DensityObserver] Mode: offload
  Compute density: 0.5695 (min: 0.5695 @ layer 0)
  Comm density:    1.0000 (CPU block granularity)
  Savings ratio:   0.0% H2D transfer reduction
  Num layers: 1
  Layer 0 density: 0.569484
============================================================

test_ruler: PASSED
```

### Per-Chunk Density 日志格式

```
[HH:MM:SS] [INFO] [xattn_bsa.py:850] [XAttn Offload] Layer0 chunk:
    q_len=4096,           # Query 长度 (固定 4096，最后一个 chunk 可能更小)
    k_len=786086,         # Key 长度 (累积增长)
    valid_q_blocks=30,    # 有效 Q blocks
    valid_k_blocks=6142,  # 有效 K blocks
    q_offset=6112,        # Q 偏移量
    selected=3269845,     # 选中的 attention blocks
    total=5882400,        # 总 attention blocks
    density=0.5559        # 当前 chunk 的 density
```

---

## 相关文档

- [`docs/xattn_density_types.md`](xattn_density_types.md) - Compute vs Comm density 定义
- [`docs/xattn_density_benchmark.md`](xattn_density_benchmark.md) - 4K-32K density 基准测试
- [`docs/xattn_density_alignment_verification.md`](xattn_density_alignment_verification.md) - Density 对齐验证
- [`results/nsys/analysis_32k_offload_full_vs_xattn.md`](../results/nsys/analysis_32k_offload_full_vs_xattn.md) - 32K NSys 分析
