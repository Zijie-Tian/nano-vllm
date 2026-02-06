# XAttention Density Types: Compute vs Communication

XAttention BSA 统计两种不同粒度的 density，它们反映不同的优化效果。

## 两种 Density 的定义

### 1. Compute Density（计算密度）

**粒度**: BSA block (128 tokens)

**公式**:
```
compute_density = selected_bsa_blocks / total_causal_bsa_blocks
```

**含义**: 实际需要计算 attention 的 blocks 占 causal 区域的比例。

**影响**: 决定 attention 计算量的减少。

### 2. Communication Density（通信密度）

**粒度**: CPU block (4096 tokens = 32 BSA blocks)

**公式**:
```
comm_density = selected_cpu_blocks / total_cpu_blocks
```

**含义**: 需要从 CPU 传输到 GPU 的 blocks 占总 blocks 的比例。

**影响**: 决定 H2D 传输量的减少。

## 为什么 Comm Density 通常高于 Compute Density

### 聚合效应

由于 CPU block 粒度是 BSA block 的 32 倍，CPU block 选择使用 `any()` 聚合：

```python
# BSA mask: [B, H, Q_bsa, K_bsa]
# Reshape to CPU block level
mask_per_cpu = mask.view(B, H, Q_bsa, num_cpu_blocks, bsa_per_cpu)
# Any BSA block selected -> whole CPU block needed
cpu_needed = mask_per_cpu.any(dim=-1).any(dim=2).any(dim=1)
```

只要 CPU block 中**任意一个**:
- Head 选择了该 block，或
- Q position 选择了该 block，或
- BSA sub-block 被选中

则整个 CPU block 都需要传输。

### 示例

| 场景 | Compute Density | Comm Density | 说明 |
|------|-----------------|--------------|------|
| 64K context, threshold=0.9 | 37% | 100% | 稀疏 blocks 均匀分布在所有 CPU blocks |
| 32K context, threshold=0.9 | 50% | 100% | 同上 |

## 测试结果

**测试日期**: 2026-02-07
**GPU**: RTX 3090 (GPU 5)
**模型**: Llama-3.1-8B-Instruct (GQA, 32 heads, 8 KV heads)

### 测试命令

```bash
# 32K Offload 模式 (3 samples)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 3 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9

# 128K Offload 模式 (1 sample)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_128k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 135000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

### Comm Density 测试结果

| Context | Samples | Pass Rate | Compute Density | Comm Density | H2D Savings |
|---------|---------|-----------|-----------------|--------------|-------------|
| 32K | 3 | 3/3 (100%) | 38.01% | **100%** | 0% |
| 64K | 1 | 1/1 (100%) | 36.91% | **100%** | 0% |
| 128K | 1 | 1/1 (100%) | ~23.5% | **100%** | 0% |

### DensityObserver 输出示例 (32K)

```
[DensityObserver] Mode: offload
  Compute density: 0.3801 (min: 0.3801 @ layer 0)
  Comm density:    1.0000 (CPU block granularity)
  Savings ratio:   0.0% H2D transfer reduction
  Num layers: 1
  Layer 0 density: 0.380100
```

## 关键发现

### Comm Density = 100% 的数学证明

三级 `any()` 聚合过程:

```
mask_per_cpu: [1, 32_heads, Q_bsa, num_cpu_blocks, 32_bsa_per_cpu]
                               ↓ any(dim=-1) : 32 BSA 子块取并
                               ↓ any(dim=2)  : Q 块取并
                               ↓ any(dim=1)  : 32 heads 取并
cpu_needed:   [1, num_cpu_blocks]
```

以 32K context 为例 (compute density = 38%):
- 单个 head 选中某 BSA block 的概率 ≈ 0.38
- **Head 并集**: 32 heads 都不选中某 BSA block 的概率 = (1-0.38)^32 ≈ 6.7×10⁻⁷
- **BSA 子块并集**: CPU block 内 32 个 BSA 子块都不被选中的概率 ≈ (6.7×10⁻⁷)^32 → 0
- 结论: 任意 CPU block 被选中的概率 → **100%**

### 当前 XAttention 的通信优化局限

1. **Compute density 有效降低**: 23-38% (随 context 增长)，计算量减少 62-77%
2. **Comm density 没有降低**: 100%（H2D 传输量没有任何减少）

### 原因分析

Attention pattern 的特点：
- 不同 heads 关注不同位置（GQA 32 heads 的并集覆盖全部位置）
- 不同 Q positions 关注不同 K positions
- 稀疏选择分布在整个 sequence 上

这导致虽然每个 (head, Q, K) 组合只选择少量 blocks，但聚合后覆盖了所有 CPU blocks。

### 代码位置

Comm density 聚合逻辑: `xattn_bsa.py:922-924`

```python
mask_per_cpu = mask_historical.view(B, H, Q_bsa, num_historical_blocks, bsa_per_cpu)
cpu_needed = mask_per_cpu.any(dim=-1).any(dim=2).any(dim=1)  # [B, num_cpu]
```

Comm density 记录: `xattn_bsa.py:947` (仅 layer_id == 0)

### 潜在优化方向

1. **Per-head block selection**: 每个 head 独立选择 CPU blocks（需要 per-head H2D）
2. **Block clustering**: 将相关 blocks 聚合到同一 CPU block
3. **Dynamic block size**: 根据 attention pattern 动态调整 CPU block 大小
4. **减小 CPU block 粒度**: 从 4096 降到更小值（但会增加 pipeline 管理开销）

## DensityObserver API

### 启用和重置

```python
from nanovllm.utils.density_observer import DensityObserver

DensityObserver.enable()
DensityObserver.complete_reset()
DensityObserver.set_mode("offload")  # or "gpu_only"
```

### 记录

```python
# Compute density (GPU-only 模式自动记录)
DensityObserver.record(layer_id, mask, causal=True)

# Comm density (Offload 模式在 select_blocks 中记录)
DensityObserver.record_comm_density(layer_id, selected_cpu_blocks, total_cpu_blocks)
```

### 获取结果

```python
# 总体 density
overall_compute = DensityObserver.get_overall_density()
overall_comm = DensityObserver.get_overall_comm_density()

# Per-layer density
per_layer_compute = DensityObserver.get_per_layer_density()
per_layer_comm = DensityObserver.get_per_layer_comm_density()

# 打印摘要
DensityObserver.print_summary()
```

## 相关文件

- `nanovllm/utils/density_observer.py`: DensityObserver 实现
- `nanovllm/kvcache/sparse/xattn_bsa.py`: XAttention BSA Policy（select_blocks 中记录 comm density）
- `tests/test_ruler.py`: RULER benchmark 测试脚本
