# XAttention Density Benchmark

GPU-only 模式下 XAttention Block Sparse Attention 的 density 测试结果。

## 测试配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Model | Llama-3.1-8B-Instruct | 32 layers, 32 heads, 8 KV heads |
| Block Size | 128 tokens | BSA kernel 固定要求 |
| Threshold | 0.9 / 0.95 | 累积注意力阈值 |
| Stride | 4 / 8 / 16 | Q/K 下采样步长 |
| Dataset | RULER niah_single_1 | Sample 0 |
| Mode | GPU-only | 无 CPU offload |

## Density 定义

```python
# Density = selected_blocks / total_causal_blocks
# 在 causal attention 下，只计算下三角区域的 blocks
# Overall density = 所有层的平均值

def compute_density(mask, causal=True):
    """
    mask: [batch, heads, q_blocks, k_blocks] boolean tensor
    """
    if causal:
        causal_mask = torch.tril(torch.ones(q_blocks, k_blocks))
        total = causal_mask.sum() * batch * heads
        selected = (mask & causal_mask).sum()
    return selected / total
```

## 测试结果

### threshold=0.9

#### Overall Density (平均)

| Context | stride=4 | stride=8 | stride=16 |
|---------|----------|----------|-----------|
| **4K** | 0.5220 (52.2%) | 0.5292 (52.9%) | 0.5430 (54.3%) |
| **8K** | 0.5152 (51.5%) | 0.5252 (52.5%) | 0.5396 (54.0%) |
| **16K** | 0.4682 (46.8%) | 0.4775 (47.8%) | 0.4888 (48.9%) |
| **32K** | 0.3700 (37.0%) | 0.4012 (40.1%) | 0.4196 (42.0%) |

#### Min Density (per layer)

| Context | stride=4 | stride=8 | stride=16 |
|---------|----------|----------|-----------|
| **4K** | 0.2805 (Layer 3) | 0.3132 (Layer 3) | 0.3376 (Layer 5) |
| **8K** | 0.2886 (Layer 5) | 0.2725 (Layer 5) | 0.2995 (Layer 5) |
| **16K** | 0.2247 (Layer 5) | 0.2349 (Layer 5) | 0.2451 (Layer 5) |
| **32K** | 0.1799 (Layer 5) | 0.1846 (Layer 5) | 0.1964 (Layer 5) |

### threshold=0.95

#### Overall Density (平均)

| Context | stride=4 | stride=8 | stride=16 |
|---------|----------|----------|-----------|
| **4K** | 0.6561 (65.6%) | 0.6699 (67.0%) | 0.6815 (68.2%) |
| **8K** | 0.6462 (64.6%) | 0.6584 (65.8%) | 0.6732 (67.3%) |
| **16K** | 0.6004 (60.0%) | 0.6114 (61.1%) | 0.6193 (61.9%) |
| **32K** | 0.4894 (48.9%) | 0.5203 (52.0%) | 0.5385 (53.9%) |

#### Min Density (per layer)

| Context | stride=4 | stride=8 | stride=16 |
|---------|----------|----------|-----------|
| **4K** | 0.3972 (Layer 3) | 0.4348 (Layer 5) | 0.4517 (Layer 4) |
| **8K** | 0.4004 (Layer 5) | 0.3906 (Layer 5) | 0.4239 (Layer 5) |
| **16K** | 0.3331 (Layer 5) | 0.3453 (Layer 5) | 0.3589 (Layer 5) |
| **32K** | 0.2656 (Layer 5) | 0.2784 (Layer 5) | 0.2917 (Layer 5) |

### threshold 对比 (stride=8)

| Context | threshold=0.9 | threshold=0.95 | 差异 |
|---------|---------------|----------------|------|
| **4K** | 0.5292 (52.9%) | 0.6699 (67.0%) | -14.1% |
| **8K** | 0.5252 (52.5%) | 0.6584 (65.8%) | -13.3% |
| **16K** | 0.4775 (47.8%) | 0.6114 (61.1%) | -13.4% |
| **32K** | 0.4012 (40.1%) | 0.5203 (52.0%) | -11.9% |

## 关键发现

### 1. Context Length 影响最大

Density 随 context length 显著下降（threshold=0.9, stride=8）：
- 4K: 52.9% density
- 8K: 52.5% density
- 16K: 47.8% density
- 32K: 40.1% density

**结论**: 长序列有更多稀疏化机会，XAttention 的优势在长序列上更明显。

### 2. Threshold 影响显著

threshold=0.9 比 0.95 的 density 低约 12-14%：
- 0.9 更激进，选择更少的 blocks
- 0.95 更保守，保留更多 blocks
- 两者准确性都不受影响（RULER NIAH 全部 PASS）

### 3. Stride 影响较小

同一 context 下，不同 stride 的 density 差异约 2-5%：
- stride 越大 → density 略高（采样越粗糙，选择更保守）
- stride=4 最激进，stride=16 最保守

### 4. Min Density 集中在中间层

- 大多数情况下 min density 出现在 Layer 5
- 中间层的稀疏性最高，首尾层相对密集
- 这符合 Transformer 注意力模式的一般规律

### 5. 最佳稀疏化配置

32K + stride=4 + threshold=0.9 达到最低 density：
- Overall: **37.0%** (节省 63% 计算)
- Min: **18.0%** (Layer 5)

### 6. 准确性稳定

所有配置下 RULER NIAH 测试都 PASS (score=1.0)，说明：
- threshold=0.9 和 0.95 都足够保守，不损失准确性
- 不同 stride 不影响最终结果

## 推荐配置

| 场景 | threshold | stride | 说明 |
|------|-----------|--------|------|
| 精度优先 | 0.95 | 8 | 保守配置，density ~52-67% |
| 平衡 | 0.9 | 8 | 默认配置，density ~40-53% |
| 性能优先 | 0.9 | 4 | 激进配置，density ~37-52% |

## 测试命令

```bash
# 基本测试
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --sample-indices 0 \
    --max-model-len 33792 \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9 \
    --sparse-stride 8 \
    --gpu-utilization 0.85

# 参数说明
# --sparse-policy XATTN_BSA    启用 XAttention Block Sparse Attention
# --sparse-threshold 0.9       累积注意力阈值 (0.9-0.99)
# --sparse-stride 8            Q/K 下采样步长 (4/8/16)
```

## DensityObserver 使用

```python
from nanovllm.utils.density_observer import DensityObserver

# 启用并重置
DensityObserver.enable()
DensityObserver.complete_reset()

# ... 运行 inference (compute_prefill 自动记录) ...

# 获取结果
summary = DensityObserver.get_summary()
# {
#     "mode": "gpu_only",
#     "overall_density": 0.40,  # 所有层的平均值
#     "per_layer_density": {0: 0.55, 1: 0.45, ...},
#     "num_layers": 32
# }

# 获取最低 density
min_layer, min_density = DensityObserver.get_min_density()

# 打印摘要
DensityObserver.print_summary()
# [DensityObserver] Mode: gpu_only
#   Overall density: 0.4012
#   Min density: 0.1846 (layer 5)
#   Num layers: 32
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `nanovllm/kvcache/sparse/xattn_bsa.py` | XAttention BSA Policy 实现 |
| `nanovllm/utils/density_observer.py` | Density 统计 Observer |
| `nanovllm/ops/xattn.py` | xattn_estimate 核心算法 |
| `tests/test_ruler.py` | RULER benchmark 测试脚本 |
