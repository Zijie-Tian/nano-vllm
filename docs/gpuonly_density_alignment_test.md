# GPU-Only Density Alignment Test Results

验证 GPU-only 模式下 `xattn_bsa.py` 的 density 计算与独立调用 `xattn_estimate` 的一致性。

## 测试配置

- **模型**: Llama-3.1-8B-Instruct (32 layers, 32 heads, 8 KV heads, head_dim=128)
- **Threshold**: 0.9 (选择覆盖 90% attention 的 blocks)
- **Block Size**: 128 tokens (BSA block)
- **Stride**: 8
- **数据集**: RULER niah_single_1 (各长度 1 sample)

## 测试结果

| Context | Tokens | Layer 0 Density | Compute Density | Min Layer | 验证结果 |
|---------|--------|-----------------|-----------------|-----------|----------|
| 4k | 3,692 | 63.8% | 52.9% | Layer 3 (31.3%) | ✅ PASSED |
| 8k | 7,892 | 65.0% | 52.5% | Layer 5 (27.3%) | ✅ PASSED |
| 16k | 15,689 | 61.6% | 47.8% | Layer 5 (23.5%) | ✅ PASSED |
| 32k | 32,485 | 50.2% | 40.1% | Layer 5 (18.5%) | ✅ PASSED |
| 64k | 64,891 | 37.0% | 29.6% | Layer 5 (12.4%) | ✅ PASSED |

## 验证指标

对于所有测试长度，验证脚本检查以下指标：

| 指标 | 预期 | 实际结果 |
|------|------|----------|
| attn_sums max diff | 0 | 0.000000e+00 |
| attn_sums mean diff | 0 | 0.000000e+00 |
| mask exact match | True | True |
| density diff | 0 | 0.000000 |

## Density 计算公式

### Total (分母)

```python
# Causal mask: Q block i 只能看到 K block 0 到 i
causal_mask[i, j] = (j <= i + q_offset_blocks)

# Total = causal 区域内的 block 数 × batch × heads
total = causal_mask.sum() × batch × heads
      = (n × (n+1) / 2) × 1 × 32  # n = valid_q_blocks
```

### Selected (分子)

```python
# 在 causal 区域内，被选中 (mask=True) 的 block 数量
selected = (mask & causal_mask).sum()
```

### Density

```python
density = selected / total
```

## 观察

1. **Density 随 context 增长而降低**: 4k (63.8%) → 64k (37.0%)，这是因为长序列中 attention 更加分散

2. **Layer 5 通常是最稀疏的层**: 在所有长度测试中，Layer 5 的 density 最低

3. **Layer 0 density 最高**: 第一层的 attention pattern 最密集，可能与 sink token 效应有关

4. **Threshold=0.9 对应 ~50% density**: 在 32k context 下，threshold=0.9 意味着选择覆盖 90% attention 的 blocks，实际 density 约 50%

## 使用方法

### Step 1: 启用 debug 保存

```python
# nanovllm/kvcache/sparse/xattn_bsa.py
_DEBUG_SAVE_MASK = True  # 改为 True
```

### Step 2: 运行 GPU-only 推理

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

### Step 3: 运行验证脚本

```bash
python tests/test_gpuonly_density_alignment.py
```

## 相关文件

- `nanovllm/kvcache/sparse/xattn_bsa.py`: XAttention BSA Policy 实现
- `nanovllm/ops/xattn.py`: xattn_estimate 函数
- `tests/test_gpuonly_density_alignment.py`: 验证脚本
