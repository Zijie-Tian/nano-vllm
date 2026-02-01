# XAttention KV Chunking Density 验证测试

## 背景

验证 XAttention Triton kernel 是否只能沿 Q 轴分 chunk，不能沿 KV 轴分 chunk。

**假设**：`softmax_fuse_block_sum` 需要完整的 K 来计算正确的归一化分母，分 chunk 后的 attention 分布与完整序列不同。

## 测试方法

1. **GPU-only 模式**：一次性对完整序列调用 `xattn_estimate`，记录 Layer 0 的 density
2. **Offload DEBUG 模式**：分 chunk 调用 `xattn_estimate`，累积 selected/total counts，计算最终 density
3. 使用相同的 `_debug_k_full` buffer 收集完整 K cache，确保输入数据一致

### 关键代码逻辑

```python
# Offload DEBUG: 每个 chunk 累积 selected/total
for each chunk:
    K_full = _debug_k_full[:, :, :total_k_len, :]  # 累积的 K
    _, mask_chunk = xattn_estimate(Q_chunk, K_full, threshold=threshold, causal=True)

    # 裁剪到有效区域，计算正确的 causal mask (考虑 Q 偏移量)
    q_offset_blocks = k_blocks - q_blocks
    causal_mask = indices <= (q_indices + q_offset_blocks)

    selected += (mask_valid & causal_mask).sum()
    total += causal_mask.sum()

density = selected / total
```

## 测试结果

### 64K 序列 (niah_single_1, 序列长度 64891)

| threshold | GPU-only selected | Offload selected | GPU-only density | Offload density | 差异 (selected) |
|-----------|------------------|------------------|------------------|-----------------|-----------------|
| **0.90** | 1,524,617 | 1,330,506 | **0.3700** | **0.3229** | 194,111 (12.7%) |
| **0.95** | 1,955,015 | 1,747,585 | **0.4744** | **0.4241** | 207,430 (10.6%) |
| **1.00** | 4,118,719 | 4,118,896 | **0.9995** | **0.9995** | -177 (~0%) |

- **total**: 4,120,896 (两种模式一致)

### 32K 序列 (niah_single_1, 序列长度 32485)

| threshold | GPU-only selected | Offload selected | GPU-only density | Offload density | 差异 (selected) |
|-----------|------------------|------------------|------------------|-----------------|-----------------|
| **0.90** | 520,314 | 466,937 | **0.5021** | **0.4506** | 53,377 (10.3%) |
| **0.95** | 647,765 | 602,953 | **0.6251** | **0.5818** | 44,812 (6.9%) |
| **1.00** | 1,036,295 | 1,036,264 | **0.9999** | **0.9999** | 31 (~0%) |

- **total**: 1,036,320 (两种模式一致)

### 汇总对比

| 序列长度 | threshold | GPU-only density | Offload density | density 差异 |
|---------|-----------|------------------|-----------------|--------------|
| 32K | 0.90 | 0.5021 | 0.4506 | 5.2% |
| 64K | 0.90 | 0.3700 | 0.3229 | 4.7% |
| 32K | 0.95 | 0.6251 | 0.5818 | 4.3% |
| 64K | 0.95 | 0.4744 | 0.4241 | 5.0% |
| 32K | 1.00 | 0.9999 | 0.9999 | ~0% |
| 64K | 1.00 | 0.9995 | 0.9995 | ~0% |

## 结论

### 1. Softmax 归一化本身是正确的

当 `threshold=1.0`（选择所有 blocks）时，GPU-only 和 Offload 模式的 density 几乎完全对齐（差异 < 0.01%）。

这说明：
- `_debug_k_full` 正确收集了完整的 K cache
- 分 chunk 调用 `xattn_estimate` 时，softmax 归一化在正确的 K 序列上计算
- causal mask 的 Q 偏移量处理正确

### 2. 问题在于 threshold 的应用方式

当 `threshold < 1.0` 时，差异显著（10-13%）：

- **GPU-only**：对完整序列一次性应用 threshold，选择 cumulative attention >= threshold 的 blocks
- **Offload**：每个 chunk 独立应用 threshold，累积 selected counts

每个 chunk 独立应用 threshold 会导致：
- 某些在 GPU-only 中被选中的 blocks，在分 chunk 时因 attention 分布不同而未被选中
- 累积的 selected 比一次性计算的要少

### 3. XAttention Triton kernel 的 KV chunking 限制

**验证结论**：XAttention 的 `xattn_estimate` 可以正确处理 KV chunking（softmax 归一化正确），但 **threshold-based block selection 不能简单累积**。

如果要在 Offload 模式下获得与 GPU-only 一致的 block selection：
1. 需要先累积所有 chunks 的 attention scores
2. 最后一次性应用 threshold 选择 blocks

或者接受 10-13% 的 density 差异，这对实际推理准确性的影响需要进一步评估。

## 测试命令

```bash
# GPU-only 模式
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py --dataset niah_single_1 --sample 0 \
    --sparse-policy xattn_bsa --sparse-threshold 0.9

# Offload 模式 (64K)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py --dataset niah_single_1 --sample 0 \
    --sparse-policy xattn_bsa --sparse-threshold 0.9 --enable-offload

# Offload 模式 (32K)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py --dataset niah_single_1 --sample 0 \
    --sparse-policy xattn_bsa --sparse-threshold 0.9 --enable-offload \
    --data-dir /home/zijie/Code/nano-vllm/tests/data/ruler_32k --max-model-len 34000
```

## 相关文件

- `nanovllm/kvcache/sparse/xattn_bsa.py`: DEBUG 代码位置
- `nanovllm/ops/xattn.py`: `xattn_estimate` 实现
- `nanovllm/utils/density_observer.py`: DensityObserver 实现
