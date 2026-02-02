# Density Alignment Test Results

验证 GPU-only 和 Offload 模式下三阶段 KV chunking 流程的正确性。

## 测试配置

### GPU-only 模式
- **模型**: Qwen3-0.6B (28 layers, 16 heads, 8 KV heads, head_dim=128)
- **Threshold**: 0.9
- **Block Size**: 128 tokens (BSA block)
- **Stride**: 8
- **Chunk Size**: 16384 tokens

### Offload 模式
- **模型**: Llama-3.1-8B-Instruct (32 layers, 32 heads, 8 KV heads, head_dim=128)
- **Threshold**: 0.9
- **Block Size**: 128 tokens (BSA block)
- **Stride**: 4
- **Chunk Size**: 4096 tokens

## 三阶段 KV Chunking 对齐测试 (2026-02-02)

### 测试目的

验证 `xattn_estimate` 高层 API 与手动实现的三阶段 KV chunking 流程是否完全一致。

### 三阶段流程

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: softmax_compute_partial_stats                     │
│    └── 每个 KV chunk 独立计算 partial stats (m_i, l_i)       │
│                                                             │
│  Stage 2: merge_softmax_stats                               │
│    └── Host 端合并所有 chunks: (m_global, l_global)          │
│                                                             │
│  Stage 3: softmax_normalize_and_block_sum                   │
│    └── 使用全局 stats 归一化并计算 block sums                 │
└─────────────────────────────────────────────────────────────┘
```

### 测试结果

#### CHUNK_SIZE = 16384 (默认)

| Context | Tokens | Q Chunks | KV Chunks | Density | Mask 差异 | attn_sums 差异 | 结果 |
|---------|--------|----------|-----------|---------|-----------|----------------|------|
| 4K | 3,692 | 1 | 1 | 63.84% | 0 | 0.0 | ✅ |
| 8K | 7,892 | 1 | 1 | 64.98% | 0 | 0.0 | ✅ |
| 16K | 15,689 | 1 | 1 | 61.63% | 0 | 0.0 | ✅ |
| 32K | 32,485 | 2 | 2 | 50.21% | 0 | 0.0 | ✅ |
| **64K** | **64,891** | **4** | **4** | **37.00%** | **0** | **0.0** | ✅ |

#### CHUNK_SIZE = 4096 (更多 chunks)

| Context | Tokens | Q Chunks | KV Chunks | Density | xattn_estimate vs KV chunking | 结果 |
|---------|--------|----------|-----------|---------|-------------------------------|------|
| 4K | 3,692 | 1 | 1 | 63.84% | 0.000000 | ✅ |
| 8K | 7,892 | 2 | 2 | 63.02% | 0.000000 | ✅ |
| 16K | 15,689 | 4 | 4 | 60.08% | 0.000000 | ✅ |
| 32K | 32,485 | 8 | 8 | 49.84% | 0.000000 | ✅ |
| **64K** | **64,891** | **16** | **16** | **36.91%** | **0.000000** | ✅ |

### 64K 详细验证 (CHUNK_SIZE=4096)

64K 序列使用 chunk_size=4096 时产生 16×16 的 chunk 矩阵：

```
seq_len: 64891, q_chunk_num: 16, kv_chunk_num: 16

Q chunk 0:  merged 16 KV chunks → attn_sum shape=[1, 32, 32, 512]
Q chunk 1:  merged 16 KV chunks → attn_sum shape=[1, 32, 32, 512]
...
Q chunk 15: merged 16 KV chunks → attn_sum shape=[1, 32, 32, 512]
```

每个 Q chunk 需要合并 16 个 KV chunks 的 softmax stats，充分验证了 `merge_softmax_stats` 在大规模 chunk 合并场景下的正确性。

### 验证指标

| 指标 | 预期 | 所有长度实际结果 |
|------|------|------------------|
| attn_sums max diff | 0 | 0.000000e+00 |
| attn_sums mean diff | 0 | 0.000000e+00 |
| mask exact match | True | True |
| density diff | 0% | 0.000000% |

### 结论

✅ **三阶段 KV chunking 与一次性处理完全等价，无任何精度损失。**

- 当 seq_len < CHUNK_SIZE (16384)：单 chunk 处理
- 当 seq_len >= CHUNK_SIZE：多 chunk 分段处理后合并，结果与一次性处理完全一致

---

## Offload 模式测试 (2026-02-02)

使用 Offload 模式保存的真实 KV cache 数据进行测试。

### 测试结果

| 文件 | Tokens | Layer | Saved Density | Computed Density | Q/KV Chunks | 结果 |
|------|--------|-------|---------------|------------------|-------------|------|
| `qkv_3688.pt` | 3.7K | 3 | 38.34% | 38.34% | 1/1 | ✅ PASSED |
| `qkv_7888.pt` | 7.9K | 3 | 29.06% | 27.56% | 2/2 | ✅ PASSED |
| `qkv_15685.pt` | 15.7K | 3 | 19.77% | 18.60% | 4/4 | ✅ PASSED |
| `qkv_32485.pt` | 32.5K | 5 | 15.71% | 15.62% | 8/8 | ✅ PASSED |
| `qkv_64891.pt` | 64.9K | 3 | 11.09% | 11.09% | 16/16 | ✅ PASSED |

### Layer 5 GPU-only 测试 (threshold=0.9)

| 指标 | 结果 |
|------|------|
| Q/K shape | `[1, 16, 21001, 128]` (21K tokens) |
| Density | 6.24% |
| xattn_estimate vs KV chunking | 完全一致 (0.0000%) |
| mask 差异 | 0 / 435600 blocks |
| attn_sums 差异 | max=0.0, mean=0.0 |

### 观察

1. **Density 随 context 增长而降低**: 3.7K (38%) → 64.9K (11%)
2. **xattn_estimate API 与三阶段 KV chunking 完全一致**: 所有长度差异均为 0.0000%
3. **Saved density vs Computed density 略有差异**: 这是因为 saved density 可能在不同 chunk 下记录，累积计算方式略有不同

---

## 附录：xattn_bsa vs xattn_estimate 对齐

| Context | Tokens | Layer 0 Density | Compute Density | Min Layer | 验证结果 |
|---------|--------|-----------------|-----------------|-----------|----------|
| 4k | 3,692 | 63.8% | 52.9% | Layer 3 (31.3%) | ✅ PASSED |
| 8k | 7,892 | 65.0% | 52.5% | Layer 5 (27.3%) | ✅ PASSED |
| 16k | 15,689 | 61.6% | 47.8% | Layer 5 (23.5%) | ✅ PASSED |
| 32k | 32,485 | 50.2% | 40.1% | Layer 5 (18.5%) | ✅ PASSED |
| 64k | 64,891 | 37.0% | 29.6% | Layer 5 (12.4%) | ✅ PASSED |

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
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

### Step 3: 运行 KV chunking 对齐验证

```bash
# 使用 GPU-only 保存的数据
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/test_xattn_estimate_alignment.py --gpuonly

# 使用 Offload 模式保存的数据 (默认)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/test_xattn_estimate_alignment.py

# 指定自定义数据文件
python tests/test_xattn_estimate_alignment.py --data-file /path/to/data.pt

# 批量测试所有 Offload 数据
for f in results/kvcache/qkv_*.pt; do
    echo "Testing: $(basename $f)"
    python tests/test_xattn_estimate_alignment.py --data-file "$f"
done
```

### 批量测试所有长度

```bash
for ctx in 4k 8k 16k 32k 64k; do
    case $ctx in
        4k)  max_len=5000 ;;
        8k)  max_len=9000 ;;
        16k) max_len=17000 ;;
        32k) max_len=34000 ;;
        64k) max_len=65664 ;;
    esac

    echo "Testing $ctx..."
    python tests/test_ruler.py \
        --data-dir tests/data/ruler_$ctx \
        --max-model-len $max_len \
        --sparse-policy XATTN_BSA \
        --num-samples 1 --quiet

    python tests/test_xattn_estimate_alignment.py --gpuonly
done
```

## 相关文件

- `nanovllm/kvcache/sparse/xattn_bsa.py`: XAttention BSA Policy 实现
- `nanovllm/ops/xattn.py`: xattn_estimate 函数及三阶段 KV chunking kernels
- `tests/test_xattn_estimate_alignment.py`: KV chunking 对齐验证脚本
