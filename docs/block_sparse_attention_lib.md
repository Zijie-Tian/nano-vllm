# Block-Sparse-Attention Library Reference

MIT Han Lab 的块稀疏注意力内核库，基于 FlashAttention 2.4.2 修改，支持多种稀疏注意力模式。

## 库信息

- **来源**: [MIT-Han-Lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- **本地路径**: `3rdparty/Block-Sparse-Attention` (submodule, branch: `tzj/minference`)
- **基于**: FlashAttention 2.4.2
- **安装位置**: `site-packages/block_sparse_attn`

## 支持的稀疏模式

### 1. Dense Attention
计算完整注意力矩阵，无稀疏化。

### 2. Token Streaming (token granularity)
固定数量的 sink tokens + local tokens，参考 [StreamingLLM](https://arxiv.org/abs/2309.17453)。

**适用场景**: 需要精确保留部分关键 token 的长上下文推理

### 3. Block Streaming (block granularity)
Block 粒度的 streaming attention，block_size = 128。

**适用场景**: 长序列推理，牺牲少量精度换取更大加速

### 4. Block Sparse
基于自定义 block mask 的稀疏注意力。

**适用场景**: 已知特定 attention 模式的工作负载

### 混合模式

**关键特性**: 支持不同 head 使用不同稀疏模式

```python
# 8 个 heads 的混合配置示例
head_mask_type = [1, 1, 0, 0, 0, -1, 0, -1]
# 含义:
# - head 0,1: blocksparse (使用 basemask[0])
# - head 2-4,6: dense
# - head 5,7: streaming
```

**Mask 类型编码**:
- `0` = Dense attention
- `-1` = Streaming attention
- `1, 2, ...` = Block sparse (使用 basemask[mask_type - 1])

## API 参考

### `block_sparse_attn_func`

通用块稀疏注意力函数，支持所有模式。

```python
from block_sparse_attn import block_sparse_attn_func

output = block_sparse_attn_func(
    q, k, v,                    # [total_tokens, heads, head_dim] unpadded
    cu_seqlens_q, cu_seqlens_k, # cumulative sequence lengths
    head_mask_type,             # [heads] tensor, 每个头的模式
    streaming_info,             # streaming 配置 (sink/local 数量)
    base_blockmask,             # [q_blocks, k_blocks, n_masks] bool tensor
    max_seqlen_q, max_seqlen_k, # 最大序列长度
    p_dropout,                  # dropout 概率 (推理时设为 0.0)
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,      # True=token streaming, False=block streaming
    return_attn_probs=False,
)
```

**关键参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `head_mask_type` | Tensor[heads] | 每个头的稀疏模式，0=dense, -1=streaming, 1+=blocksparse |
| `streaming_info` | Tensor | [sink_blocks, local_blocks] 或 [sink_tokens, local_tokens] |
| `base_blockmask` | Tensor | Block mask，形状 [q_blocks, k_blocks, n_masks] |
| `exact_streaming` | bool | True=token 粒度，False=block 粒度 streaming |

### `block_streaming_attn_func`

Block 粒度 streaming attention（block_size=128）。

```python
from block_sparse_attn import block_streaming_attn_func

output = block_streaming_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,             # [sink_blocks, local_blocks]
    max_seqlen_q, max_seqlen_k,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=True,
    return_attn_probs=False,
)
```

### `token_streaming_attn_func`

Token 粒度 streaming attention。

**注意**: 不支持反向传播（仅推理）。

```python
from block_sparse_attn import token_streaming_attn_func

output = token_streaming_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,             # [sink_tokens, local_tokens]
    max_seqlen_q, max_seqlen_k,
    deterministic=False,
    softmax_scale=None,
    return_attn_probs=False,
)
```

## 技术规格

| 特性 | 支持情况 |
|------|----------|
| **数据类型** | fp16, bf16 (bf16 需要 Ampere/Ada/Hopper GPU) |
| **Head 维度** | 32, 64, 128 |
| **Block Size** | 128 (固定) |
| **CUDA 要求** | 11.6+ |
| **PyTorch 要求** | 1.12+ |

## 性能参考

测试环境: A100 GPU, head_dim=128, 32 heads, batch_size=1

### Block Sparse 加速比
- 相比 FlashAttention2: 最高 **3-4x** 加速
- 加速随序列长度增加而提升

### Streaming 混合模式加速比
- Token streaming: 64 sink + 256 local tokens
- Block streaming: 1 sink block + 3 local blocks
- **50% Dense + 50% Streaming**: 最高 **2x** 加速

## 与 nano-vllm 的集成考虑

### 潜在集成点

1. **长上下文推理优化**
   - 使用 block streaming 减少计算量
   - 在 CPU offload 模式下减少 GPU-CPU 传输

2. **混合注意力策略**
   - 部分 head 使用 streaming（减少计算）
   - 部分 head 使用 dense（保持精度）
   - 参考 Duo Attention 论文的混合模式

3. **稀疏 offload**
   - 只 offload 重要 blocks 的 KV cache
   - 结合 `requires_block_selection` 接口

### 实现注意事项

1. **输入格式**: 库使用 unpadded 格式（`cu_seqlens`），需要与 nano-vllm 的 padded 格式转换
2. **Block size 固定**: 库固定 block_size=128，需要适配
3. **Streaming info 配置**: 需要根据模型特性调整 sink/local 数量

## 相关工作

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - 基础实现
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - Streaming attention 理论基础
- [Duo Attention](https://github.com/mit-han-lab/duo-attention) - 混合 dense/streaming 模式
- [MInference](https://arxiv.org/abs/2407.02490) - 混合 mask 方法

## 测试

库自带测试位于 `3rdparty/Block-Sparse-Attention/block_sparse_tests/`:

```bash
# 正确性测试
cd 3rdparty/Block-Sparse-Attention/block_sparse_tests/fwd/test_correctness
pytest full_test.py

# 性能测试
cd 3rdparty/Block-Sparse-Attention/block_sparse_tests/fwd/test_performance
python token_streaming.py
python blocksparse.py
```
