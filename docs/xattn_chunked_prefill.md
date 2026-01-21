# XAttention Chunked Prefill

## 概述

`xattn_estimate_chunked` 提供了 XAttention 的 chunked prefill 支持，允许将长序列分块处理，适用于显存受限或需要与 decode 请求交错执行的场景。

## 核心设计

### Chunked Prefill 模式

```
Full Prefill:     Q[0:N] × K[0:N] → Output[0:N]

Chunked Prefill:  Q[0:C] × K[0:C] → Output[0:C]
                  Q[C:2C] × K[0:2C] → Output[C:2C]
                  Q[2C:3C] × K[0:3C] → Output[2C:3C]
                  ...
```

关键特点：
- **Q 分块处理**：每次只处理一个 Q chunk
- **K/V 累积**：K/V cache 随着 chunk 处理逐步累积
- **位置感知**：通过 `q_start_pos` 参数传递当前 chunk 在原序列中的位置

## API

### xattn_estimate_chunked

```python
def xattn_estimate_chunked(
    query_states: torch.Tensor,  # (B, H, q_chunk_len, D) - 当前 Q chunk
    key_states: torch.Tensor,    # (B, H, k_len, D) - 累积的完整 K
    q_start_pos: int,            # 当前 chunk 在原序列中的起始位置
    block_size: int = 128,       # 稀疏 attention 的 block 大小
    stride: int = 8,             # 估计时的下采样步长
    threshold: float = 0.9,      # block 选择阈值
    chunk_size: int = 16384,     # Triton kernel 对齐大小
    use_triton: bool = True,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        attn_sums: (B, H, q_blocks, k_blocks) - 每个 block 的 attention 分数
        simple_mask: (B, H, q_blocks, k_blocks) - 选中的 block mask
    """
```

## 使用方式

### 外部分块（生产部署推荐）

由 LLM 框架控制 chunk 划分：

```python
# 在 attention forward 中
def forward(self, query, key, value, position_ids, kv_cache, ...):
    q_start_pos = position_ids[0].item()

    # 估计 sparse pattern
    attn_sum, mask = xattn_estimate_chunked(
        query, kv_cache.key,
        q_start_pos=q_start_pos,
        block_size=128,
        stride=4,
        threshold=0.9,
        chunk_size=4096,  # 必须与外部 chunk 大小匹配
    )

    # 使用 mask 进行 sparse attention
    ...
```

### 一致性要求

**重要**：要实现 chunked 与 standard 版本 100% 一致，必须：

1. 标准版和 chunked 版使用**相同的 `chunk_size`** 参数
2. 例如：`xattn_estimate(..., chunk_size=4096)` 和 `xattn_estimate_chunked(..., chunk_size=4096)`

## 与标准版的关系

| 函数 | 用途 |
|------|------|
| `xattn_estimate` | Full prefill 的 pattern 估计 |
| `xattn_estimate_chunked` | Chunked prefill 的 pattern 估计 |

**一致性保证**：当 `chunk_size` 参数匹配时，`xattn_estimate_chunked` 与 `xattn_estimate` 产生**完全相同**的 mask。

## 测试

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_xattn_estimate_chunked.py
```

## 验证结果

使用真实 QKV 数据（8K-64K 序列长度）测试：
- 所有 chunk_size (2048, 4096, 8192) 均达到 100% 匹配
