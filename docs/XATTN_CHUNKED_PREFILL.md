# XAttention Chunked Prefill

## 概述

`Xattn_chunked.py` 提供了 XAttention 的 chunked prefill 实现，支持将长序列分块处理，适用于显存受限或需要与 decode 请求交错执行的场景。

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

## API 说明

### 函数层级

```
Xattention_chunked_full_prefill()   ← 测试用：内部自动分 chunk
    │
    └── Xattention_chunked_prefill()   ← 生产用：单个 Q chunk 的 attention
            │
            └── xattn_estimate_chunked()   ← 核心：单个 Q chunk 的稀疏 pattern 估计
```

### xattn_estimate_chunked

稀疏 pattern 估计函数，返回 attention 分数和 block mask。

```python
def xattn_estimate_chunked(
    query_states: torch.Tensor,  # (B, H, q_chunk_len, D) - 当前 Q chunk
    key_states: torch.Tensor,    # (B, H, k_len, D) - 累积的完整 K
    q_start_pos: int,            # 当前 chunk 在原序列中的起始位置
    block_size: int,             # 稀疏 attention 的 block 大小 (通常 128)
    stride: int,                 # 估计时的下采样步长 (通常 4)
    threshold: float = 0.9,      # block 选择阈值
    chunk_size: int = 16384,     # Triton kernel 对齐大小
    ...
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        attn_sums: (B, H, q_blocks, k_blocks) - 每个 block 的 attention 分数
        simple_mask: (B, H, q_blocks, k_blocks) - 选中的 block mask
    """
```

### Xattention_chunked_prefill

单个 Q chunk 的完整 attention 计算。

```python
def Xattention_chunked_prefill(
    query_states: torch.Tensor,   # (B, H, q_chunk_len, D)
    key_states: torch.Tensor,     # (B, H, k_len, D) - 累积的 K
    value_states: torch.Tensor,   # (B, H, k_len, D) - 累积的 V
    q_start_pos: int,             # 当前 chunk 的起始位置
    stride: int,
    threshold: float = 0.8,
    block_size: int = 128,
    layer_id: int = None,         # 用于 density 统计
    num_layers: int = 32,
    ...
) -> torch.Tensor:
    """
    Returns:
        attn_output: (B, H, q_chunk_len, D) - 当前 chunk 的 attention 输出
    """
```

### Xattention_chunked_full_prefill

测试/验证用的包装函数，内部自动分 chunk。

```python
def Xattention_chunked_full_prefill(
    query_states: torch.Tensor,   # (B, H, seq_len, D) - 完整 Q
    key_states: torch.Tensor,     # (B, H, seq_len, D) - 完整 K
    value_states: torch.Tensor,   # (B, H, seq_len, D) - 完整 V
    stride: int,
    threshold: float = 0.8,
    chunk_size: int = 16384,      # 分块大小
    ...
) -> torch.Tensor:
    """
    Returns:
        attn_output: (B, H, seq_len, D) - 完整的 attention 输出
    """
```

## 使用方式

### 方式一：外部分块（生产部署推荐）

由 LLM 框架（vLLM/nanovllm）控制 chunk 划分：

```python
# 在 LLM attention 层的 forward 中
def forward(self, query, key, value, position_ids, kv_cache, ...):
    q_start_pos = position_ids[0].item()

    output = Xattention_chunked_prefill(
        query,
        kv_cache.key,    # 累积的完整 K
        kv_cache.value,  # 累积的完整 V
        q_start_pos=q_start_pos,
        stride=4,
        threshold=0.9,
        block_size=128,
        layer_id=self.layer_idx,
        num_layers=self.config.num_hidden_layers,
    )
    return output
```

### 方式二：内部分块（测试验证用）

直接调用包装函数，内部自动处理分块：

```python
output = Xattention_chunked_full_prefill(
    query, key, value,
    stride=4,
    threshold=0.9,
    chunk_size=4096,
    layer_id=layer_idx,
    num_layers=32,
)
```

### 等价性

两种方式**完全等价**，只要外部分块时：
- 每个 chunk 的 `q_start_pos` 正确传入
- K/V 是累积到当前位置的完整 KV cache

## 与标准版的关系

| 函数 | 位置 | 用途 |
|------|------|------|
| `xattn_estimate` | `Xattention.py` | Full prefill 的 pattern 估计 |
| `xattn_estimate_chunked` | `Xattn_chunked.py` | Chunked prefill 的 pattern 估计 |
| `Xattention` | `Xattention.py` | Full prefill attention |
| `Xattention_chunked_prefill` | `Xattn_chunked.py` | Chunked prefill attention |

**一致性保证**：`xattn_estimate_chunked` 与 `xattn_estimate` 产生**完全相同**的 mask（已通过测试验证）。

## 技术细节

### Padding 策略

为满足 Triton kernel 对齐要求：
- Q 和 K 都 pad 到 `chunk_size` 的整数倍（默认 16384）
- 计算后通过 `valid_q_reshaped` 和 `valid_k_reshaped` 提取有效区域

### Causal Mask 处理

- Block 级别：Q block i 只能看到 K blocks [0, q_start_block + i]
- Token 级别：由 `softmax_fuse_block_sum` kernel 的 `real_q_len` 参数控制

### Density 统计

通过 `_density_tracker` 全局变量跨层追踪密度：
- 在 `layer_id=0` 时清空
- 在 `layer_id=num_layers-1` 时输出最小密度层

## 测试

运行单元测试：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
    python tests/test_xattn_chunked.py
```

运行 RULER benchmark：

```bash
cd eval/RULER/scripts
CUDA_VISIBLE_DEVICES=0 bash run.sh llama3.1-8b-chat synthetic \
    --metric xattn_chunked --task niah_single_1
```

## 验证结果

- **单元测试**：所有序列长度（4K-64K）与标准版 100% 匹配
- **RULER benchmark**：niah_single_1 任务 100% 得分（65536 序列长度）
