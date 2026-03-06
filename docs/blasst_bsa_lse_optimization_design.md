# BLASST BSA+LSE 优化设计方案

## 概述

本文档描述使用 Block Sparse Attention (BSA) + Log-Sum-Exp (LSE) 来优化 BLASST 性能的设计思路。

**核心思想**：将 BLASST 的 subblock 计算从逐个遍历改为使用 BSA kernel 一次性处理所有非零 blocks。

---

## 当前 BLASST 实现的问题

### 1. Kernel 调用次数过多

```python
# 当前实现（blasst.py）
for block_idx in range(num_blocks):  # ~32 blocks per layer
    # 1. 计算 skip mask
    skip_mask = blasst_mask_forward(q, k_block, running_max, ...)

    # 2. 计算 attention（内部再次遍历 subblocks）
    block_o, running_max, block_lse = blasst_attn_forward(
        q, k_block, v_block, skip_mask, running_max, ...
    )
```

**统计数据**（32K context, 4096 block size）：
- Kernel 调用次数：896 次（28 layers × 32 blocks）
- `_blasst_mask_kernel`：896 次，平均 4.7ms，总耗时 4.2s
- `_blasst_attn_masked_kernel`：896 次，平均 84.6ms，总耗时 **75.8s**

### 2. 重复 QK 计算

在 `blasst_attn_forward` 内部：

```python
# 第1遍：扫描所有 subblocks 计算 local_max
while n_start < kv_end:
    scores = tl.dot(q, tl.trans(k)) * softmax_scale
    local_max = tl.maximum(local_max, tl.max(scores, axis=1))
    n_start += BLOCK_N

# 第2遍：如果 any_compute，再次扫描计算 QK+PV
if any_compute:
    while n_start < kv_end:
        scores = tl.dot(q, tl.trans(k)) * softmax_scale  # 重复计算！
        p = tl.exp(scores - safe_new_m[:, None])
        acc += tl.dot(p, v)
```

### 3. Tile-level 跳过决策稀释 Row-level 稀疏性

- **Row skip rate**：96%（基于 BLASST 阈值）
- **Tile skip rate**：仅 7.3%（因为 `BLOCK_M=64`，`0.96^64 ≈ 0.073`）
- **结果**：92.7% 的 tiles 仍进入完整计算路径

---

## BSA + LSE 优化方案

### 核心设计

```python
def compute_chunked_prefill_bsa_optimized(self, ...):
    """
    使用 BSA + LSE 优化后的 BLASST chunked prefill
    """
    # Step 1: 计算 BLASST skip mask（仍然需要，保持动态阈值特性）
    skip_mask = blasst_mask_forward(q, k_block, running_max, ln_lambda, ...)
    # skip_mask shape: [num_heads, q_len, num_kv_blocks] (bool, True=skip)

    # Step 2: 转换为 BSA blockmask
    blockmask = convert_skip_to_bsa_blockmask(
        skip_mask,
        granularity=self.granularity  # 128
    )
    # blockmask shape: [num_heads, q_tiles, kv_tiles] (bool, True=compute)

    # Step 3: 使用 BSA kernel 一次性处理所有 subblocks
    block_o, block_lse = block_sparse_attn_with_lse(
        q=q,                          # [num_heads, q_len, head_dim]
        k=k_block,                    # [num_kv_heads, 4096, head_dim]
        v=v_block,                    # [num_kv_heads, 4096, head_dim]
        blockmask=blockmask,          # [num_heads, 32, 32]
        m_block_dim=self.granularity, # 128
        n_block_dim=self.granularity, # 128
        softmax_scale=softmax_scale,
    )

    # Step 4: Online merge with running state
    historical_o, historical_lse = merge_attention_outputs(
        historical_o, historical_lse,
        block_o, block_lse
    )
    running_max = extract_running_max_from_lse(block_lse)
```

### Blockmask 转换逻辑

```python
def convert_skip_to_bsa_blockmask(skip_mask: torch.Tensor, granularity: int) -> torch.Tensor:
    """
    将 BLASST row-level skip mask 转换为 BSA block-level mask

    Args:
        skip_mask: [num_heads, q_len, num_kv_blocks], bool, True=skip
        granularity: KV block size (default 128)

    Returns:
        blockmask: [num_heads, q_blocks, kv_blocks], bool, True=compute
    """
    num_heads, q_len, num_kv_blocks = skip_mask.shape
    q_blocks = triton.cdiv(q_len, granularity)

    # 方法1: 保守策略（推荐）
    # 如果 tile 内有任何一行需要计算，整个 tile 计算
    skip_mask_reshaped = skip_mask.view(
        num_heads, q_blocks, granularity, num_kv_blocks
    )
    # blockmask=True 表示需要计算（不skip）
    blockmask = ~skip_mask_reshaped.any(dim=2)  # [heads, q_blocks, kv_blocks]

    return blockmask
```

### 预期性能提升

| 指标 | 当前 BLASST | BSA+LSE 优化 | 提升 |
|------|------------|-------------|------|
| Kernel 调用次数 | 896 次 | ~28 次 (per layer) | **32×** |
| QK 扫描次数 | 2 × 32 = 64 次 | 1 次 | **64×** |
| 单次 block 耗时 | 84.6ms (attn) + 4.7ms (mask) | ~5-10ms (估计) | **8-16×** |
| 总计算时间 | ~80s | ~5-10s (估计) | **8-16×** |

---

## 技术可行性分析

### 1. Block-SparseAttention Kernel 接口

参考 `3rdparty/Block-SparseAttention/block_sparse_attn_interface.py`：

```python
def block_sparse_attn_forward(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,      # block size (granularity)
    head_mask_type,
    streaming_info,
    row_blockmask,                  # [heads, q_blocks, kv_blocks]
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    is_causal,
    exact_streaming,
    return_softmax,
    window_size_left,
    window_size_right
):
    out, ..., softmax_lse = block_sparse_attn_cuda.fwd_block(...)
    return out, softmax_lse
```

**关键特性**：
- ✅ 支持 blockmask 输入（sparse pattern）
- ✅ 支持 GQA (`num_heads` can be multiple of `num_kv_heads`)
- ✅ 返回 `softmax_lse`（支持 LSE merge）
- ✅ 基于 FlashAttention 的高效 CUDA kernel

### 2. 需要确认的技术点

| 问题 | 状态 | 备注 |
|------|------|------|
| BSA kernel 是否支持 `causal=False`？ | 需要验证 | BLASST historical blocks 是非因果的 |
| LSE 输出格式是否与现有 merge 兼容？ | 需要验证 | 需要检查 scale 和格式 |
| Blockmask 转换的稀疏性损失？ | 需要评估 | Row-level → Block-level 可能损失 10-20% 稀疏性 |
| 是否支持 bfloat16？ | 需要验证 | 当前 BLASST 支持 fp16/bf16 |

### 3. 与现有 BSA 的区别

| 特性 | XAttn BSA Policy | BLASST BSA+LSE |
|------|-----------------|----------------|
| Block size | 4096 (CPU block) | 128 (granularity) |
| Mask 来源 | XAttention estimate | BLASST dynamic threshold |
| Skip level | Block-level | Row-level (converted to block) |
| 动态性 | Static per chunk | Dynamic with running_max |
| 调用方式 | Multiple flash_attn | Single BSA kernel |

---

## 实现路径

### Phase 1: 验证 BSA Kernel 可用性

```python
# 测试代码框架
def test_bsa_kernel_compatibility():
    """验证 BSA kernel 是否支持我们的 use case"""
    q = torch.randn(32, 4096, 128, device="cuda", dtype=torch.float16)
    k = v = torch.randn(8, 4096, 128, device="cuda", dtype=torch.float16)

    # 创建测试 blockmask（假设 50% 稀疏）
    blockmask = torch.rand(32, 32, 32, device="cuda") > 0.5

    # 尝试调用 BSA kernel
    out, lse = block_sparse_attn_cuda.fwd_block(
        q, k, v, ..., blockmask, ..., causal=False
    )

    # 验证输出正确性
    assert out.shape == q.shape
    assert lse.shape == (32, 4096)  # [heads, q_len]
```

### Phase 2: 实现 Blockmask 转换

```python
# 在 nanovllm/ops/blasst_bsa.py 中实现
class BLASSTBlockMaskConverter:
    """将 BLASST skip mask 转换为 BSA blockmask"""

    @staticmethod
    def convert(skip_mask: torch.Tensor, granularity: int) -> torch.Tensor:
        """
        Args:
            skip_mask: [num_heads, q_len, num_kv_blocks], bool, True=skip
        Returns:
            blockmask: [num_heads, q_tiles, kv_tiles], bool, True=compute
        """
        # 实现转换逻辑
        pass

    @staticmethod
    def estimate_sparsity_loss(skip_mask: torch.Tensor, granularity: int) -> float:
        """估计 row-level → block-level 转换的稀疏性损失"""
        pass
```

### Phase 3: 集成到 BLASST Policy

```python
# 在 nanovllm/kvcache/sparse/blasst.py 中添加
class BLASSTPolicyBSA(BLASSTPolicy):
    """使用 BSA+LSE 优化的 BLASST Policy"""

    def __init__(self, use_bsa_kernel=True, **kwargs):
        super().__init__(**kwargs)
        self.use_bsa_kernel = use_bsa_kernel

    def compute_chunked_prefill(self, ...):
        if self.use_bsa_kernel and self._bsa_kernel_available():
            return self._compute_with_bsa(...)
        else:
            return super().compute_chunked_prefill(...)
```

---

## 风险与权衡

### 1. 稀疏性损失（主要风险）

```
Row-level skip rate: 96%
Block-level skip rate (granularity=128): ~80% (估计)

损失原因：
- 一个 128-token block 内只要有 1 个 query 需要计算，整个 block 都要计算
- 预计损失 15-20% 的稀疏性
```

**缓解措施**：
- 保持较小的 granularity（128 比 4096 损失小）
- 使用更激进的 BLASST 阈值（如 `fixed_lambda=0.3`）

### 2. 内存开销

BSA kernel 可能需要：
- Blockmask 存储：`[heads, q_tiles, kv_tiles]` = 32 × 32 × 32 = 32KB（可忽略）
- 额外的 LSE buffer：与当前相同

### 3. 正确性风险

- BSA kernel 的正确性需要验证
- LSE merge 的数值稳定性需要测试

---

## 性能对比预测

| 场景 | 当前 BLASST | BSA+LSE | Full Attention |
|------|------------|---------|----------------|
| 32K context, 96% skip | ~90s | ~10s (估计) | ~16s |
| 64K context, 94% skip | ~180s | ~20s (估计) | ~32s |

**目标**：在保持 96%+ 准确率的前提下，将 BLASST 性能提升到接近 Full Attention 水平。

---

## 下一步行动

1. **验证 BSA kernel 接口**：确认 `block_sparse_attn_cuda.fwd_block` 支持我们的参数组合
2. **编写 PoC 代码**：实现简单的 blockmask 转换和 BSA 调用
3. **性能基准测试**：对比当前 BLASST 和 BSA 版本的性能
4. **正确性验证**：确保 BSA 版本保持与原始 BLASST 相同的准确率

---

## 相关文档

- [BLASST Implementation Report](blasst_implementation_report.md)
- [BLASST 128 Granularity](blasst_128_granularity.md)
- [BLASST Configuration Guide](blasst_configuration_guide.md)
- [Block Sparse Attention Interface](../3rdparty/Block-SparseAttention/block_sparse_attn_interface.py)
- [XAttn BSA Policy Design](xattn_bsa_policy_design.md)

---

**创建时间**: 2026-03-06
**状态**: 设计阶段
**作者**: Claude Code
