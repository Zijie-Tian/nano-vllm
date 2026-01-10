# Sparse Prefill Attention Integration Plan

## Executive Summary

本文档整合了 int-minference-1/2/3 三个分支的分析，提出统一的三种稀疏注意力策略（MInference、XAttention、FlexPrefill）集成方案。

---

## Part 1: 现状分析

### 1.1 x-attention 仓库策略对比

| 策略 | Pattern 类型 | 估计方法 | Kernel Backend |
|------|-------------|---------|----------------|
| **MInference** | Vertical + Slash | Last-64-Q attention → 列/对角线求和 | `vertical_slash_sparse_attention` (minference lib) |
| **XAttention** | Block Mask | Stride-based Q/K 下采样 → block 分数 | `block_sparse_attn_func` (MIT-HAN-LAB) |
| **FlexPrefill** | Adaptive V+S | Last-block attention + JS 散度自适应 | `triton_block_wise_attention` (custom triton) |

### 1.2 关键发现：两种 Kernel 接口

**接口 A: Index-Based (minference)**
```python
# MInference 使用 vertical+slash indices
vertical_indices = [heads, vertical_size]  # 重要 K 列位置
slash_indices = [heads, slash_size]        # 对角线偏移
output = vertical_slash_sparse_attention(q, k, v, vertical_indices, slash_indices)
```

**接口 B: Block Mask-Based (block_sparse_attn)**
```python
# XAttention/FlexPrefill 使用 boolean block mask
block_mask = torch.bool[batch, heads, q_blocks, k_blocks]  # True = 计算
output = block_sparse_attn_func(q, k, v, block_mask, ...)
```

### 1.3 当前 nanovllm MInference 实现

**文件**: `nanovllm/kvcache/sparse/minference.py`

**已实现功能**:
- `estimate_pattern()`: 使用 last-64-Q 估计 vertical+slash pattern
- `sparse_prefill_attention()`: 调用 minference kernel 执行稀疏注意力
- 支持 GQA（通过 K/V repeat_interleave）
- 支持 adaptive_budget 自适应预算

**问题**:
1. 与 XAttention/FlexPrefill 使用不同 kernel，无法统一接口
2. `sparse_prefill_attention()` 将估计和执行耦合在一起
3. 没有 BlockMask 中间表示，难以复用

---

## Part 2: 架构设计

### 2.1 设计原则

1. **向后兼容**: 保持现有 `SparsePolicy` 接口不变
2. **渐进式重构**: 添加新功能而非替换
3. **统一中间表示**: 新策略使用 `BlockMask` 作为可选中间表示
4. **可插拔 Kernel**: 支持多种 attention kernel backend

### 2.2 架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Unified Sparse Prefill Framework                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │   MInference    │  │   XAttention    │  │   FlexPrefill   │  Strategies   │
│  │   Policy        │  │   Policy        │  │   Policy        │               │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘               │
│           │                    │                    │                         │
│           │ (indices)          │ (BlockMask)        │ (BlockMask)             │
│           │                    │                    │                         │
│           ▼                    └────────┬───────────┘                         │
│  ┌─────────────────┐                    ▼                                     │
│  │   minference    │  ┌─────────────────────────────────────────────────────┐│
│  │   kernel        │  │              BlockMask Container                    ││
│  └────────┬────────┘  │  [batch, num_heads, q_blocks, k_blocks] - boolean   ││
│           │           └─────────────────────────────────────────────────────┘│
│           │                              │                                    │
│           │                              ▼                                    │
│           │           ┌─────────────────────────────────────────────────────┐│
│           │           │         block_sparse_attn_func                      ││
│           │           │         (MIT-HAN-LAB kernel)                        ││
│           │           └─────────────────────────────────────────────────────┘│
│           │                              │                                    │
│           └──────────────────────────────┼────────────────────────────────── │
│                                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Attention Output                                  │ │
│  │                   [seq_len, num_heads, head_dim]                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 新增类设计

```python
# nanovllm/kvcache/sparse/block_mask.py

@dataclass
class BlockMask:
    """Block-level attention mask container."""
    mask: torch.Tensor      # [batch, heads, q_blocks, k_blocks]
    block_size: int
    seq_len: int
    num_q_blocks: int
    num_k_blocks: int

    def sparsity_ratio(self) -> float:
        """Fraction of blocks masked out."""
        return 1.0 - self.mask.float().mean().item()

    def to_flat_indices(self, head_idx: int) -> torch.Tensor:
        """Convert to flattened block indices for a given head."""
        pass

    @classmethod
    def from_vertical_slash(
        cls,
        vertical_idx: torch.Tensor,
        slash_idx: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> "BlockMask":
        """Convert MInference-style indices to block mask."""
        pass

    def apply_causal(self) -> "BlockMask":
        """Apply causal constraint (lower triangular)."""
        pass
```

```python
# nanovllm/kvcache/sparse/kernels/block_sparse.py

def block_sparse_attention(
    q: torch.Tensor,           # [seq_len, num_heads, head_dim]
    k: torch.Tensor,           # [seq_len, num_kv_heads, head_dim]
    v: torch.Tensor,           # [seq_len, num_kv_heads, head_dim]
    block_mask: BlockMask,
) -> torch.Tensor:
    """
    Execute block sparse attention using MIT-HAN-LAB kernel.

    Handles:
    - GQA expansion (K/V heads < Q heads)
    - Tensor format conversion
    - Causal masking
    """
    from block_sparse_attn import block_sparse_attn_func
    # ... implementation
```

---

## Part 3: 实现计划

### Phase 1: 基础设施 (新增文件)

**目标**: 添加 BlockMask 和 block_sparse_attn 封装

**文件**:
- `nanovllm/kvcache/sparse/block_mask.py` (NEW)
- `nanovllm/kvcache/sparse/kernels/__init__.py` (NEW)
- `nanovllm/kvcache/sparse/kernels/block_sparse.py` (NEW)

**任务**:
1. 实现 `BlockMask` 数据类
2. 实现 `block_sparse_attention()` 封装函数
3. 处理 GQA 和 tensor 格式转换
4. 测试：使用全 True 的 block mask 验证输出正确

### Phase 2: XAttention 实现

**目标**: 移植 x-attention 的 XAttention 策略

**文件**:
- `nanovllm/kvcache/sparse/xattention.py` (NEW)
- `nanovllm/config.py` (添加 XATTENTION 枚举)
- `nanovllm/kvcache/sparse/__init__.py` (更新工厂函数)

**关键函数移植**:
```python
# From x-attention/xattn/src/Xattention.py
def xattn_estimate(q, k, block_size, stride, threshold, ...):
    # 1. Stride-based Q/K downsampling
    reshaped_k = cat([k[:, :, i::stride, :] for i in range(stride)], dim=-1)
    reshaped_q = cat([q[:, :, stride-1-i::stride, :] for i in range(stride)], dim=-1)

    # 2. Block-level attention scores
    attn_weights = matmul(reshaped_q, reshaped_k.T) / sqrt(d) / stride

    # 3. Threshold selection
    block_mask = find_blocks_chunked(attn_sum, threshold)
    return block_mask
```

**配置参数**:
```python
xattention_stride: int = 16           # Q/K 下采样步长
xattention_threshold: float = 0.9     # 累积分数阈值
xattention_block_size: int = 128      # Block 大小
```

**测试**: `python tests/test_needle.py --input-len 32768 --enable-xattention`

### Phase 3: FlexPrefill 实现

**目标**: 移植 x-attention 的 FlexPrefill 策略

**文件**:
- `nanovllm/kvcache/sparse/flexprefill.py` (NEW)
- `nanovllm/config.py` (添加 FLEXPREFILL 枚举)

**关键函数移植**:
```python
# From x-attention/xattn/src/Flexprefill.py
def get_active_blocks(q, k, gamma, tau, block_size, ...):
    # 1. Last-block attention analysis
    last_q = q[:, -block_size:, :, :]
    qk = einsum('bihd,bjhd->bhij', last_q, k)

    # 2. Vertical + slash pattern detection
    vertical = qk.mean(-2)  # Column importance
    slash = sum_all_diagonal_matrix(qk)  # Diagonal importance

    # 3. JS divergence for adaptive budget
    kl_div = js_divergence(avg_qk, vertical_pooled)
    is_sparse_head = kl_div > tau
    budget = gamma if is_sparse_head else 1.0

    # 4. Select blocks
    block_idx = transform_vertical_slash_idx(...)
    return block_mask
```

**配置参数**:
```python
flexprefill_gamma: float = 0.9        # 基础覆盖率
flexprefill_tau: float = 0.1          # JS 散度阈值
flexprefill_min_budget: int = 128     # 最小 token 预算
flexprefill_block_size: int = 128     # Block 大小
```

**测试**: `python tests/test_needle.py --input-len 32768 --enable-flexprefill`

### Phase 4: MInference 可选重构

**目标**: (可选) 让 MInference 也可以使用 block_sparse_attn

**修改文件**:
- `nanovllm/kvcache/sparse/minference.py`

**新增方法**:
```python
class MInferencePolicy(SparsePolicy):
    def __init__(self, ..., use_block_sparse: bool = False):
        self.use_block_sparse = use_block_sparse

    def estimate_block_mask(self, q, k, layer_id) -> BlockMask:
        """Convert vertical+slash indices to BlockMask."""
        vertical_idx, slash_idx = self.estimate_pattern(q, k, layer_id)
        return BlockMask.from_vertical_slash(vertical_idx, slash_idx, ...)

    def sparse_prefill_attention(self, q, k, v, layer_id):
        if self.use_block_sparse:
            block_mask = self.estimate_block_mask(q, k, layer_id)
            return block_sparse_attention(q, k, v, block_mask)
        else:
            # 使用原有 minference kernel
            return self._minference_kernel_attention(q, k, v, layer_id)
```

### Phase 5: 集成和测试

**任务**:
1. 更新 `__init__.py` 工厂函数支持所有策略
2. 更新 Config 添加所有配置参数
3. 添加性能基准测试脚本
4. 更新文档

---

## Part 4: 依赖管理

### 必需依赖

```
# requirements.txt 新增
block-sparse-attn    # MIT-HAN-LAB block sparse kernel
triton>=2.0          # FlexPrefill Triton kernels
```

### 安装说明

```bash
# block_sparse_attn from MIT-HAN-LAB
pip install git+https://github.com/mit-han-lab/Block-Sparse-Attention.git

# 或从本地安装（如果有）
cd /home/zijie/Code/x-attention/Block-Sparse-Attention
pip install -e .
```

---

## Part 5: 配置参数汇总

### SparsePolicyType 枚举

```python
class SparsePolicyType(str, Enum):
    FULL = "full"              # 全注意力（无稀疏）
    QUEST = "quest"            # Decode-only Top-K
    MINFERENCE = "minference"  # Prefill vertical+slash
    XATTENTION = "xattention"  # Prefill stride-based block
    FLEXPREFILL = "flexprefill"  # Prefill adaptive JS-divergence
```

### 策略参数对照表

| 策略 | 参数 | 默认值 | 说明 |
|------|-----|--------|------|
| MInference | `adaptive_budget` | 0.3 | 预算占 seq_len 比例 |
| MInference | `vertical_size` | 1000 | 固定 vertical 大小 |
| MInference | `slash_size` | 6096 | 固定 slash 大小 |
| XAttention | `stride` | 16 | Q/K 下采样步长 |
| XAttention | `threshold` | 0.9 | 累积分数阈值 |
| XAttention | `block_size` | 128 | Block 大小 |
| FlexPrefill | `gamma` | 0.9 | 基础覆盖率 |
| FlexPrefill | `tau` | 0.1 | JS 散度阈值 |
| FlexPrefill | `min_budget` | 128 | 最小 token 预算 |
| FlexPrefill | `block_size` | 128 | Block 大小 |

---

## Part 6: 成功标准

1. **正确性**: 所有三种策略通过 32K+ needle-in-haystack 测试
2. **性能**: 稀疏 prefill 比全注意力快 (>1.5x speedup at 64K)
3. **统一接口**: XAttention/FlexPrefill 使用 BlockMask + block_sparse_attn
4. **向后兼容**: 现有 MInference 配置继续工作
5. **可配置**: 所有策略参数可通过 LLM 配置设置

---

## Part 7: 风险评估

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|-----|--------|---------|
| block_sparse_attn 硬件兼容性 | 高 | 中 | 测试目标硬件，fallback 到 flash_attn |
| MInference → block mask 精度损失 | 中 | 低 | 对比测试输出差异 |
| Triton kernel 移植问题 | 中 | 中 | 使用非 Triton fallback |
| 内存开销增加 | 低 | 低 | block_size=128 → 1KB/head for 128K |

---

## References

- x-attention repo: `/home/zijie/Code/x-attention`
- MIT-HAN-LAB Block-Sparse-Attention: `https://github.com/mit-han-lab/Block-Sparse-Attention`
- MInference paper: https://arxiv.org/abs/2407.02490
- Current nanovllm sparse implementation: `nanovllm/kvcache/sparse/`
