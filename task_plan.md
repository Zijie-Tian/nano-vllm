# Sparse Prefill Attention Integration Plan

## Overview

This document analyzes the current MInference integration in nanovllm and proposes a framework modification to support all three sparse attention strategies from x-attention: **MInference**, **FlexPrefill**, and **XAttention**.

---

## Part 1: Analysis of x-attention Strategies

### 1.1 MInference (Minference.py)

**Core Algorithm:**
```
1. Use last 64 queries to compute attention to all keys
2. Identify vertical pattern: sum attention across query dim → important K columns
3. Identify slash pattern: sum attention along diagonals → important diagonal bands
4. Call vertical_slash_sparse_attention kernel with vertical_topk + slash_topk indices
```

**Key Characteristics:**
- Pattern representation: `vertical_indices [heads, topk]` + `slash_indices [heads, topk]`
- Kernel: `minference.ops.pit_sparse_flash_attention_v2.vertical_slash_sparse_attention`
- Per-head processing loop (processes one head at a time)
- Input format: `[batch, heads, seq, dim]` (BHSD)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vertical_size` | 1000 | Number of vertical (column) positions |
| `slash_size` | 6096 | Number of diagonal bands |
| `adaptive_budget` | None | If set, budget = seq_len * adaptive_budget |

---

### 1.2 XAttention (Xattention.py)

**Core Algorithm:**
```
1. Divide Q, K into blocks (block_size=128)
2. Create strided/coarse Q, K representations (stride=16)
3. Compute block-level attention scores: block_Q @ block_K.T
4. Select blocks where cumulative attention > threshold (top-p style)
5. Create boolean block_mask [batch, heads, q_blocks, k_blocks]
6. Call block_sparse_attn_func kernel with block_mask
```

**Key Characteristics:**
- Pattern representation: `block_mask [batch, heads, q_blocks, k_blocks]` (boolean)
- Kernel: `block_sparse_attn.block_sparse_attn_func` (MIT-HAN-LAB)
- Uses Triton kernels for efficient block-level score computation
- Input format: `[seq, heads, dim]` (before kernel call)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 128 | Tokens per block |
| `stride` | 16 | Stride for coarse Q/K |
| `threshold` | 0.8-0.9 | Cumulative attention threshold |
| `chunk_size` | auto | Processing chunk size |

---

### 1.3 FlexPrefill (Flexprefill.py)

**Core Algorithm:**
```
1. Use last block (block_size=128) queries to compute attention
2. Detect vertical pattern (sink tokens) + slash pattern (diagonals)
3. Compute JS divergence between estimated pattern and uniform distribution
4. Adaptive budget: high divergence → use fewer blocks, low divergence → more blocks
5. Transform vertical+slash to flattened block indices
6. Call triton_block_wise_prefill_attention with block_idx
```

**Key Characteristics:**
- Pattern representation: `block_idx` (list of flattened block indices per head)
- Kernel: Custom Triton `triton_block_wise_prefill_attention` (in Flexprefill.py)
- Supports GQA with `gqa_interleave` parameter
- Input format: `[batch, seq, heads, dim]` (BSHD)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.9 | Target attention coverage ratio |
| `tau` | 0 | JS divergence threshold for block sparsity |
| `min_budget` | 1 | Minimum blocks per row |
| `max_budget` | inf | Maximum blocks per row |
| `block_size` | 128 | Tokens per block |

---

## Part 2: Current nanovllm MInference Problems

### 2.1 Pattern Estimation Differences

**x-attention MInference:**
```python
# Diagonal sum using strided view
def sum_all_diagonal_matrix(mat):
    zero_mat = torch.zeros((b, h, n, n))
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), ...)
    sum_diags = torch.sum(mat_strided, 2)
    return sum_diags[:, :, 1:]
```

**nanovllm MInference:**
```python
# Diagonal sum using scatter_add (more memory intensive)
diag_indices = (seq_len - last_q + q_indices) - k_indices
slash_scores = torch.zeros(num_heads, seq_len, ...)
slash_scores.scatter_add_(1, diag_indices, qk)
```

**Problem:** The nanovllm implementation uses different diagonal accumulation logic that may produce different results and uses more memory.

### 2.2 Kernel Interface Mismatch

**x-attention calls:**
```python
vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
# Where slash = (q_len - 1) - topk indices (descending order)
```

**nanovllm calls:**
```python
# Uses minference internal conversion functions
block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(...)
_triton_mixed_sparse_attention(q, k, v, ..., block_count, block_offset, ...)
```

**Problem:** nanovllm uses a different (more complex) kernel interface path that requires index conversion.

### 2.3 Missing GQA Handling in Kernel

The x-attention MInference processes heads one-by-one in a loop, naturally handling GQA. nanovllm expands K/V before the kernel call, which works but is less memory efficient.

---

## Part 3: Framework Design Issues

### 3.1 Current SparsePolicy Interface

```python
class SparsePolicy(ABC):
    supports_prefill: bool = True
    supports_decode: bool = True
    requires_block_selection: bool = False

    @abstractmethod
    def select_blocks(self, available_blocks, ctx) -> List[int]:
        """For CPU offload block selection"""
        pass

    def sparse_prefill_attention(self, q, k, v, layer_id) -> Tensor:
        """GPU-only sparse attention"""
        raise NotImplementedError
```

**Problems:**
1. `sparse_prefill_attention()` assumes a monolithic Q/K/V → output interface
2. No abstraction for pattern estimation vs sparse computation phases
3. No common representation for different pattern types (vertical+slash, block_mask, block_idx)
4. Each strategy must handle tensor format conversion internally

### 3.2 Tensor Format Inconsistency

| Strategy | Input Format | Kernel Format |
|----------|--------------|---------------|
| MInference | `[seq, heads, dim]` | `[batch, heads, seq, dim]` |
| XAttention | `[batch, heads, seq, dim]` | `[seq, heads, dim]` |
| FlexPrefill | `[batch, seq, heads, dim]` | `[batch, seq, heads, dim]` |

This requires format conversion in each strategy.

### 3.3 Missing Kernel Dependencies

| Kernel | Source | Current Status |
|--------|--------|----------------|
| `vertical_slash_sparse_attention` | minference | ✓ Available |
| `block_sparse_attn_func` | MIT-HAN-LAB block_sparse_attn | ✗ Not integrated |
| `triton_block_wise_prefill_attention` | FlexPrefill custom | ✗ Not integrated |

---

## Part 4: Proposed Framework Modifications

### 4.1 Unified Sparse Prefill Strategy Interface

```python
from enum import Enum
from typing import Union, Optional
from dataclasses import dataclass

class SparsePatternType(Enum):
    """Pattern representation type"""
    VERTICAL_SLASH = "vertical_slash"  # MInference: (vertical_idx, slash_idx)
    BLOCK_MASK = "block_mask"          # XAttention: boolean [b, h, q_blk, k_blk]
    BLOCK_INDEX = "block_index"        # FlexPrefill: list of block indices

@dataclass
class SparsePrefillConfig:
    """Configuration for sparse prefill strategies"""
    # Common parameters
    block_size: int = 128

    # MInference parameters
    vertical_size: int = 1000
    slash_size: int = 6096
    adaptive_budget: Optional[float] = 0.3
    num_sink_tokens: int = 30
    num_recent_diags: int = 100

    # XAttention parameters
    stride: int = 16
    threshold: float = 0.9
    chunk_size: Optional[int] = None

    # FlexPrefill parameters
    gamma: float = 0.9
    tau: float = 0.0
    min_budget: Optional[int] = None
    max_budget: Optional[int] = None

class SparsePrefillStrategy(ABC):
    """
    Abstract base for sparse prefill attention strategies.

    Unlike SparsePolicy (for block selection), this is specifically
    for GPU sparse attention computation during prefill.
    """

    @property
    @abstractmethod
    def pattern_type(self) -> SparsePatternType:
        """Return the pattern representation type"""
        pass

    @abstractmethod
    def estimate_pattern(
        self,
        q: torch.Tensor,  # [seq, heads, dim]
        k: torch.Tensor,  # [seq, kv_heads, dim]
        layer_id: int,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Estimate sparse attention pattern.

        Returns pattern in strategy-specific format:
        - VERTICAL_SLASH: (vertical_indices, slash_indices)
        - BLOCK_MASK: boolean mask [batch, heads, q_blocks, k_blocks]
        - BLOCK_INDEX: list of block index tensors
        """
        pass

    @abstractmethod
    def sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: Union[Tuple[Tensor, Tensor], Tensor],
        layer_id: int,
    ) -> torch.Tensor:
        """
        Compute sparse attention using the estimated pattern.

        Returns: attention output [seq, heads, dim]
        """
        pass

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Full sparse prefill: estimate pattern + compute attention.

        Default implementation calls estimate_pattern then sparse_attention.
        Can be overridden for fused implementations.
        """
        pattern = self.estimate_pattern(q, k, layer_id)
        return self.sparse_attention(q, k, v, pattern, layer_id)
```

### 4.2 Strategy Implementations

```
nanovllm/kvcache/sparse/
├── __init__.py
├── policy.py                    # SparsePolicy (block selection, unchanged)
├── full_policy.py               # FullAttentionPolicy (unchanged)
├── quest.py                     # QuestPolicy (unchanged)
├── prefill/                     # NEW: Sparse prefill strategies
│   ├── __init__.py
│   ├── strategy.py              # SparsePrefillStrategy base class
│   ├── minference.py            # MInferenceStrategy
│   ├── xattention.py            # XAttentionStrategy
│   ├── flexprefill.py           # FlexPrefillStrategy
│   └── kernels/                 # Kernel wrappers
│       ├── __init__.py
│       ├── minference_kernel.py   # Wrap minference ops
│       ├── block_sparse_kernel.py # Wrap block_sparse_attn
│       └── flexprefill_kernel.py  # FlexPrefill Triton kernels
```

### 4.3 MInference Strategy (Fixed)

```python
# nanovllm/kvcache/sparse/prefill/minference.py

class MInferenceStrategy(SparsePrefillStrategy):
    """Fixed MInference using x-attention algorithm"""

    pattern_type = SparsePatternType.VERTICAL_SLASH

    def __init__(self, config: SparsePrefillConfig):
        self.vertical_size = config.vertical_size
        self.slash_size = config.slash_size
        self.adaptive_budget = config.adaptive_budget
        # Cached mask
        self._last_q_mask = None

    def _sum_all_diagonal_matrix(self, mat: torch.Tensor) -> torch.Tensor:
        """Use x-attention's strided view approach for diagonal sum"""
        b, h, n, m = mat.shape
        zero_mat = torch.zeros((b, h, n, n), device=mat.device, dtype=mat.dtype)
        mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
        mat_strided = mat_padded.as_strided(
            (b, h, n, n + m),
            (h * n * (2 * n + m), n * (2 * n + m), 2 * n + m + 1, 1)
        )
        sum_diags = torch.sum(mat_strided, 2)
        return sum_diags[:, :, 1:]

    def estimate_pattern(self, q, k, layer_id):
        """Estimate vertical + slash pattern using last-64 attention"""
        seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]

        # Compute budget
        if self.adaptive_budget is not None:
            budget = int(seq_len * self.adaptive_budget)
            vertical_size = int(budget * 0.2)
            slash_size = int(budget * 0.8)
        else:
            vertical_size = min(self.vertical_size, seq_len)
            slash_size = min(self.slash_size, seq_len)

        # Last-q attention (use x-attention format: [batch, heads, seq, dim])
        last_q = min(64, seq_len)
        q_last = q[-last_q:].unsqueeze(0).transpose(1, 2)  # [1, heads, last_q, dim]

        # Expand K for GQA
        if num_kv_heads < num_heads:
            k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        else:
            k_expanded = k
        k_batched = k_expanded.unsqueeze(0).transpose(1, 2)  # [1, heads, seq, dim]

        # Compute attention: [1, heads, last_q, seq]
        scale = 1.0 / math.sqrt(head_dim)
        qk = torch.einsum('bhqd,bhkd->bhqk', q_last, k_batched) * scale

        # Causal mask for last-q
        if self._last_q_mask is None or self._last_q_mask.shape != (last_q, last_q):
            arange = torch.arange(last_q, device=q.device)
            self._last_q_mask = arange[None, :] >= arange[:, None]
        qk[:, :, :, -last_q:] = torch.where(
            self._last_q_mask.unsqueeze(0).unsqueeze(0),
            qk[:, :, :, -last_q:],
            torch.tensor(float('-inf'), device=q.device)
        )

        # Softmax
        qk = F.softmax(qk, dim=-1, dtype=torch.float32)

        # Vertical: sum across query dim
        vertical = qk.sum(dim=2, keepdim=True)  # [1, heads, 1, seq]
        vertical[..., :30] = float('inf')  # Keep sink tokens
        vertical_topk = vertical.squeeze(2).topk(vertical_size, dim=-1).indices  # [1, heads, topk]

        # Slash: sum along diagonals using x-attention method
        slash = self._sum_all_diagonal_matrix(qk)[..., :-last_q+1]  # [1, heads, seq-last_q+1]
        slash[..., -100:] = float('inf')  # Keep recent diagonals
        slash_topk = (seq_len - 1) - slash.topk(slash_size, dim=-1).indices  # [1, heads, topk]

        return (vertical_topk.squeeze(0), slash_topk.squeeze(0))

    def sparse_attention(self, q, k, v, pattern, layer_id):
        """Call vertical_slash_sparse_attention per head"""
        from minference.ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention

        vertical_indices, slash_indices = pattern
        seq_len, num_heads, head_dim = q.shape

        # Format: [batch, heads, seq, dim]
        q_batched = q.unsqueeze(0).transpose(1, 2)
        k_batched = k.unsqueeze(0).transpose(1, 2)
        v_batched = v.unsqueeze(0).transpose(1, 2)

        # Handle GQA
        num_kv_heads = k.shape[1]
        if num_kv_heads < num_heads:
            groups = num_heads // num_kv_heads
            k_batched = k_batched.repeat_interleave(groups, dim=1)
            v_batched = v_batched.repeat_interleave(groups, dim=1)

        # Process each head (x-attention style)
        output = torch.empty_like(q_batched)
        for head in range(num_heads):
            q_h = q_batched[:, head:head+1]
            k_h = k_batched[:, head:head+1]
            v_h = v_batched[:, head:head+1]
            v_idx = vertical_indices[head:head+1].unsqueeze(0)
            s_idx = slash_indices[head:head+1].unsqueeze(0)

            output[:, head:head+1] = vertical_slash_sparse_attention(
                q_h, k_h, v_h, v_idx, s_idx
            )

        # Back to [seq, heads, dim]
        return output.transpose(1, 2).squeeze(0)
```

### 4.4 XAttention Strategy

```python
# nanovllm/kvcache/sparse/prefill/xattention.py

class XAttentionStrategy(SparsePrefillStrategy):
    """XAttention block sparse attention"""

    pattern_type = SparsePatternType.BLOCK_MASK

    def __init__(self, config: SparsePrefillConfig):
        self.block_size = config.block_size
        self.stride = config.stride
        self.threshold = config.threshold
        self.chunk_size = config.chunk_size

    def estimate_pattern(self, q, k, layer_id):
        """Estimate block mask using strided attention"""
        # Import x-attention estimation function
        from nanovllm.kvcache.sparse.prefill.kernels.xattention_kernel import xattn_estimate

        seq_len, num_heads, head_dim = q.shape

        # Convert to [batch, heads, seq, dim]
        q_batched = q.unsqueeze(0).transpose(1, 2)
        k_batched = k.unsqueeze(0).transpose(1, 2)

        # Handle GQA
        num_kv_heads = k.shape[1]
        if num_kv_heads < num_heads:
            k_batched = k_batched.repeat_interleave(num_heads // num_kv_heads, dim=1)

        # Compute block mask
        _, block_mask = xattn_estimate(
            q_batched, k_batched,
            block_size=self.block_size,
            stride=self.stride,
            threshold=self.threshold,
            chunk_size=self.chunk_size or 16384,
        )

        return block_mask  # [batch, heads, q_blocks, k_blocks] boolean

    def sparse_attention(self, q, k, v, pattern, layer_id):
        """Call block_sparse_attn_func"""
        from block_sparse_attn import block_sparse_attn_func

        block_mask = pattern
        seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]

        # block_sparse_attn_func expects [seq, heads, dim]
        q_work = q

        # Handle GQA
        if num_kv_heads < num_heads:
            k_work = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v_work = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
        else:
            k_work = k
            v_work = v

        # Prepare cumulative sequence lengths
        q_cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=q.device)
        k_cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=q.device)
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device=q.device)

        q_block_num = (seq_len + self.block_size - 1) // self.block_size
        k_block_num = (seq_len + self.block_size - 1) // self.block_size

        output = block_sparse_attn_func(
            q_work, k_work, v_work,
            q_cu_seqlens, k_cu_seqlens,
            head_mask_type,
            None,  # leftpad_k
            block_mask[:, :, :q_block_num, :k_block_num].contiguous(),
            seq_len, seq_len,
            p_dropout=0.0,
            deterministic=True,
            is_causal=True,
        )

        return output
```

### 4.5 FlexPrefill Strategy

```python
# nanovllm/kvcache/sparse/prefill/flexprefill.py

class FlexPrefillStrategy(SparsePrefillStrategy):
    """FlexPrefill adaptive sparse attention"""

    pattern_type = SparsePatternType.BLOCK_INDEX

    def __init__(self, config: SparsePrefillConfig):
        self.block_size = config.block_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.min_budget = config.min_budget
        self.max_budget = config.max_budget

    def estimate_pattern(self, q, k, layer_id):
        """Estimate block indices using vertical+slash + JS divergence"""
        from nanovllm.kvcache.sparse.prefill.kernels.flexprefill_kernel import get_active_blocks

        seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]

        # Convert to [batch, seq, heads, dim] (FlexPrefill format)
        q_batched = q.unsqueeze(0)  # [1, seq, heads, dim]
        k_batched = k.unsqueeze(0)  # [1, seq, kv_heads, dim]
        v_dummy = k_batched  # Not used for pattern estimation

        num_blocks = math.ceil(seq_len / self.block_size)
        min_budget = self.min_budget or 1
        max_budget = self.max_budget or num_blocks

        block_idx = get_active_blocks(
            q_batched, k_batched, v_dummy,
            self.block_size,
            self.gamma,
            math.ceil(min_budget / self.block_size),
            math.ceil(max_budget / self.block_size),
            self.tau,
            gqa_interleave=False,
        )

        return block_idx  # List[List[Tensor]] per batch per head

    def sparse_attention(self, q, k, v, pattern, layer_id):
        """Call triton_block_wise_prefill_attention"""
        from nanovllm.kvcache.sparse.prefill.kernels.flexprefill_kernel import (
            triton_block_wise_prefill_attention
        )

        block_idx = pattern
        seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]

        # Convert to [batch, seq, heads, dim]
        q_batched = q.unsqueeze(0)
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)

        output = triton_block_wise_prefill_attention(
            q_batched.to(torch.bfloat16),
            k_batched.to(torch.bfloat16),
            v_batched.to(torch.bfloat16),
            block_idx,
            self.block_size,
        )

        return output.squeeze(0)  # [seq, heads, dim]
```

### 4.6 Factory Function Update

```python
# nanovllm/kvcache/sparse/prefill/__init__.py

from enum import Enum

class SparsePrefillType(Enum):
    FULL = "full"           # No sparse, use flash attention
    MINFERENCE = "minfer"   # MInference vertical+slash
    XATTENTION = "xattn"    # XAttention block mask
    FLEXPREFILL = "flex"    # FlexPrefill adaptive

def create_sparse_prefill_strategy(
    strategy_type: SparsePrefillType,
    config: SparsePrefillConfig = None,
) -> Optional[SparsePrefillStrategy]:
    """Create a sparse prefill strategy instance"""

    if strategy_type == SparsePrefillType.FULL:
        return None  # Use standard flash attention

    config = config or SparsePrefillConfig()

    if strategy_type == SparsePrefillType.MINFERENCE:
        return MInferenceStrategy(config)
    elif strategy_type == SparsePrefillType.XATTENTION:
        return XAttentionStrategy(config)
    elif strategy_type == SparsePrefillType.FLEXPREFILL:
        return FlexPrefillStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")
```

---

## Part 5: Integration with Model Runner

### 5.1 Update Context

```python
# nanovllm/utils/context.py

@dataclass
class AttentionContext:
    # Existing fields...

    # Replace sparse_prefill_policy with sparse_prefill_strategy
    sparse_prefill_strategy: Optional["SparsePrefillStrategy"] = None
```

### 5.2 Update Attention Layer

```python
# nanovllm/layers/attention.py

def forward(self, ...):
    # ... QKV projection, RoPE, store KV ...

    if context.is_prefill:
        if context.sparse_prefill_strategy is not None:
            # Use sparse prefill strategy
            o = context.sparse_prefill_strategy.forward(q, k, v, self.layer_id)
        else:
            # Standard flash attention
            o = flash_attn_varlen_func(q, k, v, ...)
    else:
        # Decode path (unchanged)
        ...
```

### 5.3 Configuration

```python
# nanovllm/config.py

class Config:
    # Existing fields...

    # Sparse prefill configuration
    sparse_prefill_type: SparsePrefillType = SparsePrefillType.FULL
    sparse_prefill_config: SparsePrefillConfig = field(default_factory=SparsePrefillConfig)
```

---

## Part 6: Implementation Phases

### Phase 1: Framework Setup (Foundation)
1. Create `nanovllm/kvcache/sparse/prefill/` directory structure
2. Implement `SparsePrefillStrategy` base class
3. Implement `SparsePrefillConfig` dataclass
4. Add `SparsePrefillType` enum
5. Create factory function

### Phase 2: Kernel Integration
1. Wrap `minference.ops` for MInference kernel
2. Integrate `block_sparse_attn` for XAttention kernel
3. Port FlexPrefill Triton kernels from x-attention

### Phase 3: Strategy Implementations
1. Implement `MInferenceStrategy` using x-attention algorithm
2. Implement `XAttentionStrategy`
3. Implement `FlexPrefillStrategy`

### Phase 4: Model Runner Integration
1. Update `AttentionContext` with sparse strategy
2. Modify `attention.py` to use strategy
3. Update `model_runner.py` for configuration
4. Add CLI flags for strategy selection

### Phase 5: Testing & Validation
1. Unit tests for each strategy
2. Needle-in-haystack accuracy tests
3. Performance benchmarks vs full attention
4. Memory usage validation

---

## Part 7: Dependency Requirements

```
# Required packages
minference>=0.1.0      # For vertical_slash_sparse_attention
block-sparse-attn      # MIT-HAN-LAB block sparse kernel (pip install block-sparse-attn)
triton>=2.0.0          # For FlexPrefill custom kernels
```

---

## Part 8: Summary

| Strategy | Pattern Type | Kernel Source | Current Status |
|----------|--------------|---------------|----------------|
| MInference | vertical+slash | minference | Partial (needs fix) |
| XAttention | block_mask | block_sparse_attn | Not integrated |
| FlexPrefill | block_idx | Custom Triton | Not integrated |

**Key Changes:**
1. New `SparsePrefillStrategy` base class (separate from `SparsePolicy`)
2. Unified configuration via `SparsePrefillConfig`
3. Strategy pattern with `estimate_pattern()` + `sparse_attention()` phases
4. Factory function for strategy creation
5. Kernel wrappers to abstract external dependencies

**Benefits:**
- Clean separation: pattern estimation vs sparse computation
- Easy to add new strategies (implement 2 methods)
- Reusable kernels across strategies
- Consistent tensor format handling
- Single configuration point
