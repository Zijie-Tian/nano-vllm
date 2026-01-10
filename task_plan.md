# Integration Plan: Unified Sparse Prefill Attention Framework

## Executive Summary

This plan describes how to integrate three sparse prefill attention strategies (MInference, XAttention, FlexPrefill) from the x-attention repository into nanovllm's sparse attention framework. The goal is to create a unified, extensible architecture that supports all three strategies with a common interface.

---

## Part 1: Analysis of Current State

### 1.1 x-attention Repository Strategies

| Strategy | Pattern Type | Estimation Method | Output Format | Kernel Backend |
|----------|-------------|-------------------|---------------|----------------|
| **MInference** | Vertical + Slash | Last-64-Q attention → sum columns/diagonals | vertical_indices, slash_indices | `vertical_slash_sparse_attention` (minference lib) |
| **XAttention** | Block threshold | Stride-based Q/K downsampling → block attention | Boolean block mask `[B,H,Qb,Kb]` | `block_sparse_attn_func` (block_sparse_attn lib) |
| **FlexPrefill** | Adaptive V+S | Last-block attention + JS divergence budget | Flattened block indices | `triton_block_wise_attention` (custom triton) |

### 1.2 Strategy Details

#### MInference (`xattn/src/Minference.py`)
```python
# Pattern estimation
last_q = q[-64:]  # Last 64 queries
qk = einsum('qhd,khd->hqk', last_q, k)  # [heads, 64, seq_len]
qk = softmax(qk)

# Vertical: sum across query dimension
vertical_scores = qk.sum(dim=1)  # [heads, seq_len]
vertical_indices = vertical_scores.topk(vertical_size).indices

# Slash: sum along diagonals
slash_scores = sum_all_diagonal_matrix(qk)  # [heads, seq_len]
slash_indices = slash_scores.topk(slash_size).indices

# Attention via minference kernel
output = vertical_slash_sparse_attention(q, k, v, vertical_indices, slash_indices)
```

#### XAttention (`xattn/src/Xattention.py`)
```python
# Pattern estimation via xattn_estimate
# 1. Downsample Q/K with stride
reshaped_k = cat([k[:, :, i::stride, :] for i in range(stride)], dim=-1)
reshaped_q = cat([q[:, :, stride-1-i::stride, :] for i in range(stride)], dim=-1)

# 2. Compute block-level attention
attn_weights = matmul(reshaped_q, reshaped_k.T) / sqrt(head_dim) / stride
attn_sum = softmax(attn_weights).view(..., num_blocks, block_size).sum(dim=-1)

# 3. Threshold selection
block_mask = find_blocks_chunked(attn_sum, threshold=0.9)  # [B, H, Qb, Kb]

# Attention via block_sparse_attn
output = block_sparse_attn_func(q, k, v, block_mask, block_size=128)
```

#### FlexPrefill (`xattn/src/Flexprefill.py`)
```python
# 1. Last-block attention analysis
last_q = q[:, -block_size:, :, :]
qk = einsum('bihd,bjhd->bhij', last_q, k)  # [B, H, block_size, seq_len]

# 2. Detect vertical and slash patterns
vertical = qk.mean(-2)  # Column importance
slash = sum_all_diagonal_matrix(qk) / qk.shape[-2]  # Diagonal importance

# 3. Adaptive budget via JS divergence
kl_div = js_divergence(avg_qk, vertical)  # Per-head divergence
budget = adjust_budget(gamma, kl_div, tau)  # Reduce budget if sparse

# 4. Select blocks
vertical_topk = vertical.topk(num_vertical_blocks).indices
slash_topk = slash.topk(num_slash_blocks).indices
block_idx = transform_vertical_slash_idx(vertical_topk, slash_topk, num_blocks)

# Attention via custom triton kernel
output = triton_block_wise_attention(q, k, v, block_idx, block_size)
```

### 1.3 Current nanovllm Implementation

**Files:**
- `nanovllm/kvcache/sparse/policy.py` - Base `SparsePolicy` class
- `nanovllm/kvcache/sparse/minference.py` - MInference implementation
- `nanovllm/kvcache/sparse/quest.py` - Quest (decode-only) policy
- `nanovllm/kvcache/sparse/__init__.py` - Factory function

**Current MInference Implementation:**
```python
class MInferencePolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False
    requires_block_selection = False

    def sparse_prefill_attention(self, q, k, v, layer_id):
        # Estimate pattern
        vertical_indices, slash_indices = self.estimate_pattern(q, k, layer_id)

        # Execute via minference kernel
        output = _triton_mixed_sparse_attention(q, k, v, vertical_indices, slash_indices)
        return output
```

### 1.4 Issues with Current Implementation

| Issue | Description | Impact |
|-------|-------------|--------|
| **Tightly coupled estimation & attention** | `sparse_prefill_attention()` does both | Cannot reuse estimation for CPU offload |
| **No unified block mask format** | MInference uses indices, not masks | Cannot share kernel backends |
| **Different block sizes** | MInference: token-level, XAttention: 128 | Inconsistent abstraction |
| **Different kernel backends** | Each strategy has its own kernel | Code duplication, maintenance burden |
| **Scattered GQA handling** | Each strategy handles GQA differently | Inconsistent behavior |
| **No pattern caching** | Pattern estimated fresh each layer | Could cache for similar layers |

---

## Part 2: Proposed Unified Architecture

### 2.1 Design Principles

1. **Separation of Concerns**: Separate pattern estimation from attention execution
2. **Common Block Mask Abstraction**: All strategies produce boolean block masks
3. **Pluggable Kernel Backend**: Support multiple attention kernels
4. **Strategy Pattern**: Each sparse method implements a common interface
5. **CPU Offload Ready**: Pattern estimation can guide block loading

### 2.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Unified Sparse Attention Framework                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │   MInference    │  │   XAttention    │  │   FlexPrefill   │  Strategies   │
│  │   Estimator     │  │   Estimator     │  │   Estimator     │               │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘               │
│           │                    │                    │                         │
│           └────────────────────┼────────────────────┘                         │
│                                ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         BlockMask                                        │ │
│  │    [batch, num_heads, q_blocks, k_blocks] - boolean tensor               │ │
│  │    + metadata (block_size, valid_blocks, etc.)                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                │                                              │
│                                ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Sparse Attention Kernel Backend                       │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │ │
│  │  │ BlockSparse  │  │  Triton      │  │  MInference  │  (selectable)     │ │
│  │  │ Attn (HAN)   │  │  BlockWise   │  │  V+S Kernel  │                   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                │                                              │
│                                ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Attention Output                                 │ │
│  │                    [seq_len, num_heads, head_dim]                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 New Class Hierarchy

```python
# Base estimator interface
class BlockMaskEstimator(ABC):
    """Abstract base class for block mask estimation strategies."""

    @abstractmethod
    def estimate(self, q: Tensor, k: Tensor, layer_id: int) -> BlockMask:
        """Estimate which blocks to attend to."""
        pass

    @property
    @abstractmethod
    def block_size(self) -> int:
        """Block size used by this estimator."""
        pass

# Block mask container
@dataclass
class BlockMask:
    """Container for block-level attention mask."""
    mask: torch.Tensor  # [batch, heads, q_blocks, k_blocks]
    block_size: int
    num_q_blocks: int
    num_k_blocks: int

    def to_dense_mask(self) -> torch.Tensor:
        """Convert to token-level mask [batch, heads, q_len, k_len]."""
        pass

    def sparsity_ratio(self) -> float:
        """Fraction of blocks that are masked out."""
        return 1.0 - self.mask.float().mean().item()

# Sparse attention executor
class SparseAttentionExecutor:
    """Execute sparse attention using block mask."""

    def __init__(self, backend: str = "block_sparse"):
        self.backend = backend  # "block_sparse", "triton", or "minference"

    def forward(self, q, k, v, block_mask: BlockMask) -> Tensor:
        if self.backend == "block_sparse":
            return self._block_sparse_attn(q, k, v, block_mask)
        elif self.backend == "triton":
            return self._triton_block_attn(q, k, v, block_mask)
        # ...
```

### 2.4 Concrete Estimators

```python
class MInferenceEstimator(BlockMaskEstimator):
    """MInference: vertical + slash pattern estimation."""

    def __init__(self, adaptive_budget=0.3, block_size=64):
        self.adaptive_budget = adaptive_budget
        self._block_size = block_size

    def estimate(self, q, k, layer_id) -> BlockMask:
        # 1. Estimate vertical & slash indices (existing logic)
        vertical_idx, slash_idx = self._estimate_pattern(q, k)

        # 2. Convert to block mask
        block_mask = self._indices_to_block_mask(vertical_idx, slash_idx)

        return BlockMask(mask=block_mask, block_size=self._block_size, ...)

class XAttentionEstimator(BlockMaskEstimator):
    """XAttention: stride-based block estimation."""

    def __init__(self, stride=16, threshold=0.9, block_size=128):
        self.stride = stride
        self.threshold = threshold
        self._block_size = block_size

    def estimate(self, q, k, layer_id) -> BlockMask:
        # Use xattn_estimate logic
        _, block_mask = xattn_estimate(
            q, k, block_size=self._block_size,
            stride=self.stride, threshold=self.threshold
        )
        return BlockMask(mask=block_mask, block_size=self._block_size, ...)

class FlexPrefillEstimator(BlockMaskEstimator):
    """FlexPrefill: adaptive vertical+slash with JS divergence."""

    def __init__(self, gamma=0.9, tau=0.1, block_size=128):
        self.gamma = gamma
        self.tau = tau
        self._block_size = block_size

    def estimate(self, q, k, layer_id) -> BlockMask:
        # Analyze last block, compute JS divergence, select blocks
        block_idx = get_active_blocks(q, k, v=None, ...)
        block_mask = self._indices_to_mask(block_idx)
        return BlockMask(mask=block_mask, block_size=self._block_size, ...)
```

---

## Part 3: Implementation Plan

### Phase 1: Core Infrastructure (2-3 days)

#### 1.1 Create BlockMask Abstraction
**File:** `nanovllm/kvcache/sparse/block_mask.py`

```python
@dataclass
class BlockMask:
    mask: torch.Tensor  # [batch, heads, q_blocks, k_blocks]
    block_size: int
    seq_len: int  # Original sequence length
    num_q_blocks: int
    num_k_blocks: int
    device: torch.device

    @classmethod
    def from_indices(cls, block_indices: torch.Tensor, num_blocks: int, ...) -> "BlockMask":
        """Convert flattened block indices to boolean mask."""
        pass

    @classmethod
    def from_vertical_slash(cls, vertical_idx, slash_idx, num_blocks, ...) -> "BlockMask":
        """Convert MInference-style indices to block mask."""
        pass

    def to_flat_indices(self) -> torch.Tensor:
        """Convert back to flattened block indices."""
        pass

    def apply_causal_mask(self) -> "BlockMask":
        """Apply causal constraint: only lower triangular blocks."""
        pass
```

#### 1.2 Create Estimator Base Class
**File:** `nanovllm/kvcache/sparse/estimator.py`

```python
class BlockMaskEstimator(ABC):
    """Base class for sparse pattern estimation."""

    supports_gqa: bool = True

    @abstractmethod
    def estimate(self, q: Tensor, k: Tensor, layer_id: int) -> BlockMask:
        pass

    @property
    @abstractmethod
    def block_size(self) -> int:
        pass

    def reset(self) -> None:
        """Reset any cached state."""
        pass
```

#### 1.3 Create Kernel Executor
**File:** `nanovllm/kvcache/sparse/executor.py`

```python
class SparseAttentionExecutor:
    """Unified sparse attention execution."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend

    def forward(
        self,
        q: Tensor,  # [seq, heads, dim]
        k: Tensor,  # [seq, kv_heads, dim]
        v: Tensor,  # [seq, kv_heads, dim]
        block_mask: BlockMask,
    ) -> Tensor:
        # Handle GQA expansion if needed
        if k.shape[1] < q.shape[1]:
            k, v = self._expand_gqa(k, v, q.shape[1])

        # Dispatch to appropriate kernel
        if self.backend == "block_sparse":
            return self._call_block_sparse(q, k, v, block_mask)
        elif self.backend == "triton":
            return self._call_triton_block(q, k, v, block_mask)
        else:
            return self._call_fallback(q, k, v, block_mask)
```

### Phase 2: Implement Estimators (3-4 days)

#### 2.1 XAttention Estimator
**File:** `nanovllm/kvcache/sparse/xattention.py`

Port the `xattn_estimate` logic from x-attention repo:
- Stride-based Q/K downsampling
- Block-level softmax attention
- Threshold-based selection via `find_blocks_chunked`

```python
class XAttentionEstimator(BlockMaskEstimator):
    def __init__(
        self,
        stride: int = 16,
        threshold: float = 0.9,
        block_size: int = 128,
        chunk_size: int = 16384,
        use_triton: bool = True,
    ):
        self.stride = stride
        self.threshold = threshold
        self._block_size = block_size
        self.chunk_size = chunk_size
        self.use_triton = use_triton

    def estimate(self, q, k, layer_id) -> BlockMask:
        # Port xattn_estimate from x-attention
        ...
```

#### 2.2 FlexPrefill Estimator
**File:** `nanovllm/kvcache/sparse/flexprefill.py`

Port the FlexPrefill logic:
- Last-block attention analysis
- Vertical + slash pattern detection
- JS divergence-based adaptive budget
- Per-head block selection

```python
class FlexPrefillEstimator(BlockMaskEstimator):
    def __init__(
        self,
        gamma: float = 0.9,
        tau: float = 0.1,
        min_budget: int = 1,
        max_budget: Optional[int] = None,
        block_size: int = 128,
    ):
        self.gamma = gamma
        self.tau = tau
        self.min_budget = min_budget
        self.max_budget = max_budget
        self._block_size = block_size

    def estimate(self, q, k, layer_id) -> BlockMask:
        # Port get_active_blocks from x-attention
        ...
```

#### 2.3 Refactor MInference Estimator
**File:** `nanovllm/kvcache/sparse/minference.py`

Refactor existing MInferencePolicy to use new abstraction:

```python
class MInferenceEstimator(BlockMaskEstimator):
    def __init__(
        self,
        vertical_size: int = 1000,
        slash_size: int = 6096,
        adaptive_budget: Optional[float] = 0.3,
        block_size: int = 64,  # MInference uses smaller blocks
    ):
        ...

    def estimate(self, q, k, layer_id) -> BlockMask:
        # Existing estimate_pattern logic
        vertical_idx, slash_idx = self._estimate_pattern(q, k)

        # NEW: Convert to BlockMask
        return BlockMask.from_vertical_slash(vertical_idx, slash_idx, ...)
```

### Phase 3: Integration with SparsePolicy (2 days)

#### 3.1 Refactor SparsePolicy
**File:** `nanovllm/kvcache/sparse/policy.py`

```python
class SparsePolicy(ABC):
    supports_prefill: bool = True
    supports_decode: bool = True
    requires_block_selection: bool = False

    # NEW: Estimator and executor
    estimator: Optional[BlockMaskEstimator] = None
    executor: Optional[SparseAttentionExecutor] = None

    def sparse_prefill_attention(self, q, k, v, layer_id) -> Tensor:
        """Default implementation using estimator + executor."""
        if self.estimator is None or self.executor is None:
            raise NotImplementedError("Must set estimator and executor")

        block_mask = self.estimator.estimate(q, k, layer_id)
        return self.executor.forward(q, k, v, block_mask)
```

#### 3.2 Create Unified Policy Classes

```python
class MInferencePolicy(SparsePolicy):
    def __init__(self, **kwargs):
        self.estimator = MInferenceEstimator(**kwargs)
        self.executor = SparseAttentionExecutor(backend="minference")

class XAttentionPolicy(SparsePolicy):
    def __init__(self, **kwargs):
        self.estimator = XAttentionEstimator(**kwargs)
        self.executor = SparseAttentionExecutor(backend="block_sparse")

class FlexPrefillPolicy(SparsePolicy):
    def __init__(self, **kwargs):
        self.estimator = FlexPrefillEstimator(**kwargs)
        self.executor = SparseAttentionExecutor(backend="triton")
```

#### 3.3 Update Factory Function
**File:** `nanovllm/kvcache/sparse/__init__.py`

```python
def create_sparse_policy(policy_type: SparsePolicyType, **kwargs) -> SparsePolicy:
    if policy_type == SparsePolicyType.MINFERENCE:
        return MInferencePolicy(**kwargs)
    elif policy_type == SparsePolicyType.XATTENTION:
        return XAttentionPolicy(**kwargs)
    elif policy_type == SparsePolicyType.FLEXPREFILL:
        return FlexPrefillPolicy(**kwargs)
    # ...
```

### Phase 4: Configuration & Testing (2 days)

#### 4.1 Update Config
**File:** `nanovllm/config.py`

```python
class SparsePolicyType(Enum):
    FULL = auto()
    QUEST = auto()
    MINFERENCE = auto()
    XATTENTION = auto()  # NEW
    FLEXPREFILL = auto()  # NEW

@dataclass
class Config:
    # Existing fields...

    # XAttention configuration
    xattn_stride: int = 16
    xattn_threshold: float = 0.9
    xattn_block_size: int = 128

    # FlexPrefill configuration
    flexprefill_gamma: float = 0.9
    flexprefill_tau: float = 0.1
    flexprefill_block_size: int = 128
```

#### 4.2 Create Tests

**File:** `tests/test_sparse_estimators.py`
- Test each estimator's `estimate()` method
- Verify block mask shapes and causal constraints
- Test GQA handling

**File:** `tests/test_sparse_attention.py`
- End-to-end tests with needle-in-haystack
- Compare outputs between strategies
- Benchmark sparsity ratios

### Phase 5: CPU Offload Integration (3 days)

#### 5.1 Pattern-Guided Block Loading

For CPU offload mode, use the estimated block mask to guide which blocks to load:

```python
# In offload_engine.py
def load_layer_kv_with_pattern(
    self,
    layer_id: int,
    block_mask: BlockMask,
    cpu_block_ids: List[int],
) -> Tensor:
    """Load only blocks that are True in block_mask."""
    # Identify which blocks need loading based on mask
    needed_blocks = self._get_needed_blocks(block_mask)

    # Load only needed blocks
    return self._load_blocks(layer_id, needed_blocks)
```

#### 5.2 Chunked Sparse Prefill

For very long sequences with CPU offload:

```python
def run_chunked_sparse_prefill(self, seqs):
    for chunk_idx, chunk in enumerate(chunks):
        # Estimate pattern for this chunk
        block_mask = estimator.estimate(q_chunk, k_full, layer_id)

        # Load needed KV blocks from CPU
        k_sparse, v_sparse = offload_engine.load_layer_kv_with_pattern(
            layer_id, block_mask, cpu_block_ids
        )

        # Execute sparse attention
        output = executor.forward(q_chunk, k_sparse, v_sparse, block_mask)
```

---

## Part 4: File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `nanovllm/kvcache/sparse/block_mask.py` | BlockMask dataclass and utilities |
| `nanovllm/kvcache/sparse/estimator.py` | BlockMaskEstimator base class |
| `nanovllm/kvcache/sparse/executor.py` | SparseAttentionExecutor class |
| `nanovllm/kvcache/sparse/xattention.py` | XAttention estimator |
| `nanovllm/kvcache/sparse/flexprefill.py` | FlexPrefill estimator |
| `tests/test_sparse_estimators.py` | Unit tests for estimators |
| `tests/test_sparse_attention.py` | Integration tests |

### Modified Files

| File | Changes |
|------|---------|
| `nanovllm/kvcache/sparse/policy.py` | Add estimator/executor attributes |
| `nanovllm/kvcache/sparse/minference.py` | Refactor to MInferenceEstimator |
| `nanovllm/kvcache/sparse/__init__.py` | Add new policies to factory |
| `nanovllm/config.py` | Add SparsePolicyType enums and config fields |
| `nanovllm/layers/attention.py` | (Optional) Direct integration point |
| `nanovllm/kvcache/offload_engine.py` | Pattern-guided block loading |
| `docs/sparse_attention_guide.md` | Update documentation |

---

## Part 5: Dependencies

### Required Libraries

| Library | Purpose | Installation |
|---------|---------|--------------|
| `block_sparse_attn` | MIT-HAN-LAB block sparse kernel | `pip install block-sparse-attn` |
| `minference` | MInference CUDA kernels | `pip install minference` |
| `triton` | FlexPrefill custom kernels | `pip install triton` |
| `flash_attn` | Baseline attention | Already installed |

### Kernel Compatibility

| Kernel | Block Sizes | GQA Support | Causal Support |
|--------|-------------|-------------|----------------|
| `block_sparse_attn_func` | 16, 32, 64, 128 | Yes (via head_mask_type) | Yes |
| `triton_block_wise_attention` | 32, 64, 128 | Yes (via gqa_interleave) | Yes |
| `vertical_slash_sparse_attention` | N/A (token-level) | Yes | Yes |

---

## Part 6: Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Kernel incompatibility | Some strategies may not work | Support multiple backends, fallback to full attention |
| Memory overhead from block mask | Extra memory for large sequences | Use sparse representation for masks, lazy evaluation |
| Performance regression | Estimation overhead | Cache patterns across similar layers, optimize Triton kernels |
| Numerical differences | Different strategies give different results | Document expected behavior, provide correctness tests |

---

## Part 7: Success Criteria

1. **Correctness**: All three strategies pass needle-in-haystack test at 32K+ context
2. **Performance**: Prefill speed >= full attention baseline (sparse computation saves time)
3. **Memory**: Peak memory <= full attention (pattern estimation doesn't add significant overhead)
4. **Extensibility**: Adding a new strategy requires only implementing `BlockMaskEstimator`
5. **Compatibility**: Works with both GPU-only and CPU offload modes

---

## Part 8: Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Core Infrastructure | 2-3 days | BlockMask, Estimator base, Executor |
| Phase 2: Implement Estimators | 3-4 days | XAttention, FlexPrefill, MInference refactor |
| Phase 3: Policy Integration | 2 days | Unified SparsePolicy, factory update |
| Phase 4: Config & Testing | 2 days | Config fields, unit tests, integration tests |
| Phase 5: CPU Offload | 3 days | Pattern-guided loading, chunked sparse prefill |
| **Total** | **12-14 days** | Full integration |

---

## Appendix: Key Code Snippets from x-attention

### A.1 xattn_estimate Core Logic
```python
# From xattn/src/Xattention.py
reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1)
reshaped_query = torch.cat([(pad_query_states[:, :, (stride - 1 - q) :: stride, :]) for q in range(stride)], dim=-1)

attn_weights_slice = torch.matmul(chunked_query, reshaped_key.transpose(2, 3))
attn_weights_slice = attn_weights_slice / math.sqrt(head_dim) / stride / norm
attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32)

attn_sum = attn_weights_slice.view(..., num_blocks, block_size, ...).sum(dim=-1).sum(dim=-2)
simple_mask = find_blocks_chunked(attn_sum, current_index, threshold=0.9, ...)
```

### A.2 FlexPrefill get_active_blocks
```python
# From xattn/src/Flexprefill.py
last_q = q[:, -block_size:, :, :] / math.sqrt(head_dim)
qk = torch.einsum("bihgd, bjhgd -> bhgij", last_q, k)
qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)

slash = sum_all_diagonal_matrix(qk) / qk.shape[-2]
vertical = qk.mean(-2)

kl_div = square_root_js_divergence(avg_qk, vertical)
block_sparse_mask = kl_div < tau  # Heads that need block sparse

num_vertical_blocks = score_cover_topk(vertical, gamma)
num_slash_blocks = score_cover_topk(slash, gamma)
```

### A.3 MInference sum_all_diagonal_matrix
```python
# From xattn/src/Minference.py
def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device)
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided(
        (1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)
    )
    sum_diags = torch.sum(mat_strided, 2)
    return sum_diags[:, :, 1:]
```
