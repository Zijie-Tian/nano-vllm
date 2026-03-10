"""
XAttention Block Sparse Attention (BSA) Policy for nano-vllm.

This module implements XAttention-inspired block sparse attention for chunked prefill.

Key design:
1. Use xattn_estimate_chunked to estimate sparse block mask
2. Use BSA kernel for efficient sparse attention computation
3. Support chunked prefill with q_start_pos for correct position handling

Note: Decode phase is not supported - use FullAttentionPolicy for decode.
"""

import logging
import torch
import torch.cuda.nvtx as nvtx
from typing import List, Tuple, TYPE_CHECKING

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.utils.density_observer import DensityObserver

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)

# Global storage for mask debugging
_DEBUG_SAVE_MASK = False  # Set to True to save masks for comparison
_DEBUG_MASK_STORAGE = {}

# Check BSA availability
try:
    from block_sparse_attn import block_sparse_attn_func

    BSA_AVAILABLE = True
except ImportError:
    BSA_AVAILABLE = False
    logger.warning(
        "block_sparse_attn not available, XAttentionBSAPolicy will fallback to dense"
    )

# Check xattn_estimate_chunked availability
try:
    from nanovllm.ops.xattn import xattn_estimate_chunked  # noqa: F401

    XATTN_AVAILABLE = True
except ImportError:
    XATTN_AVAILABLE = False
    logger.warning("xattn_estimate_chunked not available")


def expand_kv_for_gqa(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expand KV for Grouped Query Attention.

    Args:
        key_states: [B, num_kv_heads, seq_len, head_dim]
        value_states: [B, num_kv_heads, seq_len, head_dim]
        num_heads: Number of query heads

    Returns:
        Expanded (key, value) with shape [B, num_heads, seq_len, head_dim]
    """
    num_kv_heads = key_states.shape[1]
    if num_heads == num_kv_heads:
        return key_states, value_states
    num_groups = num_heads // num_kv_heads
    return (
        key_states.repeat_interleave(num_groups, dim=1),
        value_states.repeat_interleave(num_groups, dim=1),
    )


class XAttentionBSAPolicy(SparsePolicy):
    """
    XAttention Block Sparse Attention policy for chunked prefill.

    Uses xattn_estimate_chunked to estimate sparse mask, then BSA kernel
    for efficient sparse attention computation.

    Note:
        - Only supports prefill phase (decode uses FullAttentionPolicy)
        - BSA block size is fixed at 128 tokens
    """

    supports_prefill = True
    supports_decode = False  # Decode uses FullAttentionPolicy
    requires_block_selection = False  # Selection happens internally

    # BSA requires 128-token blocks
    BSA_BLOCK_SIZE = 128

    def __init__(
        self,
        threshold: float = 0.95,  # High threshold for accuracy testing
        stride: int = 8,
        chunk_size: int = 4096,  # Match offload Q chunk size for density alignment
        block_size: int = 128,
        samples_per_chunk: int = 128,
        use_triton: bool = True,
        estimate_block_size: int = 1024,  # Optimized block size for softmax_fuse_block_sum
    ):
        """
        Initialize XAttention BSA policy.

        Args:
            threshold: Cumulative attention threshold for block selection (0-1)
                       Higher values = more blocks selected = less sparse
            stride: Stride for Q/K reshape in estimation (typically 8)
            chunk_size: Processing chunk size for xattn_estimate (Triton alignment)
            block_size: BSA block size (must be 128)
            samples_per_chunk: Samples per chunk for estimation (unused)
            use_triton: Whether to use Triton kernels
            estimate_block_size: Block size for softmax_fuse_block_sum in select_blocks.
                                 Default 1024 is optimal (15x faster than 4096).
                                 Must be a factor of cpu_block_size (e.g., 4096/1024=4).
        """
        self.threshold = threshold
        self.stride = stride
        self.chunk_size = chunk_size
        self.use_triton = use_triton
        self.estimate_block_size = estimate_block_size
        self._num_heads = None  # Set during first forward

        # Sparse metadata: stores attention scores per layer
        # Dict[layer_id, Tensor[num_q_blocks, num_k_blocks]]
        self.sparse_metadata: dict = {}

        # Statistics for density tracking
        self._stats_total_available_blocks = 0
        self._stats_total_selected_blocks = 0
        self._stats_num_chunks = 0

        # Pre-allocated GQA expansion buffers (GPU-only mode)
        # Set by alloc_policy_metadata(), None if not pre-allocated
        self._k_expanded: torch.Tensor | None = None
        self._v_expanded: torch.Tensor | None = None
        self._max_seq_len: int = 0

        # Pre-allocated mask buffer for chunked prefill (offload mode)
        # Stores BSA-level mask from select_blocks for use in compute_chunked_prefill
        # Shape: [1, num_heads, max_q_bsa_blocks, max_k_bsa_blocks]
        self._prefill_mask_buffer: torch.Tensor | None = None
        self._current_mask_q_bsa: int = 0  # Current Q BSA blocks in buffer
        self._current_mask_k_bsa: int = 0  # Current K BSA blocks in buffer

        # Selected block indices for mask extraction in compute_chunked_prefill
        # Stores the indices of selected CPU blocks in available_blocks
        self._selected_cpu_indices: List[int] = []
        self._bsa_per_cpu: int = 0  # BSA blocks per CPU block

        # =====================================================================
        # Pre-allocated buffers for 3-stage KV chunking (offload mode)
        # =====================================================================
        # Partial softmax stats: m (max) and l (exp sum) for each KV chunk
        # Shape: [max_kv_chunks, batch, heads, q_reshaped_len]
        self._m_partial_buffer: torch.Tensor | None = None
        self._l_partial_buffer: torch.Tensor | None = None

        # Block sums buffer: normalized attention sums for all K blocks
        # Shape: [batch, heads, max_q_bsa_blocks, max_k_bsa_blocks]
        self._block_sums_buffer: torch.Tensor | None = None

        # Configuration for KV chunking
        self._max_kv_chunks: int = 0
        self._cpu_block_size: int = 0  # Tokens per CPU block (set at runtime)

    def alloc_policy_metadata(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        enable_cpu_offload: bool = False,
    ) -> None:
        """
        Pre-allocate GQA expansion buffers for GPU-only mode.

        These buffers are used by compute_prefill() to avoid dynamic allocation
        during forward pass. The buffers are sized for max_seq_len and sliced
        to actual seq_len during use.

        Memory usage: 2 * num_heads * max_seq_len * head_dim * dtype_size
        For 64K seq, 32 heads, 128 dim, fp16: 2 * 32 * 65536 * 128 * 2 = 1 GB

        Args:
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            dtype: Data type
            device: Target device
        """
        # Pre-allocate mask buffer for chunked prefill (offload mode)
        # mask shape: [1, num_heads, max_q_bsa_blocks, max_k_bsa_blocks]
        # This is needed regardless of GQA
        max_q_bsa_blocks = self.chunk_size // self.BSA_BLOCK_SIZE
        max_k_bsa_blocks = max_seq_len // self.BSA_BLOCK_SIZE
        mask_shape = (1, num_heads, max_q_bsa_blocks, max_k_bsa_blocks)
        self._prefill_mask_buffer = torch.empty(
            mask_shape, dtype=torch.bool, device=device
        )
        mask_memory_mb = num_heads * max_q_bsa_blocks * max_k_bsa_blocks / (1024 * 1024)
        logger.info(
            f"[XAttn] Pre-allocated mask buffer: shape={mask_shape}, memory={mask_memory_mb:.1f} MB"
        )

        # =====================================================================
        # Pre-allocate buffers for 3-stage KV chunking (offload mode)
        # =====================================================================
        # Calculate max KV chunks: historical blocks + current chunk
        # Use cpu_block_size as KV chunk granularity (will be set at runtime)
        # For now, estimate based on chunk_size (actual cpu_block_size may differ)
        estimated_cpu_block_size = 4096  # Default, will be overwritten
        max_kv_chunks = (
            max_seq_len // estimated_cpu_block_size
        ) + 1  # +1 for current chunk

        # Q reshaped length for one chunk
        q_reshaped_len = self.chunk_size // self.stride
        estimated_cpu_block_size // self.stride

        # Partial stats buffers: [max_kv_chunks, batch=1, heads, q_reshaped_len]
        m_partial_shape = (max_kv_chunks, 1, num_heads, q_reshaped_len)
        self._m_partial_buffer = torch.empty(
            m_partial_shape, dtype=torch.float32, device=device
        )
        self._l_partial_buffer = torch.empty(
            m_partial_shape, dtype=torch.float32, device=device
        )

        # Block sums buffer: [batch=1, heads, max_q_bsa_blocks, max_k_bsa_blocks]
        block_sums_shape = (1, num_heads, max_q_bsa_blocks, max_k_bsa_blocks)
        self._block_sums_buffer = torch.empty(
            block_sums_shape, dtype=dtype, device=device
        )

        self._max_kv_chunks = max_kv_chunks

        # Memory calculation
        m_l_memory_mb = (
            2 * max_kv_chunks * num_heads * q_reshaped_len * 4 / (1024 * 1024)
        )
        block_sums_memory_mb = (
            num_heads
            * max_q_bsa_blocks
            * max_k_bsa_blocks
            * dtype.itemsize
            / (1024 * 1024)
        )
        logger.info(
            f"[XAttn] Pre-allocated KV chunking buffers: "
            f"m/l shape={m_partial_shape} ({m_l_memory_mb:.1f} MB), "
            f"block_sums shape={block_sums_shape} ({block_sums_memory_mb:.1f} MB)"
        )

        # Skip GQA buffers in offload mode
        # Chunked prefill uses compute_chunked_prefill() which handles GQA inline
        if enable_cpu_offload:
            logger.info(
                "[XAttn] Offload mode: skipping GQA expansion buffers (saves ~16GB for 1M seq)"
            )
            return

        # GPU-only mode: pre-allocate GQA buffers for compute_prefill()
        # Only allocate if GQA (num_heads != num_kv_heads)
        if num_heads == num_kv_heads:
            logger.info(
                f"[XAttn] No GQA expansion needed (num_heads == num_kv_heads = {num_heads})"
            )
            return

        # Shape: [1, num_heads, max_seq_len, head_dim] for xattn_estimate format
        # Also used for BSA which expects [seq_len, num_heads, head_dim]
        shape = (1, num_heads, max_seq_len, head_dim)
        self._k_expanded = torch.empty(shape, dtype=dtype, device=device)
        self._v_expanded = torch.empty(shape, dtype=dtype, device=device)
        self._max_seq_len = max_seq_len

        memory_mb = (
            2 * num_heads * max_seq_len * head_dim * dtype.itemsize / (1024 * 1024)
        )
        logger.info(
            f"[XAttn] Pre-allocated GQA buffers: shape={shape}, memory={memory_mb:.1f} MB"
        )

    # =========================================================================
    # GPU-only methods (non-chunked)
    # =========================================================================

    def compute_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        layer_id: int,
        block_tables: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        GPU-only prefill attention using XAttention + BSA.

        This method implements sparse attention for GPU-only mode:
        1. Estimate block importance using xattn_estimate
        2. Compute sparse attention using block_sparse_attn_func

        Args:
            q: Query tensor [total_q, num_heads, head_dim] (varlen packed)
            k: Key tensor [total_kv, num_kv_heads, head_dim] (varlen packed)
            v: Value tensor [total_kv, num_kv_heads, head_dim] (varlen packed)
            cu_seqlens_q: Cumulative sequence lengths for Q [batch+1]
            cu_seqlens_k: Cumulative sequence lengths for K [batch+1]
            max_seqlen_q: Maximum Q sequence length
            max_seqlen_k: Maximum K sequence length
            softmax_scale: Softmax scaling factor
            layer_id: Transformer layer index
            block_tables: Paged attention block tables (not used for XAttention)

        Returns:
            Attention output [total_q, num_heads, head_dim]
        """
        # Fallback to flash attention when:
        # 1. block_tables provided (paged KV cache / prefix cache) - XAttention expects contiguous K, V
        # 2. BSA kernel not available
        # 3. xattn_estimate not available
        if block_tables is not None or not BSA_AVAILABLE or not XATTN_AVAILABLE:
            from flash_attn import flash_attn_varlen_func

            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                block_table=block_tables,
            )

        from nanovllm.ops.xattn import xattn_estimate

        # Set DensityObserver mode on first layer
        if layer_id == 0:
            DensityObserver.set_mode("gpu_only")

        # Get dimensions
        total_q, num_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape

        # For now, assume batch_size = 1 (single sequence)
        # TODO: Support batched varlen format
        batch_size = cu_seqlens_q.shape[0] - 1
        if batch_size != 1:
            # Fallback to flash attention for batched input
            from flash_attn import flash_attn_varlen_func

            logger.warning(
                f"[XAttn] batch_size={batch_size} > 1, falling back to flash attention"
            )
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
            )

        q_len = max_seqlen_q
        k_len = max_seqlen_k

        # Convert from varlen format [total, heads, dim] to [batch, heads, seq, dim]
        # q: [q_len, num_heads, head_dim] -> [1, num_heads, q_len, head_dim]
        Q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, q_len, head_dim]
        K = k.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, k_len, head_dim]
        V = v.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, k_len, head_dim]

        # Expand KV for GQA - use pre-allocated buffers if available
        if num_heads != num_kv_heads:
            num_groups = num_heads // num_kv_heads
            if self._k_expanded is not None and k_len <= self._max_seq_len:
                # Use pre-allocated buffers with in-place expansion
                K_exp = self._k_expanded[:, :, :k_len, :]
                V_exp = self._v_expanded[:, :, :k_len, :]
                # In-place GQA expansion: [1, num_kv_heads, k_len, head_dim] -> [1, num_heads, k_len, head_dim]
                # Reshape K to [1, num_kv_heads, 1, k_len, head_dim] and broadcast to [1, num_kv_heads, num_groups, k_len, head_dim]
                K_exp.view(1, num_kv_heads, num_groups, k_len, head_dim).copy_(
                    K.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
                )
                V_exp.view(1, num_kv_heads, num_groups, k_len, head_dim).copy_(
                    V.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
                )
            else:
                # Fallback: dynamic allocation (when buffers not pre-allocated or seq too long)
                K_exp, V_exp = expand_kv_for_gqa(K, V, num_heads)
        else:
            K_exp, V_exp = K, V

        # Estimate block importance and get sparse mask
        with nvtx.range("xattn_estimate"):
            attn_sums, mask = xattn_estimate(
                Q,
                K_exp,
                chunk_size=self.chunk_size,
                block_size=self.BSA_BLOCK_SIZE,
                stride=self.stride,
                threshold=self.threshold,
                use_triton=self.use_triton,
                causal=True,
            )

        # Debug: Save Q, K, mask, attn_sums for external verification
        if _DEBUG_SAVE_MASK and layer_id == 0:
            import os

            valid_q_blocks = (q_len + self.BSA_BLOCK_SIZE - 1) // self.BSA_BLOCK_SIZE
            valid_k_blocks = (k_len + self.BSA_BLOCK_SIZE - 1) // self.BSA_BLOCK_SIZE
            mask_valid = mask[:, :, :valid_q_blocks, :valid_k_blocks]
            attn_sums_valid = attn_sums[:, :, :valid_q_blocks, :valid_k_blocks]
            save_dir = "/home/zijie/Code/nano-vllm/results/mask_alignment"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/gpuonly_layer{layer_id}.pt"
            torch.save(
                {
                    # Input tensors (GQA-expanded)
                    "Q": Q.clone().cpu(),  # [1, num_heads, q_len, head_dim]
                    "K": K_exp.clone().cpu(),  # [1, num_heads, k_len, head_dim]
                    # xattn_estimate parameters
                    "chunk_size": self.chunk_size,
                    "block_size": self.BSA_BLOCK_SIZE,
                    "stride": self.stride,
                    "threshold": self.threshold,
                    # Output for comparison
                    "mask": mask_valid.clone().cpu(),
                    "attn_sums": attn_sums_valid.clone().cpu(),
                    # Metadata
                    "q_len": q_len,
                    "k_len": k_len,
                    "valid_q_blocks": valid_q_blocks,
                    "valid_k_blocks": valid_k_blocks,
                },
                save_path,
            )
            logger.info(
                f"[DEBUG] Saved Q/K/mask to {save_path}, Q={Q.shape}, K={K_exp.shape}"
            )

        # Compute block counts
        q_block_num = (q_len + self.BSA_BLOCK_SIZE - 1) // self.BSA_BLOCK_SIZE
        k_block_num = (k_len + self.BSA_BLOCK_SIZE - 1) // self.BSA_BLOCK_SIZE

        # Prepare tensors for BSA
        # q, k, v need to be [seq_len, num_heads, head_dim]
        q_bsa = q  # Already [q_len, num_heads, head_dim]

        # For GQA with BSA, reuse the expanded K_exp, V_exp (convert to BSA format)
        # K_exp: [1, num_heads, k_len, head_dim] -> [k_len, num_heads, head_dim]
        if num_heads != num_kv_heads:
            k_bsa = K_exp.squeeze(0).transpose(0, 1)  # [k_len, num_heads, head_dim]
            v_bsa = V_exp.squeeze(0).transpose(0, 1)  # [k_len, num_heads, head_dim]
        else:
            k_bsa = k
            v_bsa = v

        # Prepare BSA inputs
        cu_seqlens_q_bsa = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
        cu_seqlens_k_bsa = torch.tensor([0, k_len], dtype=torch.int32, device=k.device)
        head_groups = torch.ones(num_heads, dtype=torch.int32, device=q.device)

        # Trim mask to actual block counts
        mask_trimmed = mask[:, :, :q_block_num, :k_block_num].contiguous()

        # Compute sparse attention using BSA
        with nvtx.range("xattn_bsa_compute"):
            output = block_sparse_attn_func(
                q_bsa,
                k_bsa,
                v_bsa,
                cu_seqlens_q_bsa,
                cu_seqlens_k_bsa,
                head_groups,
                None,  # key_padding_mask
                mask_trimmed,
                q_len,
                k_len,
                p_dropout=0.0,
                deterministic=True,
                is_causal=True,
            )

        # Record density for all layers via DensityObserver
        # Note: DensityObserver.record only computes density when enabled,
        # otherwise it returns immediately without any GPU-CPU sync
        DensityObserver.record(layer_id, mask_trimmed, causal=True)

        return output

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        GPU-only decode attention - delegates to FullAttentionPolicy.

        XAttention is designed for long prefill sequences. For decode (single token),
        we use FullAttentionPolicy which calls flash_attn_with_kvcache.
        """
        from nanovllm.kvcache.sparse.full_policy import FullAttentionPolicy

        return FullAttentionPolicy().compute_decode(
            q, k_cache, v_cache, cache_seqlens, softmax_scale, layer_id, block_tables
        )

    # =========================================================================
    # Chunked offload methods
    # =========================================================================

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select important blocks using 3-stage KV chunking algorithm.

        This method implements the same algorithm as tests/test_xattn_estimate_alignment.py:
        1. For each KV chunk: compute attention scores and partial softmax stats
        2. Merge all partial stats to get global m and l
        3. For each KV chunk: normalize with global stats and compute block sums
        4. Use find_blocks_chunked to select important blocks

        This approach:
        - Uses O(S×C) peak memory instead of O(S²)
        - Produces identical density to GPU-only xattn_estimate
        - Supports ultra-long contexts

        Args:
            available_blocks: List of CPU block IDs (historical blocks only)
            offload_engine: OffloadEngine for loading blocks
            ctx: PolicyContext with metadata
            q: Query tensor [seq_len, num_heads, head_dim] for current chunk
            k: Key tensor [seq_len, num_kv_heads, head_dim] for current chunk

        Returns:
            Selected block IDs based on attention threshold
        """
        if q is None:
            return available_blocks

        # CRITICAL: Wait for all previous prefill offloads to complete before loading from CPU
        # This ensures that the K data we load from k_cache_cpu is actually valid.
        # Without this sync, we may load stale/uninitialized data because the async offload
        # from the previous chunk hasn't finished yet.
        if available_blocks and offload_engine is not None:
            offload_engine.wait_all_prefill_offloads()

        from nanovllm.ops.xattn import (
            flat_group_gemm_fuse_reshape,
            softmax_compute_partial_stats,
            softmax_normalize_and_block_sum,
            merge_softmax_stats,
            find_blocks_chunked,
        )
        import math

        layer_id = ctx.layer_id

        # Set DensityObserver mode on first layer
        if layer_id == 0:
            DensityObserver.set_mode("offload")

        # ================================================================
        # Step 0: Setup parameters
        # ================================================================
        # Convert Q to [batch, heads, seq_len, head_dim]
        Q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, q_len, head_dim]

        num_heads = Q.shape[1]
        head_dim = Q.shape[3]
        q_len = Q.shape[2]

        # Alignment requirements
        BLOCK_M = 128  # Triton block size
        alignment = self.stride * BLOCK_M  # 8 * 128 = 1024

        if q_len < alignment:
            # Q too short, skip estimation and return all blocks
            logger.debug(
                f"[XAttn] select_blocks: q_len={q_len} < alignment={alignment}, skipping estimation"
            )
            return available_blocks

        # Pad Q to alignment
        padded_q_len = ((q_len + alignment - 1) // alignment) * alignment
        q_pad_size = padded_q_len - q_len
        if q_pad_size > 0:
            Q = torch.nn.functional.pad(Q, (0, 0, 0, q_pad_size), value=0)

        # Get CPU block size from context
        cpu_block_size = ctx.block_size  # e.g., 4096 tokens per CPU block
        self._cpu_block_size = cpu_block_size

        # KV chunk parameters (use CPU block as KV chunk unit)
        num_historical_blocks = len(available_blocks)
        historical_k_len = num_historical_blocks * cpu_block_size
        total_k_len = historical_k_len + q_len  # Include current chunk

        # Reshaped dimensions
        reshaped_block_size = self.BSA_BLOCK_SIZE // self.stride  # 128/8 = 16
        q_reshaped_len = padded_q_len // self.stride
        kv_chunk_reshaped = cpu_block_size // self.stride

        # BSA blocks per CPU block
        bsa_per_cpu = cpu_block_size // self.BSA_BLOCK_SIZE  # 4096/128 = 32

        # Global K position parameters
        # Q在全局K序列中的位置 (按照 test_xattn_estimate_alignment.py 的逻辑)
        # 对于 chunked softmax，我们需要计算 Q 在整个序列中的 BSA block 偏移
        # k_block_num = total BSA blocks (padded), q_block_num = Q's BSA blocks (padded)
        padded_total_k_len = ((total_k_len + alignment - 1) // alignment) * alignment
        k_block_num = padded_total_k_len // self.BSA_BLOCK_SIZE
        q_block_num = padded_q_len // self.BSA_BLOCK_SIZE
        chunk_start = (
            k_block_num - q_block_num
        ) * reshaped_block_size  # Q 在 reshaped 空间的起始
        chunk_end = chunk_start + q_reshaped_len

        # real_q_len: 用于 softmax 归一化的有效 Q 长度
        k_reshaped_seq_len = padded_total_k_len // self.stride
        k_reshaped_num_to_pad = (padded_total_k_len - total_k_len) // self.stride

        # Softmax scale
        norm = 1.0
        scale = 1.4426950408889634 / math.sqrt(head_dim) / self.stride / norm
        segment_size = min(4096, reshaped_block_size)

        # ================================================================
        # Step 1: First pass - compute partial stats for all KV chunks
        # ================================================================
        m_chunks = []
        l_chunks = []
        num_historical_blocks + 1  # +1 for current chunk

        # Get compute_stream for all compute kernels (like attention computation)
        compute_stream = offload_engine.compute_stream

        # Ring buffer pipeline config (shared by pass1 and pass2)
        load_slots = list(range(offload_engine.num_ring_slots))
        num_slots = len(load_slots)
        num_blocks_hist = len(available_blocks)

        with nvtx.range("xattn_estimate_pass1"):
            # Process historical blocks (from CPU) with ring buffer pipeline
            if num_blocks_hist > 0:
                # Preload first N blocks into ring buffer slots
                num_preload = min(num_slots, num_blocks_hist)
                for i in range(num_preload):
                    offload_engine.load_k_only_to_slot_layer(
                        load_slots[i],
                        layer_id,
                        available_blocks[i],
                        chunk_idx=available_blocks[i],
                    )

                for kv_chunk_idx in range(num_blocks_hist):
                    current_slot = load_slots[kv_chunk_idx % num_slots]

                    # wait_slot_layer makes compute_stream wait for H2D transfer
                    offload_engine.wait_slot_layer(current_slot)

                    # All compute kernels run on compute_stream (like attention computation)
                    with torch.cuda.stream(compute_stream):
                        k_block = offload_engine.get_k_for_slot(
                            current_slot
                        )  # [1, block_size, num_kv_heads, head_dim]
                        K_chunk = k_block.transpose(
                            1, 2
                        )  # [1, num_kv_heads, block_size, head_dim]

                        # GQA expansion
                        num_kv_heads = K_chunk.shape[1]
                        if num_heads != num_kv_heads:
                            num_groups = num_heads // num_kv_heads
                            K_chunk = K_chunk.repeat_interleave(num_groups, dim=1)

                        # KV offset in reshaped space
                        kv_offset_reshaped = kv_chunk_idx * kv_chunk_reshaped

                        # Compute raw attention scores
                        attn_weights_kv = flat_group_gemm_fuse_reshape(
                            Q,
                            K_chunk,
                            self.stride,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            is_causal=False,  # K 不完整，不能在这里用 causal
                        )

                        # Compute partial stats (带 causal mask)
                        m_partial, l_partial = softmax_compute_partial_stats(
                            attn_weights_kv,
                            reshaped_block_size,
                            segment_size,
                            scale,
                            chunk_start=chunk_start,
                            kv_offset=kv_offset_reshaped,
                            is_causal=True,
                        )
                        m_chunks.append(m_partial)
                        l_chunks.append(l_partial)

                        offload_engine.record_slot_compute_done(current_slot)
                        del attn_weights_kv

                    # Issue next H2D transfer (overlap with compute on different slot)
                    next_idx = kv_chunk_idx + num_slots
                    if next_idx < num_blocks_hist:
                        next_slot = load_slots[next_idx % num_slots]
                        offload_engine.load_k_only_to_slot_layer(
                            next_slot,
                            layer_id,
                            available_blocks[next_idx],
                            chunk_idx=available_blocks[next_idx],
                        )

            # Process current chunk K (already on GPU) on compute_stream
            with torch.cuda.stream(compute_stream):
                # k: [seq_len, num_kv_heads, head_dim] -> [1, num_kv_heads, seq_len, head_dim]
                K_current = k.unsqueeze(0).transpose(1, 2)

                # GQA expansion for current chunk
                num_kv_heads = K_current.shape[1]
                if num_heads != num_kv_heads:
                    num_groups = num_heads // num_kv_heads
                    K_current = K_current.repeat_interleave(num_groups, dim=1)

                # Pad current K to alignment
                curr_k_len = K_current.shape[2]
                padded_curr_k_len = (
                    (curr_k_len + alignment - 1) // alignment
                ) * alignment
                if padded_curr_k_len != curr_k_len:
                    K_current = torch.nn.functional.pad(
                        K_current, (0, 0, 0, padded_curr_k_len - curr_k_len), value=0
                    )

                # KV offset for current chunk
                kv_offset_current = num_historical_blocks * kv_chunk_reshaped

                # Compute attention scores for current chunk
                attn_weights_curr = flat_group_gemm_fuse_reshape(
                    Q,
                    K_current,
                    self.stride,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    is_causal=False,
                )

                # Compute partial stats for current chunk
                m_partial_curr, l_partial_curr = softmax_compute_partial_stats(
                    attn_weights_curr,
                    reshaped_block_size,
                    segment_size,
                    scale,
                    chunk_start=chunk_start,
                    kv_offset=kv_offset_current,
                    is_causal=True,
                )
                m_chunks.append(m_partial_curr)
                l_chunks.append(l_partial_curr)

                del attn_weights_curr

        # ================================================================
        # Step 2: Merge all partial stats (on compute_stream)
        # ================================================================
        with torch.cuda.stream(compute_stream):
            with nvtx.range("xattn_estimate_merge"):
                m_global, l_global = merge_softmax_stats(m_chunks, l_chunks)

            del m_chunks, l_chunks

        # ================================================================
        # Step 3: Second pass - normalize and compute block sums
        # ================================================================
        attn_sum_per_kv = []

        with nvtx.range("xattn_estimate_pass2"):
            # Process historical blocks again with ring buffer pipeline
            if num_blocks_hist > 0:
                # Preload first N blocks into ring buffer slots
                num_preload = min(num_slots, num_blocks_hist)
                for i in range(num_preload):
                    offload_engine.load_k_only_to_slot_layer(
                        load_slots[i],
                        layer_id,
                        available_blocks[i],
                        chunk_idx=available_blocks[i],
                    )

                for kv_chunk_idx in range(num_blocks_hist):
                    current_slot = load_slots[kv_chunk_idx % num_slots]

                    # wait_slot_layer makes compute_stream wait for H2D transfer
                    offload_engine.wait_slot_layer(current_slot)

                    # All compute kernels run on compute_stream
                    with torch.cuda.stream(compute_stream):
                        k_block = offload_engine.get_k_for_slot(current_slot)
                        K_chunk = k_block.transpose(1, 2)

                        num_kv_heads = K_chunk.shape[1]
                        if num_heads != num_kv_heads:
                            num_groups = num_heads // num_kv_heads
                            K_chunk = K_chunk.repeat_interleave(num_groups, dim=1)

                        kv_offset_reshaped = kv_chunk_idx * kv_chunk_reshaped

                        # Recompute attention scores (trade-off: compute vs memory)
                        attn_weights_kv = flat_group_gemm_fuse_reshape(
                            Q,
                            K_chunk,
                            self.stride,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            is_causal=False,
                        )

                        # Normalize with global stats and compute block sums
                        block_sum_kv = softmax_normalize_and_block_sum(
                            attn_weights_kv,
                            m_global,
                            l_global,
                            reshaped_block_size,
                            segment_size,
                            chunk_start=chunk_start,
                            real_q_len=k_reshaped_seq_len - k_reshaped_num_to_pad,
                            scale=scale,
                            kv_offset=kv_offset_reshaped,
                            is_causal=True,
                        )
                        attn_sum_per_kv.append(block_sum_kv)

                        offload_engine.record_slot_compute_done(current_slot)
                        del attn_weights_kv

                    # Issue next H2D transfer (overlap with compute on different slot)
                    next_idx = kv_chunk_idx + num_slots
                    if next_idx < num_blocks_hist:
                        next_slot = load_slots[next_idx % num_slots]
                        offload_engine.load_k_only_to_slot_layer(
                            next_slot,
                            layer_id,
                            available_blocks[next_idx],
                            chunk_idx=available_blocks[next_idx],
                        )

            # Process current chunk on compute_stream
            with torch.cuda.stream(compute_stream):
                # Recompute attention scores for current chunk
                attn_weights_curr = flat_group_gemm_fuse_reshape(
                    Q,
                    K_current,
                    self.stride,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    is_causal=False,
                )

                block_sum_curr = softmax_normalize_and_block_sum(
                    attn_weights_curr,
                    m_global,
                    l_global,
                    reshaped_block_size,
                    segment_size,
                    chunk_start=chunk_start,
                    real_q_len=k_reshaped_seq_len - k_reshaped_num_to_pad,
                    scale=scale,
                    kv_offset=kv_offset_current,
                    is_causal=True,
                )
                attn_sum_per_kv.append(block_sum_curr)
                del attn_weights_curr, K_current

        # ================================================================
        # Step 4: Concatenate block sums and select blocks (on compute_stream)
        # ================================================================
        with torch.cuda.stream(compute_stream):
            attn_sum_concat = torch.cat(attn_sum_per_kv, dim=-1)
            del attn_sum_per_kv, m_global, l_global

            # Calculate q_block offset for find_blocks_chunked
            # This is the number of BSA blocks before Q in the full sequence
            q_reshaped_len // reshaped_block_size
            current_index = (
                k_block_num - q_block_num
            )  # Q starts at this BSA block index

            with nvtx.range("xattn_find_blocks"):
                mask = find_blocks_chunked(
                    attn_sum_concat,
                    current_index=current_index,
                    threshold=self.threshold,
                    num_to_choose=None,
                    decoding=False,
                    mode="prefill",
                    causal=True,
                )

            # Apply causal mask post-processing (same as xattn.py lines 1300-1306)
            mask[:, :, -q_block_num:, -q_block_num:] = torch.where(
                torch.tril(
                    torch.ones(
                        q_block_num, q_block_num, dtype=torch.bool, device=mask.device
                    ),
                    diagonal=0,
                ),
                mask[:, :, -q_block_num:, -q_block_num:],
                False,
            )

        # ================================================================
        # Step 5: Record density (all layers for alignment with GPU-only)
        # ================================================================
        # Trim mask to valid region
        valid_q_blocks = (q_len + self.BSA_BLOCK_SIZE - 1) // self.BSA_BLOCK_SIZE
        valid_k_blocks = (total_k_len + self.BSA_BLOCK_SIZE - 1) // self.BSA_BLOCK_SIZE
        mask_valid = mask[:, :, :valid_q_blocks, :valid_k_blocks]
        attn_sums_valid = attn_sum_concat[:, :, :valid_q_blocks, :valid_k_blocks]

        # Compute causal mask for density calculation
        q_offset_blocks = valid_k_blocks - valid_q_blocks
        indices = torch.arange(valid_k_blocks, device=mask.device).unsqueeze(0)
        q_indices = torch.arange(valid_q_blocks, device=mask.device).unsqueeze(1)
        causal_mask = indices <= (q_indices + q_offset_blocks)

        chunk_total = (
            causal_mask.sum().item() * mask_valid.shape[0] * mask_valid.shape[1]
        )
        chunk_selected = (
            (mask_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
        )

        DensityObserver.record_counts(layer_id, chunk_selected, chunk_total)
        # Only log for layer 0 to avoid spam (density is recorded for all layers)
        if layer_id == 0:
            logger.info(
                f"[XAttn Offload] Layer{layer_id} chunk: q_len={q_len}, k_len={total_k_len}, "
                f"valid_q_blocks={valid_q_blocks}, valid_k_blocks={valid_k_blocks}, "
                f"q_offset={q_offset_blocks}, selected={chunk_selected}, total={chunk_total}, "
                f"density={chunk_selected / chunk_total:.4f}"
            )

        # Debug: Save mask and attention sums for comparison
        if _DEBUG_SAVE_MASK:
            import os

            chunk_idx = ctx.query_chunk_idx if ctx else 0
            save_dir = "/home/zijie/Code/nano-vllm/results/mask_alignment"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/offload_layer{layer_id}_chunk{chunk_idx}.pt"
            torch.save(
                {
                    "mask": mask_valid.clone().cpu(),
                    "attn_sums": attn_sums_valid.clone().cpu(),
                    "q_len": q_len,
                    "k_len": total_k_len,
                    "valid_q_blocks": valid_q_blocks,
                    "valid_k_blocks": valid_k_blocks,
                    "current_index": current_index,
                    "chunk_start": chunk_start,
                },
                save_path,
            )
            logger.info(f"[DEBUG] Saved mask to {save_path}")

        del attn_sum_concat

        # ================================================================
        # Step 6: Extract historical mask and aggregate to CPU blocks
        # ================================================================
        B, H, Q_bsa, K_bsa_total = mask.shape
        historical_k_bsa = num_historical_blocks * bsa_per_cpu

        # Save mask to buffer for compute_chunked_prefill (if needed later)
        if self._prefill_mask_buffer is not None and historical_k_bsa > 0:
            self._prefill_mask_buffer[:, :, :Q_bsa, :historical_k_bsa].copy_(
                mask[:, :, :, :historical_k_bsa]
            )
            self._current_mask_q_bsa = Q_bsa
            self._current_mask_k_bsa = historical_k_bsa

        # Aggregate to CPU block level (union across heads, Q blocks, BSA blocks per CPU)
        if num_historical_blocks == 0:
            return []

        mask_historical = mask[:, :, :, :historical_k_bsa]
        mask_per_cpu = mask_historical.view(
            B, H, Q_bsa, num_historical_blocks, bsa_per_cpu
        )
        cpu_needed = mask_per_cpu.any(dim=-1).any(dim=2).any(dim=1)  # [B, num_cpu]

        selected_indices = cpu_needed[0].nonzero().squeeze(-1).tolist()
        if isinstance(selected_indices, int):
            selected_indices = [selected_indices]

        selected_block_ids = [available_blocks[i] for i in selected_indices]

        # Always include first block (sink) and last block for safety
        if available_blocks and available_blocks[0] not in selected_block_ids:
            selected_block_ids.insert(0, available_blocks[0])
        if available_blocks and available_blocks[-1] not in selected_block_ids:
            selected_block_ids.append(available_blocks[-1])

        # Update statistics (only for layer 0 to avoid overcounting)
        if layer_id == 0 and available_blocks:
            self._stats_total_available_blocks += len(available_blocks)
            self._stats_total_selected_blocks += len(selected_block_ids)
            self._stats_num_chunks += 1

            # Record communication density to DensityObserver
            # Comm density = selected_cpu_blocks / available_cpu_blocks
            # This is different from compute density (BSA block granularity)
            DensityObserver.record_comm_density(
                layer_id=layer_id,
                selected_cpu_blocks=len(selected_block_ids),
                total_cpu_blocks=len(available_blocks),
            )

            # Log per-chunk density
            chunk_density = len(selected_block_ids) / len(available_blocks)
            logger.debug(
                f"[XAttn] chunk={ctx.query_chunk_idx}, available={len(available_blocks)}, "
                f"selected={len(selected_block_ids)}, chunk_density={chunk_density:.1%}"
            )

        return selected_block_ids

    def compute_chunked_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        current_chunk_idx: int,
        seq: "Sequence",
        num_tokens: int,
        selected_blocks: List[int],
    ) -> torch.Tensor:
        """
        Compute attention for chunked prefill using XAttention sparse block selection.

        This method handles the chunked prefill computation:
        1. Load and compute attention to historical chunks (using selected_blocks)
        2. Compute attention to current chunk
        3. Merge all results

        Note: The BSA-level mask is saved in self._prefill_mask_buffer by select_blocks().
        Currently we use flash_attn_with_lse for computation (supports LSE merge).
        TODO: Optimize to use BSA kernel with the saved mask for per-head sparse attention.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim] (unused, from prefill buffer)
            v: Value tensor [seq_len, num_kv_heads, head_dim] (unused, from prefill buffer)
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            current_chunk_idx: Current chunk index
            seq: Sequence object
            num_tokens: Number of tokens in current chunk
            selected_blocks: List of CPU block IDs selected by select_blocks

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        # Use FlashInfer-based implementations (more optimized)
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Use the pre-selected blocks directly
        cpu_block_table = selected_blocks

        # Note: BSA mask is available in self._prefill_mask_buffer (saved by select_blocks)
        # Mask shape: [1, num_heads, Q_bsa, K_bsa] where Q_bsa = self._current_mask_q_bsa
        # Selected indices: self._selected_cpu_indices, bsa_per_cpu: self._bsa_per_cpu
        # TODO: Use this mask with BSA kernel for per-head sparse attention optimization

        if cpu_block_table:
            with nvtx.range("xattn_compute_historical"):
                load_slots = list(range(offload_engine.num_ring_slots))
                num_blocks = len(cpu_block_table)

                if len(load_slots) == 1:
                    # Only 1 slot - use synchronous mode
                    slot = load_slots[0]
                    for block_idx in range(num_blocks):
                        cpu_block_id = cpu_block_table[block_idx]
                        offload_engine.load_to_slot_layer(
                            slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id
                        )
                        offload_engine.wait_slot_layer(slot)

                        with torch.cuda.stream(compute_stream):
                            prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
                            prev_o, prev_lse = flash_attn_with_lse(
                                q_batched,
                                prev_k,
                                prev_v,
                                softmax_scale=softmax_scale,
                                causal=False,
                            )
                            if o_acc is None:
                                o_acc, lse_acc = prev_o, prev_lse
                            else:
                                o_acc, lse_acc = merge_attention_outputs(
                                    o_acc, lse_acc, prev_o, prev_lse
                                )
                            offload_engine.record_slot_compute_done(slot)
                else:
                    # Multiple slots - use pipeline
                    num_slots = len(load_slots)
                    num_preload = min(num_slots, num_blocks)
                    for i in range(num_preload):
                        cpu_block_id = cpu_block_table[i]
                        offload_engine.load_to_slot_layer(
                            load_slots[i],
                            layer_id,
                            cpu_block_id,
                            chunk_idx=cpu_block_id,
                        )

                    for block_idx in range(num_blocks):
                        current_slot = load_slots[block_idx % num_slots]

                        offload_engine.wait_slot_layer(current_slot)

                        with torch.cuda.stream(compute_stream):
                            prev_k, prev_v = offload_engine.get_kv_for_slot(
                                current_slot
                            )
                            prev_o, prev_lse = flash_attn_with_lse(
                                q_batched,
                                prev_k,
                                prev_v,
                                softmax_scale=softmax_scale,
                                causal=False,
                            )
                            offload_engine.record_slot_compute_done(current_slot)

                            if o_acc is None:
                                o_acc, lse_acc = prev_o, prev_lse
                            else:
                                o_acc, lse_acc = merge_attention_outputs(
                                    o_acc, lse_acc, prev_o, prev_lse
                                )

                        # Issue next transfer
                        next_block_idx = block_idx + num_slots
                        if next_block_idx < num_blocks:
                            next_slot = load_slots[next_block_idx % num_slots]
                            next_cpu_block_id = cpu_block_table[next_block_idx]
                            offload_engine.load_to_slot_layer(
                                next_slot,
                                layer_id,
                                next_cpu_block_id,
                                chunk_idx=next_cpu_block_id,
                            )

        # Compute attention to current chunk (causal mask)
        with nvtx.range("xattn_compute_current"):
            with torch.cuda.stream(compute_stream):
                k_curr, v_curr = offload_engine.get_prefill_buffer_slice(
                    layer_id, num_tokens
                )
                current_o, current_lse = flash_attn_with_lse(
                    q_batched,
                    k_curr,
                    v_curr,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

        # Merge historical and current attention
        with nvtx.range("xattn_compute_merge"):
            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    final_o = current_o
                else:
                    final_o, _ = merge_attention_outputs(
                        o_acc, lse_acc, current_o, current_lse
                    )

        # Sync default stream with compute_stream before returning
        torch.cuda.default_stream().wait_stream(compute_stream)

        # Remove batch dimension: [1, seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim]
        return final_o.squeeze(0)

    def compute_chunked_decode(
        self,
        q: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        seq: "Sequence",
        selected_blocks: List[int],
    ) -> torch.Tensor:
        """
        XAttention does not support decode phase.
        """
        raise NotImplementedError(
            "XAttentionBSAPolicy does not support decode phase. "
            "Use FullAttentionPolicy for decode."
        )

    def reset(self) -> None:
        """Reset policy state and clear sparse metadata."""
        self.sparse_metadata.clear()
        # Don't reset statistics here - they accumulate across the entire prefill

    def reset_stats(self) -> None:
        """Reset density statistics."""
        self._stats_total_available_blocks = 0
        self._stats_total_selected_blocks = 0
        self._stats_num_chunks = 0

    def get_density_stats(self) -> dict:
        """Get density statistics."""
        if self._stats_total_available_blocks == 0:
            return {
                "total_available_blocks": 0,
                "total_selected_blocks": 0,
                "num_chunks": 0,
                "overall_density": 0.0,
            }
        return {
            "total_available_blocks": self._stats_total_available_blocks,
            "total_selected_blocks": self._stats_total_selected_blocks,
            "num_chunks": self._stats_num_chunks,
            "overall_density": self._stats_total_selected_blocks
            / self._stats_total_available_blocks,
        }

    def print_density_stats(self) -> None:
        """Print density statistics summary."""
        stats = self.get_density_stats()
        logger.info(
            f"[XAttn BSA] Density Stats: chunks={stats['num_chunks']}, "
            f"available={stats['total_available_blocks']}, "
            f"selected={stats['total_selected_blocks']}, "
            f"density={stats['overall_density']:.1%}"
        )

    def offload_prefill_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
        **kwargs,
    ) -> None:
        super().offload_prefill_chunk(
            offload_engine, layer_id, cpu_block_id, num_tokens, **kwargs
        )

    def offload_decode_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        **kwargs,
    ) -> None:
        super().offload_decode_chunk(offload_engine, layer_id, cpu_block_id, **kwargs)

    def __repr__(self) -> str:
        return f"XAttentionBSAPolicy(threshold={self.threshold}, stride={self.stride})"
