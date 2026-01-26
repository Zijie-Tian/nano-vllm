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
from typing import List, Tuple, TYPE_CHECKING

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)

# Check BSA availability
try:
    from block_sparse_attn import block_sparse_attn_func
    BSA_AVAILABLE = True
except ImportError:
    BSA_AVAILABLE = False
    logger.warning("block_sparse_attn not available, XAttentionBSAPolicy will fallback to dense")

# Check xattn_estimate_chunked availability
try:
    from nanovllm.ops.xattn import xattn_estimate_chunked
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
        chunk_size: int = 16384,
        block_size: int = 128,
        samples_per_chunk: int = 128,
        use_triton: bool = True,
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
        """
        self.threshold = threshold
        self.stride = stride
        self.chunk_size = chunk_size
        self.use_triton = use_triton
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

    def alloc_policy_metadata(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
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
        # Only allocate if GQA (num_heads != num_kv_heads)
        if num_heads == num_kv_heads:
            logger.info(f"[XAttn] No GQA expansion needed (num_heads == num_kv_heads = {num_heads})")
            return

        # Shape: [1, num_heads, max_seq_len, head_dim] for xattn_estimate format
        # Also used for BSA which expects [seq_len, num_heads, head_dim]
        shape = (1, num_heads, max_seq_len, head_dim)
        self._k_expanded = torch.empty(shape, dtype=dtype, device=device)
        self._v_expanded = torch.empty(shape, dtype=dtype, device=device)
        self._max_seq_len = max_seq_len

        memory_mb = 2 * num_heads * max_seq_len * head_dim * dtype.itemsize / (1024 * 1024)
        logger.info(f"[XAttn] Pre-allocated GQA buffers: shape={shape}, memory={memory_mb:.1f} MB")

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
        # When block_tables is provided (paged KV cache / prefix cache),
        # fallback to flash_attn as XAttention expects contiguous K, V
        if block_tables is not None:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                block_table=block_tables,
            )

        if not BSA_AVAILABLE:
            # Fallback to flash attention if BSA not available
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
            )

        if not XATTN_AVAILABLE:
            # Fallback to flash attention if xattn not available
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
            )

        from nanovllm.ops.xattn import xattn_estimate

        # Get dimensions
        total_q, num_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape

        # For now, assume batch_size = 1 (single sequence)
        # TODO: Support batched varlen format
        batch_size = cu_seqlens_q.shape[0] - 1
        if batch_size != 1:
            # Fallback to flash attention for batched input
            from flash_attn import flash_attn_varlen_func
            logger.warning(f"[XAttn] batch_size={batch_size} > 1, falling back to flash attention")
            return flash_attn_varlen_func(
                q, k, v,
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
        _, mask = xattn_estimate(
            Q, K_exp,
            chunk_size=self.chunk_size,
            block_size=self.BSA_BLOCK_SIZE,
            threshold=self.threshold,
            use_triton=self.use_triton,
            causal=True,
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
        output = block_sparse_attn_func(
            q_bsa, k_bsa, v_bsa,
            cu_seqlens_q_bsa,
            cu_seqlens_k_bsa,
            head_groups,
            None,  # key_padding_mask
            mask_trimmed,
            q_len, k_len,
            p_dropout=0.0,
            deterministic=True,
            is_causal=True,
        )

        # Update statistics (layer 0 only to avoid overcounting)
        if layer_id == 0:
            selected_blocks = mask_trimmed.sum().item()
            total_blocks = q_block_num * k_block_num * num_heads
            density = selected_blocks / total_blocks if total_blocks > 0 else 1.0
            logger.debug(f"[XAttn GPU-only] layer={layer_id}, q_blocks={q_block_num}, "
                        f"k_blocks={k_block_num}, density={density:.1%}")

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
    ) -> List[int]:
        """
        Compute attention scores for all available blocks using flat_group_gemm,
        then use softmax_fuse_block_sum and find_blocks_chunked to select important blocks.

        This method:
        1. Loads each K block from CPU
        2. Computes Q@K^T attention scores using XAttention stride reshape
        3. Applies softmax_fuse_block_sum to get block-level attention
        4. Uses find_blocks_chunked to select blocks based on threshold

        Args:
            available_blocks: List of CPU block IDs
            offload_engine: OffloadEngine for loading blocks
            ctx: PolicyContext with query tensor and metadata

        Returns:
            Selected block IDs based on attention threshold
        """
        if not available_blocks or ctx.query is None:
            return available_blocks

        from nanovllm.ops.xattn import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum, find_blocks_chunked
        import math

        layer_id = ctx.layer_id
        q = ctx.query  # [seq_len, num_heads, head_dim]

        # Convert Q to [batch, heads, seq_len, head_dim]
        # q: [seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
        Q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_dim]

        num_heads = Q.shape[1]
        head_dim = Q.shape[3]
        q_len = Q.shape[2]

        # flat_group_gemm requires q_len to be divisible by stride * BLOCK_M (typically 8 * 128 = 1024)
        # Pad Q if necessary
        BLOCK_M = 128  # Triton block size
        alignment = self.stride * BLOCK_M
        if q_len < alignment:
            # Q too short, skip estimation and return all blocks
            logger.debug(f"[XAttn] select_blocks: q_len={q_len} < alignment={alignment}, skipping estimation")
            return available_blocks

        # Pad Q to alignment
        padded_q_len = ((q_len + alignment - 1) // alignment) * alignment
        if padded_q_len != q_len:
            pad_size = padded_q_len - q_len
            Q = torch.nn.functional.pad(Q, (0, 0, 0, pad_size), value=0)

        q_reshaped_len = padded_q_len // self.stride

        # Use a single slot for loading (synchronous mode for simplicity)
        slot = 0
        attn_scores_list = []

        # Get block size from context
        block_size = ctx.block_size  # tokens per CPU block (e.g., 1024)
        reshaped_block_size = block_size // self.stride  # e.g., 1024/8 = 128

        for cpu_block_id in available_blocks:
            # Load K block from CPU to GPU (cpu_block_id is chunk index)
            offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id)
            offload_engine.wait_slot_layer(slot)

            # Get KV: [1, block_size, num_kv_heads, head_dim]
            k_block, _ = offload_engine.get_kv_for_slot(slot)

            # Convert K to [batch, heads, k_len, head_dim]
            # k_block: [1, block_size, num_kv_heads, head_dim] -> [1, num_kv_heads, block_size, head_dim]
            K_chunk = k_block.transpose(1, 2)

            # Handle GQA: expand K heads to match Q heads
            num_kv_heads = K_chunk.shape[1]
            if num_heads != num_kv_heads:
                num_groups = num_heads // num_kv_heads
                K_chunk = K_chunk.repeat_interleave(num_groups, dim=1)

            # Pad K if necessary (k_len must be divisible by stride * BLOCK_N)
            k_len = K_chunk.shape[2]
            BLOCK_N = 128
            k_alignment = self.stride * BLOCK_N
            if k_len < k_alignment:
                # K too short, pad it
                pad_size = k_alignment - k_len
                K_chunk = torch.nn.functional.pad(K_chunk, (0, 0, 0, pad_size), value=0)

            # Compute attention scores using flat_group_gemm_fuse_reshape
            # Output: [batch, heads, q_len/stride, k_len/stride]
            attn_chunk = flat_group_gemm_fuse_reshape(
                Q, K_chunk, self.stride,
                chunk_start=0,
                chunk_end=q_reshaped_len,
                is_causal=False
            )
            attn_scores_list.append(attn_chunk)

            # Mark slot as done for reuse
            offload_engine.record_slot_compute_done(slot)

        # Concatenate all attention scores along K dimension
        # Each chunk: [1, heads, q_reshaped_len, block_reshaped_len]
        # Result: [1, heads, q_reshaped_len, total_k_reshaped_len]
        if not attn_scores_list:
            return available_blocks

        attn_scores = torch.cat(attn_scores_list, dim=-1)
        # Free intermediate list immediately
        del attn_scores_list

        # Step 2: Apply softmax_fuse_block_sum to get block-level attention
        # block_size = reshaped_block_size so each CPU block maps to exactly 1 output block
        # This ensures block_sums.shape[-1] == num_available_blocks (1:1 mapping)
        norm = 1.0  # Normalization factor
        scale = 1.4426950408889634 / math.sqrt(head_dim) / self.stride / norm  # log2(e) with scaling
        segment_size = min(4096, reshaped_block_size)

        block_sums = softmax_fuse_block_sum(
            attn_scores,
            reshaped_block_size,  # Use CPU block size in reshaped space (1024/8=128)
            segment_size,
            chunk_start=0,
            chunk_end=q_reshaped_len,
            real_q_len=q_reshaped_len,
            scale=scale,
            is_causal=False,  # Historical blocks are all before current chunk
        )
        # block_sums shape: [batch, heads, q_blocks, k_blocks]
        # where k_blocks == len(available_blocks) (1:1 mapping with CPU blocks)

        # Step 3: Use find_blocks_chunked to get selection mask
        # current_index = 0 since we're looking at historical blocks only
        mask = find_blocks_chunked(
            block_sums,
            current_index=0,
            threshold=self.threshold,
            num_to_choose=None,
            decoding=False,
            mode="prefill",
            causal=False,  # Historical blocks don't need causal mask
        )
        # mask shape: [batch, num_heads, q_blocks, k_blocks] - boolean
        # where k_blocks == len(available_blocks)

        # GQA-aware aggregation:
        # For GQA, multiple Q heads share one KV head. We need to select a block
        # if ANY Q head within the same KV head group selects it.
        # mask: [batch, num_heads, q_blocks, k_blocks]
        # Reshape to [batch, num_kv_heads, num_groups, q_blocks, k_blocks]
        batch_size, num_q_heads, q_blocks, k_blocks = mask.shape
        # num_kv_heads was set in the K loading loop above (line ~199)
        # num_groups = num_heads // num_kv_heads (for GQA)
        num_groups = num_heads // num_kv_heads if num_heads != num_kv_heads else 1

        if num_groups > 1:
            # Reshape: [batch, num_kv_heads, num_groups, q_blocks, k_blocks]
            mask_gqa = mask.view(batch_size, num_kv_heads, num_groups, q_blocks, k_blocks)
            # Aggregate within each KV head group: any Q head selects -> KV head selects
            mask_per_kv_head = mask_gqa.any(dim=2)  # [batch, num_kv_heads, q_blocks, k_blocks]
        else:
            mask_per_kv_head = mask  # [batch, num_heads, q_blocks, k_blocks]

        # Aggregate across KV heads and q_blocks using majority voting
        # Instead of any(), use voting: select if >50% of kv_heads select it
        # mask_per_kv_head: [batch, num_kv_heads, q_blocks, k_blocks]
        # Sum across kv_heads and q_blocks to get vote count per k_block
        vote_count = mask_per_kv_head[0].float().sum(dim=0).sum(dim=0)  # [k_blocks]
        total_votes = num_kv_heads * q_blocks
        vote_ratio = vote_count / total_votes

        # Select blocks with >50% votes (majority voting)
        vote_threshold = 0.5
        block_selected = vote_ratio > vote_threshold
        selected_block_ids = [available_blocks[i] for i, sel in enumerate(block_selected.tolist()) if sel]

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

            # Log per-chunk density
            chunk_density = len(selected_block_ids) / len(available_blocks)
            logger.debug(f"[XAttn] chunk={ctx.query_chunk_idx}, available={len(available_blocks)}, "
                        f"selected={len(selected_block_ids)}, chunk_density={chunk_density:.1%}")

        # Free intermediate tensors to prevent memory leak
        del attn_scores, block_sums, mask, mask_per_kv_head, vote_count, vote_ratio, block_selected

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
        from nanovllm.ops.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Use the pre-selected blocks directly
        cpu_block_table = selected_blocks

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            if len(load_slots) == 1:
                # Only 1 slot - use synchronous mode
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id)
                    offload_engine.wait_slot_layer(slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_batched, prev_k, prev_v,
                            softmax_scale=softmax_scale,
                            causal=False,
                        )
                        if o_acc is None:
                            o_acc, lse_acc = prev_o, prev_lse
                        else:
                            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)
                        offload_engine.record_slot_compute_done(slot)
            else:
                # Multiple slots - use pipeline
                num_slots = len(load_slots)
                num_preload = min(num_slots, num_blocks)
                for i in range(num_preload):
                    cpu_block_id = cpu_block_table[i]
                    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_id, chunk_idx=cpu_block_id)

                for block_idx in range(num_blocks):
                    current_slot = load_slots[block_idx % num_slots]

                    offload_engine.wait_slot_layer(current_slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_batched, prev_k, prev_v,
                            softmax_scale=softmax_scale,
                            causal=False,
                        )
                        offload_engine.record_slot_compute_done(current_slot)

                        if o_acc is None:
                            o_acc, lse_acc = prev_o, prev_lse
                        else:
                            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

                    # Issue next transfer
                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id, chunk_idx=next_cpu_block_id)

        # Compute attention to current chunk (causal mask)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            current_o, current_lse = flash_attn_with_lse(
                q_batched, k_curr, v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        # Merge historical and current attention
        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

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
            "overall_density": self._stats_total_selected_blocks / self._stats_total_available_blocks,
        }

    def print_density_stats(self) -> None:
        """Print density statistics summary."""
        stats = self.get_density_stats()
        logger.info(f"[XAttn BSA] Density Stats: chunks={stats['num_chunks']}, "
                   f"available={stats['total_available_blocks']}, "
                   f"selected={stats['total_selected_blocks']}, "
                   f"density={stats['overall_density']:.1%}")

    def __repr__(self) -> str:
        return f"XAttentionBSAPolicy(threshold={self.threshold}, stride={self.stride})"
