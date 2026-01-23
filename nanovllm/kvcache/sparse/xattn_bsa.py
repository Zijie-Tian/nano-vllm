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
        threshold: float = 0.1,  # Very low threshold for aggressive sparsity testing
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
            # Load K block from CPU to GPU
            offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
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
        # Store in sparse_metadata for later use in compute_chunked_prefill
        self.sparse_metadata[layer_id] = attn_scores

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

        # Compute density = selected / total
        density = len(selected_block_ids) / len(available_blocks) if available_blocks else 1.0

        # Debug output: show block selection results
        if layer_id == 0:  # Only log for layer 0 to avoid spam
            # Count True per head to see head-level sparsity
            # mask shape: [batch, num_heads, q_blocks, k_blocks]
            per_head_selected = mask[0, :, 0, :].sum(dim=-1)  # [num_heads] - selected blocks per head
            per_head_density = per_head_selected.float() / k_blocks

            print(f"[XAttn DEBUG] chunk={ctx.query_chunk_idx}, "
                  f"blocks={len(available_blocks)}, "
                  f"final_selected={len(selected_block_ids)}, "
                  f"final_density={density:.1%}, "
                  f"per_head_density={[f'{d:.0%}' for d in per_head_density[:8].tolist()]}...")  # First 8 heads

            # Exit early after 40 chunks for faster debugging
            if ctx.query_chunk_idx >= 40:
                print(f"[XAttn DEBUG] Exiting early after {ctx.query_chunk_idx} chunks for debugging")
                import sys
                sys.exit(0)

        # Always include first block (sink) and last block for safety
        if available_blocks and available_blocks[0] not in selected_block_ids:
            selected_block_ids.insert(0, available_blocks[0])
        if available_blocks and available_blocks[-1] not in selected_block_ids:
            selected_block_ids.append(available_blocks[-1])

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

        TODO: Implement sparse attention computation using selected_blocks.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim] (current chunk)
            v: Value tensor [seq_len, num_kv_heads, head_dim] (current chunk)
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
        # TODO: Implement sparse attention with selected_blocks
        # For now, return zeros as placeholder
        return torch.zeros_like(q)

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

    def __repr__(self) -> str:
        return f"XAttentionBSAPolicy(threshold={self.threshold}, stride={self.stride})"
