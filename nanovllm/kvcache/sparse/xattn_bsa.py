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
        threshold: float = 0.9,
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

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Return all blocks - actual selection happens in compute_chunked_prefill.
        """
        return available_blocks

    def _load_all_historical_kv(
        self,
        cpu_block_table: List[int],
        layer_id: int,
        offload_engine: "OffloadEngine",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load all historical K/V from CPU to GPU.

        Args:
            cpu_block_table: List of CPU block IDs
            layer_id: Current layer index
            offload_engine: OffloadEngine instance

        Returns:
            (k_hist, v_hist) with shape [total_tokens, kv_heads, head_dim]
        """
        if not cpu_block_table:
            return None, None

        k_list = []
        v_list = []

        for cpu_block_id in cpu_block_table:
            k_block, v_block = offload_engine.load_block_full_from_cpu(
                cpu_block_id, layer_id
            )
            k_list.append(k_block)
            v_list.append(v_block)

        # Concatenate: [num_blocks, block_size, kv_heads, head_dim] -> [total_tokens, kv_heads, head_dim]
        k_hist = torch.cat(k_list, dim=0)
        v_hist = torch.cat(v_list, dim=0)

        return k_hist, v_hist

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
    ) -> torch.Tensor:
        """
        Compute attention for chunked prefill.

        NOTE: The current XAttention + BSA implementation has memory issues
        (loads all historical K/V at once, losing the benefit of sparse attention).
        Until a proper ring-buffer-based sparse implementation is ready,
        we fallback to the dense attention pipeline which is memory-efficient.

        TODO: Implement proper sparse attention with ring buffer pipeline:
        1. Use xattn_estimate_chunked to identify important blocks
        2. Only load selected blocks using ring buffer
        3. Compute sparse attention on selected blocks only

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

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        # Use dense fallback which is memory-efficient (ring buffer pipeline)
        # This is temporary until proper sparse implementation is ready
        return self._compute_dense_fallback(
            q, k, v, layer_id, softmax_scale, offload_engine,
            kvcache_manager, current_chunk_idx, seq, num_tokens
        )

    def _compute_dense_fallback(
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
    ) -> torch.Tensor:
        """
        Fallback to dense attention when BSA/XAttn not available.
        Uses FullAttentionPolicy's proven pipeline.
        """
        from nanovllm.ops.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        logger.debug(f"[XAttn] FALLBACK to dense: layer={layer_id}, chunk={current_chunk_idx}")

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Get historical CPU blocks
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

        # Process historical blocks using pipeline
        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            if len(load_slots) == 1:
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
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
                num_slots = len(load_slots)
                num_preload = min(num_slots, num_blocks)
                for i in range(num_preload):
                    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

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

                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id)

        # Compute attention to current chunk (causal)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            current_o, current_lse = flash_attn_with_lse(
                q_batched, k_curr, v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        # Merge historical and current
        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

        torch.cuda.default_stream().wait_stream(compute_stream)
        return final_o.squeeze(0)

    def compute_chunked_decode(
        self,
        q: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        seq: "Sequence",
    ) -> torch.Tensor:
        """
        XAttention does not support decode phase.
        """
        raise NotImplementedError(
            "XAttentionBSAPolicy does not support decode phase. "
            "Use FullAttentionPolicy for decode."
        )

    def reset(self) -> None:
        """Reset policy state."""
        pass

    def __repr__(self) -> str:
        return f"XAttentionBSAPolicy(threshold={self.threshold}, stride={self.stride})"
