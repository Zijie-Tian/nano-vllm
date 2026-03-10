"""
COMPASS sparse attention policy for chunked prefill.

This is a placeholder implementation that:
1. select_blocks returns empty list (no historical blocks selected)
2. compute_chunked_prefill uses full attention computation on current chunk only

This serves as a baseline for development and testing of the chunked prefill flow
without loading any historical KV blocks.
"""

import logging
import torch
from typing import List, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class COMPASSPolicy(SparsePolicy):
    """
    COMPASS sparse attention policy.

    Current implementation:
    - select_blocks: Returns empty list (no historical blocks loaded)
    - compute_chunked_prefill: Only computes attention on current chunk (causal)

    This is essentially a "current chunk only" baseline for chunked prefill.
    """

    # COMPASS supports both prefill and decode for now
    supports_prefill = True
    supports_decode = True

    def __init__(self):
        """Initialize with statistics tracking."""
        self._stats_num_chunks = 0

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select blocks - currently returns all blocks (FullAttention behavior).

        TODO: Implement COMPASS block selection logic.
        For now, this just tests the chunked prefill flow with full attention.
        """
        # For now, return all blocks to test basic chunked prefill flow
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1
            logger.debug(
                f"[COMPASS] chunk={ctx.query_chunk_idx}, "
                f"available={len(available_blocks)}, selected={len(available_blocks)}"
            )
        return available_blocks

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats_num_chunks = 0

    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            "num_chunks": self._stats_num_chunks,
        }

    # ========================================================================
    # GPU-only methods (non-chunked) - Not supported for now
    # ========================================================================

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
        block_tables=None,
    ) -> torch.Tensor:
        """GPU-only prefill - not supported, use FullAttentionPolicy instead."""
        raise NotImplementedError(
            "COMPASS policy only supports chunked prefill mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables=None,
    ) -> torch.Tensor:
        """GPU-only decode - not supported, use FullAttentionPolicy instead."""
        raise NotImplementedError(
            "COMPASS policy only supports chunked decode mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    # ========================================================================
    # Chunked offload methods
    # ========================================================================

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
        Compute attention for chunked prefill.

        Currently implements full attention (loads all selected_blocks).
        TODO: Implement COMPASS sparse attention computation.

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
            selected_blocks: List of CPU block IDs to load

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        # Use FlashInfer-based implementations (same as FullAttentionPolicy)
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        logger.debug(
            f"[COMPASS] compute_chunked_prefill called, "
            f"layer={layer_id}, chunk={current_chunk_idx}, "
            f"num_tokens={num_tokens}, selected_blocks={len(selected_blocks)}"
        )

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Load and compute attention on selected historical blocks
        cpu_block_table = selected_blocks

        if cpu_block_table:
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
                        load_slots[i], layer_id, cpu_block_id, chunk_idx=cpu_block_id
                    )

                for block_idx in range(num_blocks):
                    current_slot = load_slots[block_idx % num_slots]

                    offload_engine.wait_slot_layer(current_slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
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
        Compute attention for chunked decode.

        COMPASS uses full attention on all prefilled blocks during decode
        (since select_blocks returns all available blocks in decode phase).

        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            seq: Sequence object
            selected_blocks: List of CPU block IDs to process (already filtered)

        Returns:
            Attention output [batch_size, 1, num_heads, head_dim]
        """
        # COMPASS uses FullAttentionPolicy for decode computation
        # (select_blocks already returns all blocks in decode phase)
        from .full_policy import FullAttentionPolicy

        logger.debug(
            f"[COMPASS] compute_chunked_decode using FullAttentionPolicy, "
            f"layer={layer_id}, selected_blocks={len(selected_blocks)}"
        )

        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q,
            layer_id,
            softmax_scale,
            offload_engine,
            kvcache_manager,
            seq,
            selected_blocks,
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
        return "COMPASSPolicy()"
