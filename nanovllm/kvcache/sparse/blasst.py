"""
BLASST sparse attention policy for chunked prefill.

BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

Key insight: Use online softmax running max to dynamically skip unimportant blocks.
Skip condition: local_max - running_max < ln(λ)
Threshold formula: λ = a / L (inverse with sequence length)

Implementation notes:
- IO communication: Same as FullAttention (load all blocks)
- Compute decision: Skip merging blocks based on BLASST condition
- Running max: Extracted from LSE (log-sum-exp) of flash attention
"""

import logging
import math
import torch
from typing import List, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class BLASSTPolicy(SparsePolicy):
    """
    BLASST sparse attention policy.

    Implementation:
    - select_blocks: Returns all available blocks (load all, decide later)
    - compute_chunked_prefill: Dynamic skip based on running max threshold

    Threshold configuration:
    - a: Inverse formula parameter (λ = a / L), default 16384
    - fixed_lambda: If set, use fixed threshold instead of formula
    """

    # BLASST supports both prefill and decode
    supports_prefill = True
    supports_decode = True

    def __init__(self, a: int = 16384, fixed_lambda: float = None):
        """
        Initialize BLASST policy.

        Args:
            a: Inverse formula numerator (λ = a / L). Default 16384 gives λ=0.5 at 32K.
            fixed_lambda: If set, use fixed threshold instead of inverse formula.
        """
        self.a = a
        self.fixed_lambda = fixed_lambda
        self._stats_num_chunks = 0
        self._stats_skipped_blocks = 0
        self._stats_total_blocks = 0

    def _get_lambda(self, seq_len: int) -> float:
        """
        Compute threshold λ based on sequence length.

        Formula: λ = a / L (inverse relationship)

        Args:
            seq_len: Current sequence length

        Returns:
            Threshold value λ
        """
        if self.fixed_lambda is not None:
            return self.fixed_lambda
        # Inverse formula: longer sequences → smaller threshold
        return self.a / max(seq_len, 1)

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

        TODO: Implement BLASST block selection logic.
        For now, this just tests the chunked prefill flow with full attention.
        """
        # For now, return all blocks to test basic chunked prefill flow
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1
            logger.debug(f"[BLASST] chunk={ctx.query_chunk_idx}, "
                        f"available={len(available_blocks)}, selected={len(available_blocks)}")
        return available_blocks

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats_num_chunks = 0
        self._stats_skipped_blocks = 0
        self._stats_total_blocks = 0

    def get_stats(self) -> dict:
        """Get statistics."""
        skip_rate = 0.0
        if self._stats_total_blocks > 0:
            skip_rate = self._stats_skipped_blocks / self._stats_total_blocks
        return {
            "num_chunks": self._stats_num_chunks,
            "skipped_blocks": self._stats_skipped_blocks,
            "total_blocks": self._stats_total_blocks,
            "skip_rate": skip_rate,
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
            "BLASST policy only supports chunked prefill mode. "
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
            "BLASST policy only supports chunked decode mode. "
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
        Compute attention for chunked prefill with BLASST dynamic skipping.

        BLASST Algorithm:
        1. Load all KV blocks (same as FullAttention)
        2. For each block, compute attention and extract local_max from LSE
        3. Apply skip condition: local_max - running_max < ln(λ)
        4. Skip merging if condition met, otherwise merge and update running_max

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
        # Use FlashInfer-based implementations
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        # Calculate sequence length for threshold computation
        # Total length = already prefilled + current chunk
        total_seq_len = len(seq) if seq else num_tokens
        lambda_val = self._get_lambda(total_seq_len)
        ln_lambda = math.log(lambda_val)

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Running max: per-token, cross-heads aggregation
        # Shape: [q_len], initialize with -inf
        running_max = torch.full(
            (q.shape[0],),
            float('-inf'),
            device=q.device,
            dtype=torch.float32
        )

        # Track skip statistics
        skipped_blocks = 0
        total_historical_blocks = 0

        # Load and compute attention on historical blocks with BLASST skipping
        cpu_block_table = selected_blocks

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)
            total_historical_blocks = num_blocks

            # Log at layer 0 for debugging
            if layer_id == 0:
                logger.info(f"[BLASST] Chunk {current_chunk_idx}: "
                           f"seq_len={total_seq_len}, λ={lambda_val:.4f}, "
                           f"blocks={num_blocks}")

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

                        # BLASST: Extract local_max from LSE
                        # LSE shape: [1, num_heads, q_len] or [1, q_len, num_heads]
                        # -> max over heads to get [q_len]
                        lse_squeezed = prev_lse.squeeze(0)  # [num_heads, q_len] or [q_len, num_heads]
                        local_max = lse_squeezed.max(dim=0)[0]  # Try dim=0 first
                        if local_max.shape[0] != q.shape[0]:
                            # Shape mismatch, try dim=-1
                            local_max = lse_squeezed.max(dim=-1)[0]
                        # Ensure shape is [q_len]
                        if local_max.dim() == 0:
                            local_max = local_max.unsqueeze(0)
                        local_max = local_max.to(running_max.dtype)

                        # BLASST skip condition: local_max - running_max < ln(λ)
                        skip_mask = (local_max - running_max) < ln_lambda

                        if skip_mask.all():
                            # All queries skip this block
                            skipped_blocks += 1
                            # Still update running_max for subsequent decisions
                            running_max = torch.maximum(running_max, local_max)
                        else:
                            # At least some queries need this block
                            if o_acc is None:
                                o_acc, lse_acc = prev_o, prev_lse
                            else:
                                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)
                            # Update running_max only when block is used
                            running_max = torch.maximum(running_max, local_max)

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

                        # BLASST: Extract local_max from LSE
                        # LSE shape: [1, num_heads, q_len] or [1, q_len, num_heads]
                        # -> max over heads to get [q_len]
                        lse_squeezed = prev_lse.squeeze(0)  # [num_heads, q_len] or [q_len, num_heads]
                        local_max = lse_squeezed.max(dim=0)[0]  # Try dim=0 first
                        if local_max.shape[0] != q.shape[0]:
                            # Shape mismatch, try dim=-1
                            local_max = lse_squeezed.max(dim=-1)[0]
                        # Ensure shape is [q_len]
                        if local_max.dim() == 0:
                            local_max = local_max.unsqueeze(0)
                        local_max = local_max.to(running_max.dtype)

                        # BLASST skip condition: decision based on current running_max
                        skip_mask = (local_max - running_max) < ln_lambda

                        if skip_mask.all():
                            # All queries skip this block
                            skipped_blocks += 1
                            # Still update running_max for subsequent decisions
                            running_max = torch.maximum(running_max, local_max)
                        else:
                            # At least some queries need this block
                            if o_acc is None:
                                o_acc, lse_acc = prev_o, prev_lse
                            else:
                                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)
                            # Update running_max only when block is used
                            running_max = torch.maximum(running_max, local_max)

                        offload_engine.record_slot_compute_done(current_slot)

                    # Issue next transfer
                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id, chunk_idx=next_cpu_block_id)

            # Update statistics
            self._stats_skipped_blocks += skipped_blocks
            self._stats_total_blocks += total_historical_blocks

            if layer_id == 0:
                skip_rate = skipped_blocks / total_historical_blocks if total_historical_blocks > 0 else 0.0
                logger.info(f"[BLASST] Chunk {current_chunk_idx}: skipped {skipped_blocks}/{total_historical_blocks} "
                           f"({skip_rate:.1%}) blocks")

        # Compute attention to current chunk (causal mask, always included)
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

        # Remove batch dimension
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

        BLASST uses full attention on all prefilled blocks during decode
        (since select_blocks returns all available blocks).

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
        # BLASST uses FullAttentionPolicy for decode computation
        # (select_blocks already returns all blocks)
        from .full_policy import FullAttentionPolicy

        logger.debug(f"[BLASST] compute_chunked_decode using FullAttentionPolicy, "
                    f"layer={layer_id}, selected_blocks={len(selected_blocks)}")

        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq, selected_blocks
        )

    def __repr__(self) -> str:
        if self.fixed_lambda is not None:
            return f"BLASSTPolicy(fixed_lambda={self.fixed_lambda})"
        return f"BLASSTPolicy(a={self.a})"
