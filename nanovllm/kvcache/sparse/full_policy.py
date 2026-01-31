"""
Full attention policy - loads all blocks (no sparsity).

This serves as a baseline and default policy when sparse
attention is not needed.
"""

import logging
import torch
from typing import List, Optional, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext
from nanovllm.utils.context import get_context

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class FullAttentionPolicy(SparsePolicy):
    """
    Full attention policy that loads all available blocks.

    This is the default behavior with no sparsity - all previous
    KV cache blocks are loaded for each query chunk.

    Use this as:
    - A baseline for comparing sparse policies
    - When you need full attention accuracy
    - For short sequences where sparsity isn't beneficial
    """

    # Full attention supports both prefill and decode
    supports_prefill = True
    supports_decode = True

    def __init__(self):
        """Initialize with statistics tracking."""
        self._stats_total_blocks = 0
        self._stats_num_chunks = 0

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """Return all blocks - no sparsity."""
        # Update statistics (only for layer 0 to avoid overcounting)
        if ctx.layer_id == 0 and available_blocks:
            self._stats_total_blocks += len(available_blocks)
            self._stats_num_chunks += 1
            logger.debug(f"[Full] chunk={ctx.query_chunk_idx}, blocks={len(available_blocks)}, density=100.0%")
        return available_blocks

    def reset_stats(self) -> None:
        """Reset density statistics."""
        self._stats_total_blocks = 0
        self._stats_num_chunks = 0

    def get_density_stats(self) -> dict:
        """Get density statistics."""
        return {
            "total_available_blocks": self._stats_total_blocks,
            "total_selected_blocks": self._stats_total_blocks,  # Full = all selected
            "num_chunks": self._stats_num_chunks,
            "overall_density": 1.0,  # Always 100%
        }

    def print_density_stats(self) -> None:
        """Print density statistics summary."""
        stats = self.get_density_stats()
        logger.info(f"[Full Policy] Density Stats: chunks={stats['num_chunks']}, "
                   f"blocks={stats['total_available_blocks']}, density=100.0%")

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
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        GPU-only prefill attention using flash_attn_varlen_func.

        This is the simplest implementation - just call flash attention directly.
        For sparse policies, this method would implement block selection.
        """
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

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        GPU-only decode attention using flash_attn_with_kvcache.

        This is the simplest implementation - just call flash attention directly.
        For sparse policies, this method would implement block selection.
        """
        from flash_attn import flash_attn_with_kvcache

        # q is [batch, num_heads, head_dim], need to add seq dim
        return flash_attn_with_kvcache(
            q.unsqueeze(1),  # [batch, 1, heads, dim]
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )

    # =========================================================================
    # Chunked offload methods
    # =========================================================================

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
        Compute full attention for chunked prefill.

        This method handles the chunked prefill computation:
        1. Load and compute attention to historical chunks (using selected_blocks)
        2. Compute attention to current chunk
        3. Merge all results

        Note: Block selection is done by the caller before invoking this method.

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
            selected_blocks: List of CPU block IDs to process (already filtered)

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        # Use FlashInfer-based implementations (more optimized)
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        logger.debug(f"[DEBUG] FullPolicy.compute_chunked_prefill called, "
                     f"layer={layer_id}, chunk={current_chunk_idx}, num_tokens={num_tokens}, "
                     f"selected_blocks={len(selected_blocks)}")

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
                    # cpu_block_id is the chunk index (block N = chunk N)
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
                    cpu_block_id = cpu_block_table[block_idx]

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

        # Step 4: Compute attention to current chunk (causal mask)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            current_o, current_lse = flash_attn_with_lse(
                q_batched, k_curr, v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        # Step 5: Merge historical and current attention
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
        Compute full attention for chunked decode.

        This method handles the chunked decode computation:
        1. Load blocks via pipeline using selected_blocks (ring buffer or cross-layer)
        2. Read accumulated decode tokens from decode buffer
        3. Merge all results

        Note: Block selection is done by the caller before invoking this method.

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
        # Use FlashInfer-based implementations (more optimized)
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

        # Use the pre-selected blocks directly
        cpu_block_table = selected_blocks
        if layer_id == 0:
            logger.debug(f"Decode attention: selected_blocks={len(selected_blocks)}, seq.block_table={list(seq.block_table)}")
        if not cpu_block_table:
            raise RuntimeError("Chunked decode attention failed: no prefilled CPU blocks available")

        # Calculate valid tokens in the last CPU block
        # CRITICAL: Use original prefill length, not current seq length!
        # CPU blocks are fixed after prefill, their content doesn't change during decode.
        # Note: We need to get all prefilled blocks to determine last_block_valid_tokens
        block_size = kvcache_manager.block_size
        all_prefilled_blocks = kvcache_manager.get_prefilled_cpu_blocks(seq)
        total_prefill_tokens = kvcache_manager.get_prefill_len(seq)  # Original prefill length
        last_block_valid_tokens = total_prefill_tokens % block_size
        if last_block_valid_tokens == 0 and total_prefill_tokens > 0:
            last_block_valid_tokens = block_size  # Last block was exactly full

        # Determine if selected_blocks contains the last prefilled block
        # If not, all selected blocks are full blocks (use block_size as valid tokens)
        last_prefilled_block = all_prefilled_blocks[-1] if all_prefilled_blocks else None
        selected_contains_last = (cpu_block_table and cpu_block_table[-1] == last_prefilled_block)
        effective_last_block_tokens = last_block_valid_tokens if selected_contains_last else block_size

        # Use ring buffer pipeline for loading prefilled blocks
        load_slots = offload_engine.decode_load_slots
        o_acc, lse_acc = self._decode_ring_buffer_pipeline(
            q_batched, cpu_block_table, load_slots, offload_engine,
            block_size, effective_last_block_tokens, layer_id, softmax_scale
        )

        # Now attend to accumulated decode tokens from per-layer decode buffer
        # Compute decode position information internally
        seq_len = len(seq)
        decode_pos_in_block = (seq_len - 1) % block_size
        decode_start_pos = kvcache_manager.get_decode_start_pos(seq)
        decode_start_pos_in_block = decode_start_pos % block_size
        num_accumulated = decode_pos_in_block - decode_start_pos_in_block + 1

        # Sync compute_stream with default stream before reading decode_buffer
        compute_stream = offload_engine.compute_stream
        compute_stream.wait_stream(torch.cuda.default_stream())

        with torch.cuda.stream(compute_stream):
            if num_accumulated > 0:
                # Read from per-layer decode buffer
                decode_k = offload_engine.decode_k_buffer[layer_id, decode_start_pos_in_block:decode_pos_in_block+1]
                decode_v = offload_engine.decode_v_buffer[layer_id, decode_start_pos_in_block:decode_pos_in_block+1]
                decode_k = decode_k.unsqueeze(0)
                decode_v = decode_v.unsqueeze(0)

                decode_o, decode_lse = flash_attn_with_lse(
                    q_batched, decode_k, decode_v,
                    softmax_scale=softmax_scale,
                    causal=False,
                )

                if o_acc is None:
                    o_acc = decode_o
                else:
                    o_acc, _ = merge_attention_outputs(o_acc, lse_acc, decode_o, decode_lse)

        if o_acc is None:
            raise RuntimeError("Chunked decode attention failed: no KV available")

        # Sync back to default stream before returning
        torch.cuda.default_stream().wait_stream(compute_stream)

        return o_acc

    def _decode_ring_buffer_pipeline(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: list,
        load_slots: list,
        offload_engine: "OffloadEngine",
        block_size: int,
        last_block_valid_tokens: int,
        layer_id: int,
        softmax_scale: float,
    ):
        """
        Ring buffer pipeline for decode prefill loading.

        Loads one block at a time, computes attention, and merges results.
        Uses load_to_slot_layer / wait_slot_layer / get_kv_for_slot methods.
        """
        # Use FlashInfer-based implementations (more optimized)
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        num_blocks = len(cpu_block_table)
        if num_blocks == 0:
            return None, None

        if not load_slots:
            return None, None

        o_acc, lse_acc = None, None
        num_slots = len(load_slots)
        compute_stream = offload_engine.compute_stream

        # Phase 1: Pre-load up to num_slots blocks
        num_preload = min(num_slots, num_blocks)
        for i in range(num_preload):
            cpu_block_id = cpu_block_table[i]
            offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_id, chunk_idx=cpu_block_id, is_prefill=False)

        # Phase 2: Process blocks with pipeline
        for block_idx in range(num_blocks):
            current_slot = load_slots[block_idx % num_slots]
            cpu_block_id = cpu_block_table[block_idx]

            # Wait for current slot's transfer to complete
            offload_engine.wait_slot_layer(current_slot)

            with torch.cuda.stream(compute_stream):
                # Get KV from slot
                prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)

                # Handle partial last block
                is_last_block = (block_idx == num_blocks - 1)
                if is_last_block and last_block_valid_tokens < block_size:
                    prev_k = prev_k[:, :last_block_valid_tokens, :, :]
                    prev_v = prev_v[:, :last_block_valid_tokens, :, :]

                # Compute attention
                prev_o, prev_lse = flash_attn_with_lse(
                    q_batched, prev_k, prev_v,
                    softmax_scale=softmax_scale,
                    causal=False,
                )

                # Record compute done for slot reuse
                offload_engine.record_slot_compute_done(current_slot)

            # Start loading next block (pipeline)
            next_block_idx = block_idx + num_slots
            if next_block_idx < num_blocks:
                next_cpu_block_id = cpu_block_table[next_block_idx]
                offload_engine.load_to_slot_layer(current_slot, layer_id, next_cpu_block_id, chunk_idx=next_cpu_block_id, is_prefill=False)

            # Merge with accumulated
            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

        return o_acc, lse_acc

    def __repr__(self) -> str:
        return "FullAttentionPolicy()"
