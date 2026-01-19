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

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
    ) -> List[int]:
        """Return all blocks - no sparsity."""
        return available_blocks

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
        Compute full attention for chunked prefill.

        This method handles the complete chunked prefill flow:
        1. Get historical blocks
        2. Select blocks via select_blocks
        3. Load and compute attention to historical chunks
        4. Compute attention to current chunk
        5. Merge all results

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

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        logger.debug(f"[DEBUG] FullPolicy.compute_chunked_prefill called, "
                     f"layer={layer_id}, chunk={current_chunk_idx}, num_tokens={num_tokens}")

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Step 1: Get historical blocks
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

        # Step 2: Apply select_blocks to filter blocks
        if cpu_block_table:
            num_chunks = current_chunk_idx + 1
            policy_ctx = PolicyContext(
                query_chunk_idx=current_chunk_idx,
                num_query_chunks=num_chunks,
                layer_id=layer_id,
                query=None,  # Prefill typically doesn't use query for selection
                is_prefill=True,
                block_size=kvcache_manager.block_size,
                total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
            )
            cpu_block_table = self.select_blocks(cpu_block_table, offload_engine, policy_ctx)
            logger.debug(f"[DEBUG] select_blocks: output={len(cpu_block_table)} blocks")

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            if len(load_slots) == 1:
                # Only 1 slot - use synchronous mode
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
                # Multiple slots - use pipeline
                num_slots = len(load_slots)
                num_preload = min(num_slots, num_blocks)
                for i in range(num_preload):
                    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

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
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id)

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
    ) -> torch.Tensor:
        """
        Compute full attention for chunked decode.

        This method handles the complete chunked decode flow:
        1. Get prefilled CPU blocks
        2. Apply select_blocks for block filtering
        3. Load blocks via pipeline (ring buffer or cross-layer)
        4. Read accumulated decode tokens from decode buffer
        5. Merge all results

        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            seq: Sequence object

        Returns:
            Attention output [batch_size, 1, num_heads, head_dim]
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

        # Get only PREFILLED CPU blocks (exclude the current decode block)
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)
        if layer_id == 0:
            logger.debug(f"Decode attention: cpu_block_table={cpu_block_table}, seq.block_table={list(seq.block_table)}")
        if not cpu_block_table:
            raise RuntimeError("Chunked decode attention failed: no prefilled CPU blocks available")

        # Calculate valid tokens in the last CPU block
        # CRITICAL: Use original prefill length, not current seq length!
        # CPU blocks are fixed after prefill, their content doesn't change during decode.
        block_size = kvcache_manager.block_size
        num_prefill_blocks = len(cpu_block_table)
        total_prefill_tokens = kvcache_manager.get_prefill_len(seq)  # Original prefill length
        last_block_valid_tokens = total_prefill_tokens % block_size
        if last_block_valid_tokens == 0 and total_prefill_tokens > 0:
            last_block_valid_tokens = block_size  # Last block was exactly full

        # Apply sparse policy (self) for block filtering
        policy_ctx = PolicyContext(
            query_chunk_idx=0,
            num_query_chunks=1,
            layer_id=layer_id,
            query=q_batched,
            is_prefill=False,
            block_size=kvcache_manager.block_size,
            total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
        )
        cpu_block_table = self.select_blocks(cpu_block_table, offload_engine, policy_ctx)

        # Use ring buffer pipeline for loading prefilled blocks
        load_slots = offload_engine.decode_load_slots
        o_acc, lse_acc = self._decode_ring_buffer_pipeline(
            q_batched, cpu_block_table, load_slots, offload_engine,
            block_size, last_block_valid_tokens, layer_id, softmax_scale
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
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

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
            offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

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
                offload_engine.load_to_slot_layer(current_slot, layer_id, cpu_block_table[next_block_idx])

            # Merge with accumulated
            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

        return o_acc, lse_acc

    def __repr__(self) -> str:
        return "FullAttentionPolicy()"
