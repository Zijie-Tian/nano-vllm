"""
Full attention policy - loads all blocks (no sparsity).

This serves as a baseline and default policy when sparse
attention is not needed.
"""

import torch
from typing import List, Optional

from .policy import SparsePolicy, PolicyContext
from nanovllm.utils.context import get_context


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
        ctx: PolicyContext,
    ) -> List[int]:
        """Return all blocks - no sparsity."""
        return available_blocks

    def compute_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine,
        current_chunk_idx: int,
        seq,
    ) -> torch.Tensor:
        """
        Compute full attention for chunked prefill.

        This method handles the complete chunked prefill flow:
        1. Load historical blocks from CPU
        2. Compute attention to historical chunks
        3. Compute attention to current chunk
        4. Merge all results

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim] (unused, from prefill buffer)
            v: Value tensor [seq_len, num_kv_heads, head_dim] (unused, from prefill buffer)
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            current_chunk_idx: Current chunk index
            seq: ChunkedSequence

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        num_tokens = q.shape[0]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Step 1: Get and load historical blocks
        cpu_block_table = seq.kvcache_manager.get_prefilled_cpu_blocks(seq)

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

        # Step 2: Compute attention to current chunk (causal mask)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            current_o, current_lse = flash_attn_with_lse(
                q_batched, k_curr, v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        # Step 3: Merge historical and current attention
        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

        # Sync default stream with compute_stream before returning
        torch.cuda.default_stream().wait_stream(compute_stream)

        # Remove batch dimension: [1, seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim]
        return final_o.squeeze(0)

    def __repr__(self) -> str:
        return "FullAttentionPolicy()"
