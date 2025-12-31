import logging
import torch
import torch.cuda.nvtx
from torch import nn
import triton
import triton.language as tl

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.kvcache.sparse.policy import PolicyContext

logger = logging.getLogger(__name__)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Layer ID set by model_runner after model creation
        self.layer_id: int = -1

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.is_chunked_prefill:
                # Chunked prefill: merge attention from previous KV
                o = self._chunked_prefill_attention(q, k, v, context)
            elif context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if context.is_chunked_prefill:
                # Chunked decode: need to load all KV from CPU+GPU
                o = self._chunked_decode_attention(q, k, v, context)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o

    def _chunked_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """
        Compute attention with unified ring buffer for chunked prefill.

        Ring buffer design:
        - Current chunk's KV is written to ring_slot[chunk_idx % N]
        - Previous chunks' KV are loaded from CPU using N-1 available slots
        - Pipeline: pre-fill slots, then process with overlapped load/compute

        For each layer:
        1. Current chunk's KV is in k_batched, v_batched (just written by model)
        2. Load previous chunks from CPU using available slots (pipeline)
        3. Compute attention against previous KV (no causal mask)
        4. Compute attention against current KV (causal)
        5. Merge all results using online softmax
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        current_chunk_idx = context.current_chunk_idx
        torch.cuda.nvtx.range_push(f"ChunkedPrefill: L{self.layer_id} Chunk{current_chunk_idx}")

        # q, k, v shape: [total_tokens, num_heads, head_dim]
        # Reshape for flash attention: [batch, seq, heads, dim]
        q_batched = q.unsqueeze(0)  # [1, total_tokens, heads, dim]
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)

        o_acc = None
        lse_acc = None

        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq if hasattr(context, 'chunked_seq') else None

        if kvcache_manager is not None and seq is not None and self.layer_id >= 0:
            # Get prefilled CPU blocks (blocks from previous chunks)
            cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

            # Apply sparse policy if enabled
            if cpu_block_table and kvcache_manager.sparse_policy is not None:
                num_chunks = getattr(context, 'num_chunks', current_chunk_idx + 1)
                policy_ctx = PolicyContext(
                    query_chunk_idx=current_chunk_idx,
                    num_query_chunks=num_chunks,
                    layer_id=self.layer_id,
                    query=None,  # Prefill typically doesn't use query for selection
                    is_prefill=True,
                    block_size=kvcache_manager.block_size,
                    total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
                )
                cpu_block_table = kvcache_manager.sparse_policy.select_blocks(
                    cpu_block_table, policy_ctx
                )

            if cpu_block_table:
                offload_engine = kvcache_manager.offload_engine

                # Get write slot for current chunk and available load slots
                write_slot = offload_engine.get_write_slot_for_prefill(current_chunk_idx)
                load_slots = offload_engine.get_load_slots_for_prefill(write_slot)
                pipeline_depth = len(load_slots)

                if pipeline_depth == 0:
                    # Only 1 slot total, cannot pipeline - use sync loading
                    o_acc, lse_acc = self._sync_load_previous_chunks(
                        q_batched, cpu_block_table, offload_engine
                    )
                else:
                    # Use ring buffer pipeline
                    o_acc, lse_acc = self._ring_buffer_pipeline_load(
                        q_batched, cpu_block_table, load_slots, offload_engine,
                        current_chunk_idx
                    )


        # Compute attention against current chunk's KV (with causal mask)
        torch.cuda.nvtx.range_push(f"FlashAttn: L{self.layer_id} CurrentChunk (causal)")
        current_o, current_lse = flash_attn_with_lse(
            q_batched,
            k_batched,
            v_batched,
            softmax_scale=self.scale,
            causal=True,
        )
        torch.cuda.nvtx.range_pop()

        # Merge with accumulated
        if o_acc is None:
            final_o = current_o
        else:
            # IMPORTANT: o_acc was computed on compute_stream. We need to sync before
            # reading it on the default stream for the merge operation.
            if kvcache_manager is not None and hasattr(kvcache_manager, 'offload_engine'):
                offload_engine = kvcache_manager.offload_engine
                torch.cuda.default_stream().wait_stream(offload_engine.compute_stream)

            torch.cuda.nvtx.range_push(f"MergeAttn: L{self.layer_id}")
            final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()  # ChunkedPrefill

        # Remove batch dimension: [1, total_tokens, heads, dim] -> [total_tokens, heads, dim]
        return final_o.squeeze(0)

    def _sync_load_previous_chunks(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: list,
        offload_engine,
    ):
        """Synchronous loading fallback when pipeline_depth=0."""
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        o_acc, lse_acc = None, None
        compute_stream = offload_engine.compute_stream

        for block_idx, cpu_block_id in enumerate(cpu_block_table):
            # Load to slot 0 (single slot)
            offload_engine.load_to_slot_layer(0, self.layer_id, cpu_block_id)
            offload_engine.wait_slot_layer(0, self.layer_id)

            # IMPORTANT: Must use compute_stream to match wait_slot_layer
            with torch.cuda.stream(compute_stream):
                prev_k, prev_v = offload_engine.get_kv_for_slot(0, self.layer_id)

                prev_o, prev_lse = flash_attn_with_lse(
                    q_batched, prev_k, prev_v,
                    softmax_scale=self.scale,
                    causal=False,
                )

                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

        return o_acc, lse_acc

    def _ring_buffer_pipeline_load(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: list,
        load_slots: list,
        offload_engine,
        current_chunk_idx: int = -1,
    ):
        """
        Ring buffer async pipeline loading with double buffering.

        Uses compute_done events to ensure safe buffer reuse:
        - Before loading to slot X, wait for previous compute on slot X to finish
        - Before computing on slot X, wait for load to slot X to finish

        Timeline with 2 slots (A, B):
        ┌──────────────┐
        │ Load B0→A    │
        └──────────────┘
                       ┌──────────────┐ ┌──────────────┐
                       │ Load B1→B    │ │ Load B2→A    │ ...
                       └──────────────┘ └──────────────┘
                                      ↘               ↘
                        ┌──────────────┐ ┌──────────────┐
                        │ Compute(A)   │ │ Compute(B)   │ ...
                        └──────────────┘ └──────────────┘

        The load_to_slot_layer internally waits for compute_done[slot] before
        starting the transfer, ensuring no data race.
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        num_blocks = len(cpu_block_table)
        if num_blocks == 0:
            return None, None

        pipeline_depth = len(load_slots)
        if pipeline_depth == 0:
            return None, None

        o_acc, lse_acc = None, None

        if pipeline_depth == 1:
            # Only 1 slot available, cannot pipeline - use synchronous mode
            # IMPORTANT: Must use compute_stream to match synchronization in
            # load_to_slot_layer (waits for compute_done) and wait_slot_layer
            slot = load_slots[0]
            compute_stream = offload_engine.compute_stream
            for block_idx in range(num_blocks):
                cpu_block_id = cpu_block_table[block_idx]
                offload_engine.load_to_slot_layer(slot, self.layer_id, cpu_block_id)
                offload_engine.wait_slot_layer(slot, self.layer_id)

                with torch.cuda.stream(compute_stream):
                    # Debug: call hooks on compute_stream (synchronized with transfer)
                    if offload_engine.debug_mode:
                        offload_engine._call_debug_hooks(slot, self.layer_id, cpu_block_id)

                    prev_k, prev_v = offload_engine.get_kv_for_slot(slot, self.layer_id)
                    prev_o, prev_lse = flash_attn_with_lse(
                        q_batched, prev_k, prev_v,
                        softmax_scale=self.scale,
                        causal=False,
                    )
                    # Record compute done so next load can safely reuse this slot
                    offload_engine.record_slot_compute_done(slot, self.layer_id)
                    if o_acc is None:
                        o_acc, lse_acc = prev_o, prev_lse
                    else:
                        o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)
            return o_acc, lse_acc

        # N-way pipeline: use ALL available slots for maximum overlap
        # Pipeline depth = num_slots - 1 (num_slots blocks in flight)
        num_slots = len(load_slots)

        # Phase 1: Pre-load up to num_slots blocks to fill the pipeline
        # This starts all transfers in parallel, utilizing full PCIe bandwidth
        num_preload = min(num_slots, num_blocks)
        for i in range(num_preload):
            offload_engine.load_to_slot_layer(load_slots[i], self.layer_id, cpu_block_table[i])

        # Phase 2: Main loop - compute and immediately reuse slot for next transfer
        # Use dedicated compute_stream (not default stream) to enable overlap with transfers
        compute_stream = offload_engine.compute_stream

        for block_idx in range(num_blocks):
            torch.cuda.nvtx.range_push(f"PipelineBlock: L{self.layer_id} B{block_idx}")

            # Cycle through slots: slot[block_idx % num_slots]
            current_slot = load_slots[block_idx % num_slots]
            cpu_block_id = cpu_block_table[block_idx]

            # Wait for current slot's transfer to complete (on compute_stream)
            offload_engine.wait_slot_layer(current_slot, self.layer_id)

            # Compute attention on current slot's data
            # IMPORTANT: Use dedicated compute_stream to avoid implicit sync with default stream
            with torch.cuda.stream(compute_stream):
                # Debug: call hooks on compute_stream (synchronized with transfer)
                if offload_engine.debug_mode:
                    offload_engine._call_debug_hooks(current_slot, self.layer_id, cpu_block_id)

                torch.cuda.nvtx.range_push(f"FlashAttn: L{self.layer_id} PrevBlock{block_idx}")
                prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot, self.layer_id)
                prev_o, prev_lse = flash_attn_with_lse(
                    q_batched, prev_k, prev_v,
                    softmax_scale=self.scale,
                    causal=False,
                )
                torch.cuda.nvtx.range_pop()

                # Record compute done - this allows the next transfer to safely overwrite this slot
                offload_engine.record_slot_compute_done(current_slot, self.layer_id)

            # Immediately start loading the NEXT block into this slot (if more blocks remain)
            # Key insight: reuse current_slot immediately after compute is done!
            next_block_idx = block_idx + num_slots
            if next_block_idx < num_blocks:
                offload_engine.load_to_slot_layer(current_slot, self.layer_id, cpu_block_table[next_block_idx])

            # Merge with accumulated (also on compute_stream for consistency)
            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

            torch.cuda.nvtx.range_pop()  # PipelineBlock

        return o_acc, lse_acc

    def _chunked_decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """
        Compute decode attention with double-buffering using decode_load_slots.

        Decode uses:
        - decode_slot (slot[0]): writes new token's KV
        - decode_load_slots (slots[1:]): load previous chunks from CPU

        Pipeline design:
        - First half of decode_load_slots: 'compute' buffer
        - Second half: 'prefetch' buffer
        - Double-buffer between them for async overlap

        Timeline:
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │Load C0→buf0 │ │Load C1→buf1 │ │Load C2→buf0 │ ...
        └─────────────┘ └─────────────┘ └─────────────┘
                      ↘               ↘               ↘
                       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                       │ Attn(C0)    │ │ Attn(C1)    │ │ Attn(C2)    │
                       └─────────────┘ └─────────────┘ └─────────────┘
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq

        # Get only PREFILLED CPU blocks (exclude the current decode block)
        # The decode block's KV is still in GPU decode_slot, not yet offloaded to CPU
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)
        if self.layer_id == 0:
            logger.debug(f"Decode attention: cpu_block_table={cpu_block_table}, seq.block_table={list(seq.block_table)}")
        if not cpu_block_table:
            raise RuntimeError("Chunked decode attention failed: no prefilled CPU blocks available")

        # Apply sparse policy if enabled
        if kvcache_manager.sparse_policy is not None:
            policy_ctx = PolicyContext(
                query_chunk_idx=0,
                num_query_chunks=1,
                layer_id=self.layer_id,
                query=q_batched,  # Decode provides query for query-aware selection
                is_prefill=False,
                block_size=kvcache_manager.block_size,
                total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
            )
            cpu_block_table = kvcache_manager.sparse_policy.select_blocks(
                cpu_block_table, policy_ctx
            )

        offload_engine = kvcache_manager.offload_engine
        compute_stream = offload_engine.compute_stream

        # Chunk size = capacity of each double buffer region (compute/prefetch)
        # Each region uses half of decode_load_slots
        chunk_size = max(1, len(offload_engine.decode_load_slots) // 2)
        num_chunks = (len(cpu_block_table) + chunk_size - 1) // chunk_size

        # Check if double buffering is possible (need at least 2 separate regions)
        # With only 1 load slot, compute and prefetch regions overlap -> can't double buffer
        can_double_buffer = len(offload_engine.decode_load_slots) >= 2

        o_acc = None
        lse_acc = None

        # Double buffering state: True = use Compute region, False = use Prefetch region
        use_compute = True

        # Pre-load first chunk to Compute region (async)
        first_chunk_ids = cpu_block_table[:min(chunk_size, len(cpu_block_table))]
        offload_engine.load_to_compute_layer(self.layer_id, first_chunk_ids)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(cpu_block_table))
            num_blocks_in_chunk = end - start

            # Wait for current buffer to be ready on compute_stream
            # The load runs on transfer_stream_main, compute runs on compute_stream
            compute_stream.wait_stream(offload_engine.transfer_stream_main)

            # All computation on explicit compute_stream
            with torch.cuda.stream(compute_stream):
                # Get KV from current buffer FIRST, before prefetching overwrites it
                if use_compute:
                    k_chunk, v_chunk = offload_engine.get_kv_for_compute(
                        self.layer_id, num_blocks_in_chunk
                    )
                else:
                    k_chunk, v_chunk = offload_engine.get_kv_for_prefetch(
                        self.layer_id, num_blocks_in_chunk
                    )

                # Compute attention for this chunk
                o_chunk, lse_chunk = flash_attn_with_lse(
                    q_batched, k_chunk, v_chunk,
                    softmax_scale=self.scale,
                    causal=False,
                )

                # Merge with accumulated
                if o_acc is None:
                    o_acc, lse_acc = o_chunk, lse_chunk
                else:
                    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o_chunk, lse_chunk)

            # Trigger async prefetch/load of next chunk to the OTHER buffer
            # This happens AFTER attention completes, so the data is no longer needed
            if chunk_idx + 1 < num_chunks:
                next_start = end
                next_end = min(next_start + chunk_size, len(cpu_block_table))
                next_chunk_ids = cpu_block_table[next_start:next_end]
                if can_double_buffer:
                    if use_compute:
                        # Current in Compute, prefetch next to Prefetch region
                        offload_engine.load_to_prefetch_layer(self.layer_id, next_chunk_ids)
                    else:
                        # Current in Prefetch, prefetch next to Compute region
                        offload_engine.load_to_compute_layer(self.layer_id, next_chunk_ids)
                else:
                    # Sync fallback: load next chunk to same slot (always compute region)
                    offload_engine.load_to_compute_layer(self.layer_id, next_chunk_ids)

            # Swap buffers for next iteration (only matters if can_double_buffer)
            use_compute = not use_compute

        # Now attend to Decode region (contains accumulated decode tokens)
        pos_in_block = context.decode_pos_in_block
        start_pos = context.decode_start_pos_in_block
        num_accumulated = pos_in_block - start_pos + 1

        with torch.cuda.stream(compute_stream):
            if num_accumulated > 0:
                decode_k = offload_engine.k_cache_gpu[self.layer_id, offload_engine.decode_slot, start_pos:pos_in_block+1]
                decode_v = offload_engine.v_cache_gpu[self.layer_id, offload_engine.decode_slot, start_pos:pos_in_block+1]
                decode_k = decode_k.unsqueeze(0)
                decode_v = decode_v.unsqueeze(0)

                decode_o, decode_lse = flash_attn_with_lse(
                    q_batched, decode_k, decode_v,
                    softmax_scale=self.scale,
                    causal=False,
                )

                if o_acc is None:
                    o_acc = decode_o
                else:
                    o_acc, _ = merge_attention_outputs(o_acc, lse_acc, decode_o, decode_lse)

        if o_acc is None:
            raise RuntimeError("Chunked decode attention failed: no KV available")

        # Sync back to default stream before returning
        # Caller expects result to be ready on default stream
        torch.cuda.default_stream().wait_stream(compute_stream)

        return o_acc
