import logging
import torch
import torch.cuda.nvtx
from torch import nn

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.kvcache.sparse.policy import PolicyContext

logger = logging.getLogger(__name__)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Store key/value tensors into KV cache using slot mapping.

    This is a pure PyTorch implementation replacing the previous Triton kernel.
    Uses index_copy_ for efficient in-place scatter operation.

    Args:
        key: [N, num_kv_heads, head_dim]
        value: [N, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] or similar
        v_cache: same shape as k_cache
        slot_mapping: [N] with values as flat indices, -1 means skip
    """
    is_capturing = torch.cuda.is_current_stream_capturing()

    if is_capturing:
        # During CUDA graph capture, assume all slots are valid.
        # CUDA graphs don't support data-dependent operations like boolean indexing.
        # This is safe because decode (captured) always has valid slots.
        valid_slots = slot_mapping
        valid_keys = key
        valid_values = value
    else:
        # Normal execution: filter out invalid slots (slot == -1)
        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return
        valid_slots = slot_mapping[valid_mask]
        valid_keys = key[valid_mask]  # [M, num_kv_heads, head_dim]
        valid_values = value[valid_mask]

    # Flatten cache and KV for scatter operation
    # Cache is viewed as [total_slots, D] where D = num_kv_heads * head_dim
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    total_slots = k_cache.numel() // D

    k_cache_flat = k_cache.view(total_slots, D)
    v_cache_flat = v_cache.view(total_slots, D)
    valid_keys_flat = valid_keys.reshape(-1, D)
    valid_values_flat = valid_values.reshape(-1, D)

    # In-place scatter using index_copy_
    # 即使 valid_slots 为空张量，index_copy_ 也是安全的（不会修改数据）。
    k_cache_flat.index_copy_(0, valid_slots.long(), valid_keys_flat)
    v_cache_flat.index_copy_(0, valid_slots.long(), valid_values_flat)


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

        # Determine if we're in chunked offload mode
        is_chunked_offload = (
            context.is_chunked_prefill and
            hasattr(context, 'kvcache_manager') and
            context.kvcache_manager is not None and
            hasattr(context.kvcache_manager, 'offload_engine')
        )
        
        #! Ensure synchronization before accessing k_cache/v_cache
        # torch.cuda.synchronize()
        #! =======================================================

        if is_chunked_offload and context.is_prefill:
            # Chunked prefill mode: write KV to per-layer prefill buffer (not GPU slot)
            # This enables fully async offloads since each layer has its own buffer.
            offload_engine = context.kvcache_manager.offload_engine
            compute_stream = offload_engine.compute_stream

            # Wait for default stream to ensure slot_mapping tensor transfer is complete
            compute_stream.wait_stream(torch.cuda.default_stream())

            with torch.cuda.stream(compute_stream):
                # Write KV to per-layer prefill buffer (contiguous write, no slot_mapping)
                # k, v shape: [num_tokens, kv_heads, head_dim]
                num_tokens = k.shape[0]
                offload_engine.prefill_k_buffer[self.layer_id, :num_tokens].copy_(k)
                offload_engine.prefill_v_buffer[self.layer_id, :num_tokens].copy_(v)
        elif is_chunked_offload:
            # Chunked decode mode: use compute_stream for store_kvcache
            # This ensures proper synchronization with per-layer offload
            compute_stream = context.kvcache_manager.offload_engine.compute_stream
            if k_cache.numel() and v_cache.numel():
                # CRITICAL: Wait for default stream to ensure slot_mapping tensor transfer is complete
                # slot_mapping is created with non_blocking=True on default stream, but we use it
                # on compute_stream. Without this sync, index_copy_ can get corrupted indices.
                compute_stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(compute_stream):
                    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        else:
            # Normal mode: store on default stream
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
                # Store current decode token to per-layer decode buffer
                # This is needed because GPU cache has no layer dimension,
                # so all layers would overwrite each other in decode_slot.
                kvcache_manager = context.kvcache_manager
                offload_engine = kvcache_manager.offload_engine
                pos_in_block = context.decode_pos_in_block
                # k, v shape: [1, kv_heads, head_dim]
                offload_engine.decode_k_buffer[self.layer_id, pos_in_block].copy_(k.squeeze(0))
                offload_engine.decode_v_buffer[self.layer_id, pos_in_block].copy_(v.squeeze(0))
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
        Compute attention with per-layer prefill buffer for async offload.

        Optimized design:
        - Current chunk's KV is written to per-layer prefill buffer (not GPU slot)
        - Previous chunks' KV are loaded from CPU using GPU slots
        - Each layer offloads from its own buffer - no waiting required!

        For each layer:
        1. Current chunk's KV is in prefill_buffer[layer_id] (just written by model)
        2. Load previous chunks from CPU using available slots (pipeline)
        3. Compute attention against previous KV (no causal mask)
        4. Compute attention against current KV from prefill buffer (causal)
        5. Merge all results using online softmax
        6. Async offload prefill buffer to CPU (no waiting!)
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        current_chunk_idx = context.current_chunk_idx
        torch.cuda.nvtx.range_push(f"ChunkedPrefill: L{self.layer_id} Chunk{current_chunk_idx}")

        # q shape: [total_tokens, num_heads, head_dim]
        q_batched = q.unsqueeze(0)  # [1, total_tokens, heads, dim]
        num_tokens = k.shape[0]

        o_acc = None
        lse_acc = None

        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq if hasattr(context, 'chunked_seq') else None
        offload_engine = kvcache_manager.offload_engine if kvcache_manager is not None else None

        if kvcache_manager is not None and seq is not None and self.layer_id >= 0:
            # Get prefilled CPU blocks (blocks from previous chunks)
            cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

            # Apply sparse policy if enabled
            sparse_policy = kvcache_manager.sparse_policy

            # === XAttention BSA: Policy handles entire sparse prefill ===
            # Check if policy has sparse_prefill_attention method (XAttention BSA)
            if (sparse_policy is not None and
                hasattr(sparse_policy, 'sparse_prefill_attention') and
                getattr(sparse_policy, 'supports_prefill', False)):
                # Use policy's sparse_prefill_attention method
                # Pass softmax_scale from attention layer
                # IMPORTANT: Don't return early - we still need to do KV offload below!
                o = sparse_policy.sparse_prefill_attention(q, k, v, self.layer_id, self.scale)
                # Convert back to batched format for consistency with standard flow
                o_acc = o.unsqueeze(0)  # [seq_len, heads, dim] -> [1, seq_len, heads, dim]
                lse_acc = None  # sparse_prefill_attention returns final output, not intermediate LSE
                # Skip standard flow processing since we already computed attention
                cpu_block_table = None  # Signal to skip historical chunk processing

            # === Standard sparse policy (Quest, etc.) ===
            if cpu_block_table and sparse_policy is not None:
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
                cpu_block_table = sparse_policy.select_blocks(
                    cpu_block_table, policy_ctx
                )

            if cpu_block_table:
                # Get available load slots (all slots can be used since we use prefill buffer)
                load_slots = list(range(offload_engine.num_ring_slots))
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

        # Get compute stream for all attention operations
        compute_stream = offload_engine.compute_stream if offload_engine is not None else None

        # Compute attention against current chunk's KV from prefill buffer (with causal mask)
        # Skip this if XAttention BSA already computed full attention (o_acc is set, lse_acc is None)
        needs_current_chunk_attention = (lse_acc is not None or o_acc is None)

        if needs_current_chunk_attention:
            if compute_stream is not None:
                with torch.cuda.stream(compute_stream):
                    torch.cuda.nvtx.range_push(f"FlashAttn: L{self.layer_id} CurrentChunk (causal)")
                    # Get KV from per-layer prefill buffer
                    k_batched, v_batched = offload_engine.get_prefill_buffer_slice(self.layer_id, num_tokens)
                    current_o, current_lse = flash_attn_with_lse(
                        q_batched,
                        k_batched,
                        v_batched,
                        softmax_scale=self.scale,
                        causal=True,
                    )
                    torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push(f"FlashAttn: L{self.layer_id} CurrentChunk (causal)")
                k_batched = k.unsqueeze(0)
                v_batched = v.unsqueeze(0)
                current_o, current_lse = flash_attn_with_lse(
                    q_batched,
                    k_batched,
                    v_batched,
                    softmax_scale=self.scale,
                    causal=True,
                )
                torch.cuda.nvtx.range_pop()

        # Merge with accumulated (all on compute_stream for consistency)
        if o_acc is None:
            # No accumulated attention (standard flow or XAttention BSA with no historical chunks)
            final_o = current_o if needs_current_chunk_attention else o_acc
        else:
            # Has accumulated attention (XAttention BSA with historical chunks)
            if needs_current_chunk_attention:
                # Need to merge historical (from XAttention BSA) with current chunk
                if compute_stream is not None:
                    with torch.cuda.stream(compute_stream):
                        torch.cuda.nvtx.range_push(f"MergeAttn: L{self.layer_id}")
                        final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)
                        torch.cuda.nvtx.range_pop()
                else:
                    torch.cuda.nvtx.range_push(f"MergeAttn: L{self.layer_id}")
                    final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)
                    torch.cuda.nvtx.range_pop()
            else:
                # XAttention BSA already computed everything
                final_o = o_acc

        torch.cuda.nvtx.range_pop()  # ChunkedPrefill

        # Per-layer ASYNC offload: offload prefill buffer to CPU
        # No waiting required! Each layer has its own buffer and stream.
        if offload_engine is not None and seq is not None:
            cpu_block_ids, _ = kvcache_manager.get_all_cpu_blocks(seq)
            if current_chunk_idx < len(cpu_block_ids):
                cpu_block_id = cpu_block_ids[current_chunk_idx]
                # Async offload - no waiting, fully parallel across layers
                offload_engine.offload_prefill_buffer_async(
                    self.layer_id, cpu_block_id, num_tokens
                )

        # Sync default stream with compute_stream before returning
        # This ensures the result is ready for the rest of the model (layernorm, MLP)
        if compute_stream is not None:
            torch.cuda.default_stream().wait_stream(compute_stream)

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
            offload_engine.wait_slot_layer(0)

            # IMPORTANT: Must use compute_stream to match wait_slot_layer
            with torch.cuda.stream(compute_stream):
                prev_k, prev_v = offload_engine.get_kv_for_slot(0)

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
                offload_engine.wait_slot_layer(slot)

                with torch.cuda.stream(compute_stream):
                    # Debug: call hooks on compute_stream (synchronized with transfer)
                    if offload_engine.debug_mode:
                        offload_engine._call_debug_hooks(slot, self.layer_id, cpu_block_id)

                    prev_k, prev_v = offload_engine.get_kv_for_slot(slot)

                    prev_o, prev_lse = flash_attn_with_lse(
                        q_batched, prev_k, prev_v,
                        softmax_scale=self.scale,
                        causal=False,
                    )
                    # Record compute done so next load can safely reuse this slot
                    offload_engine.record_slot_compute_done(slot)
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
            offload_engine.wait_slot_layer(current_slot)

            # Compute attention on current slot's data
            # IMPORTANT: Use dedicated compute_stream to avoid implicit sync with default stream
            with torch.cuda.stream(compute_stream):
                # Debug: call hooks on compute_stream (synchronized with transfer)
                if offload_engine.debug_mode:
                    offload_engine._call_debug_hooks(current_slot, self.layer_id, cpu_block_id)

                torch.cuda.nvtx.range_push(f"FlashAttn: L{self.layer_id} PrevBlock{block_idx}")
                prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)

                prev_o, prev_lse = flash_attn_with_lse(
                    q_batched, prev_k, prev_v,
                    softmax_scale=self.scale,
                    causal=False,
                )
                torch.cuda.nvtx.range_pop()

                # Record compute done - this allows the next transfer to safely overwrite this slot
                offload_engine.record_slot_compute_done(current_slot)

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
        Compute decode attention using cross-layer pipeline.

        Optimization: Uses double-buffered layer cache to overlap H2D transfer
        with computation across layers:
        - Layer N computes while Layer N+1's data is being loaded
        - Each layer only waits for its own data, not all layers' data

        This reduces effective latency from O(num_layers * transfer_time) to
        O(transfer_time + num_layers * compute_time) when transfer < compute.
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq

        # Get only PREFILLED CPU blocks (exclude the current decode block)
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)
        if self.layer_id == 0:
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

        # Apply sparse policy if enabled (Quest does Top-K selection for decode)
        sparse_policy = kvcache_manager.sparse_policy
        if sparse_policy is not None:
            policy_ctx = PolicyContext(
                query_chunk_idx=0,
                num_query_chunks=1,
                layer_id=self.layer_id,
                query=q_batched,
                is_prefill=False,
                block_size=kvcache_manager.block_size,
                total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
            )
            cpu_block_table = sparse_policy.select_blocks(
                cpu_block_table, policy_ctx
            )

        offload_engine = kvcache_manager.offload_engine

        # Use cross-layer pipeline if active (initialized in model_runner)
        if offload_engine.is_pipeline_active():
            o_acc, lse_acc = self._decode_with_layer_pipeline(
                q_batched, cpu_block_table, offload_engine,
                block_size, last_block_valid_tokens
            )
        else:
            # Fallback to original ring buffer pipeline
            load_slots = offload_engine.decode_load_slots
            o_acc, lse_acc = self._decode_ring_buffer_pipeline(
                q_batched, cpu_block_table, load_slots, offload_engine,
                block_size, last_block_valid_tokens
            )

        # Now attend to accumulated decode tokens from per-layer decode buffer
        pos_in_block = context.decode_pos_in_block
        start_pos = context.decode_start_pos_in_block
        num_accumulated = pos_in_block - start_pos + 1

        # Sync compute_stream with default stream before reading decode_buffer
        compute_stream = offload_engine.compute_stream
        compute_stream.wait_stream(torch.cuda.default_stream())

        with torch.cuda.stream(compute_stream):
            if num_accumulated > 0:
                # Read from per-layer decode buffer
                decode_k = offload_engine.decode_k_buffer[self.layer_id, start_pos:pos_in_block+1]
                decode_v = offload_engine.decode_v_buffer[self.layer_id, start_pos:pos_in_block+1]
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
        torch.cuda.default_stream().wait_stream(compute_stream)

        return o_acc

    def _decode_ring_buffer_pipeline(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: list,
        load_slots: list,
        offload_engine,
        block_size: int,
        last_block_valid_tokens: int,
    ):
        """
        Ring buffer pipeline for decode prefill loading (same mechanism as prefill).

        Loads one block at a time, computes attention, and merges results.
        Uses the same load_to_slot_layer / wait_slot_layer / get_kv_for_slot
        methods as prefill for proven correctness.
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
            offload_engine.load_to_slot_layer(load_slots[i], self.layer_id, cpu_block_table[i])

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
                    softmax_scale=self.scale,
                    causal=False,
                )

                # Record compute done for slot reuse
                offload_engine.record_slot_compute_done(current_slot)

            # Start loading next block (pipeline)
            next_block_idx = block_idx + num_slots
            if next_block_idx < num_blocks:
                offload_engine.load_to_slot_layer(current_slot, self.layer_id, cpu_block_table[next_block_idx])

            # Merge with accumulated
            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

        return o_acc, lse_acc

    def _decode_with_layer_pipeline(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: list,
        offload_engine,
        block_size: int,
        last_block_valid_tokens: int,
    ):
        """
        Decode using cross-layer pipeline for optimized H2D transfer.

        This method uses pre-loaded layer buffers instead of loading
        blocks one by one. The pipeline loads the next layer's data
        while the current layer computes, achieving transfer/compute overlap.

        The key insight is that each layer needs the SAME blocks but from
        different layers of CPU cache. By double-buffering and pipelining
        across layers, we reduce total latency.
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        num_blocks = len(cpu_block_table)
        if num_blocks == 0:
            return None, None

        compute_stream = offload_engine.compute_stream

        # Get KV from pre-loaded layer buffer (triggers next layer loading)
        prev_k, prev_v = offload_engine.get_decode_layer_kv(self.layer_id, num_blocks)

        # prev_k, prev_v shape: [num_blocks, block_size, kv_heads, head_dim]
        # Reshape to [1, num_blocks * block_size, kv_heads, head_dim]
        total_tokens = num_blocks * block_size

        # Handle partial last block
        if last_block_valid_tokens < block_size:
            # Only use valid tokens from last block
            actual_tokens = (num_blocks - 1) * block_size + last_block_valid_tokens
            # Flatten and truncate
            prev_k_flat = prev_k.reshape(-1, prev_k.shape[-2], prev_k.shape[-1])[:actual_tokens]
            prev_v_flat = prev_v.reshape(-1, prev_v.shape[-2], prev_v.shape[-1])[:actual_tokens]
        else:
            prev_k_flat = prev_k.reshape(-1, prev_k.shape[-2], prev_k.shape[-1])
            prev_v_flat = prev_v.reshape(-1, prev_v.shape[-2], prev_v.shape[-1])

        # Add batch dimension: [1, total_tokens, kv_heads, head_dim]
        prev_k_batched = prev_k_flat.unsqueeze(0)
        prev_v_batched = prev_v_flat.unsqueeze(0)

        # Compute attention on all prefilled blocks at once
        with torch.cuda.stream(compute_stream):
            o_acc, lse_acc = flash_attn_with_lse(
                q_batched, prev_k_batched, prev_v_batched,
                softmax_scale=self.scale,
                causal=False,
            )

        return o_acc, lse_acc
