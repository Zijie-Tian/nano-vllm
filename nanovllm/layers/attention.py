import logging
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

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
        Compute attention with three-region GPU buffer for chunked prefill.

        For chunked prefill:
        1. Load previous KV from CPU using Compute/Prefetch region (if any previous chunks)
        2. Compute attention against previous KV chunks (no causal mask)
        3. Compute attention against current chunk's KV (causal)
        4. Merge all results using online softmax

        Three-region design guarantees: current chunk's KV is in Compute region, previous KV is loaded
        from CPU to Prefetch region, so write and load regions never overlap.
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q, k, v shape: [total_tokens, num_heads, head_dim]
        # Reshape for flash attention: [batch, seq, heads, dim]
        q_batched = q.unsqueeze(0)  # [1, total_tokens, heads, dim]
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)

        o_acc = None
        lse_acc = None

        # Load previous KV from CPU using Compute/Prefetch region
        # Note: context.offload_engine is actually HybridKVCacheManager
        kvcache_manager = context.offload_engine
        seq = context.chunked_seq if hasattr(context, 'chunked_seq') else None

        if kvcache_manager is not None and seq is not None and self.layer_id >= 0:
            # Get prefilled CPU blocks (blocks already written in previous chunks)
            cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

            if cpu_block_table:
                offload_engine = kvcache_manager.offload_engine
                # Use Prefetch region to load previous KV (won't conflict with current Compute region)
                prefetch_size = offload_engine.num_prefetch_blocks
                num_chunks = (len(cpu_block_table) + prefetch_size - 1) // prefetch_size

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * prefetch_size
                    end = min(start + prefetch_size, len(cpu_block_table))
                    num_blocks_in_chunk = end - start
                    chunk_ids = cpu_block_table[start:end]

                    # Load this chunk to Prefetch region (per-layer loading)
                    # Each layer loads only its own KV, avoiding the bug where layer 0
                    # loads all layers and overwrites data before other layers can read it
                    offload_engine.load_to_prefetch_layer(self.layer_id, chunk_ids)

                    # Wait for this layer's Prefetch region and get KV
                    offload_engine.wait_prefetch_layer(self.layer_id)
                    prev_k, prev_v = offload_engine.get_kv_for_prefetch(
                        self.layer_id, num_blocks_in_chunk
                    )

                    # Compute attention against this chunk (no causal mask)
                    prev_o, prev_lse = flash_attn_with_lse(
                        q_batched,
                        prev_k,
                        prev_v,
                        softmax_scale=self.scale,
                        causal=False,
                    )

                    # Merge with accumulated
                    if o_acc is None:
                        o_acc, lse_acc = prev_o, prev_lse
                    else:
                        o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

        # Compute attention against current chunk's KV (with causal mask)
        current_o, current_lse = flash_attn_with_lse(
            q_batched,
            k_batched,
            v_batched,
            softmax_scale=self.scale,
            causal=True,
        )

        # Merge with accumulated
        if o_acc is None:
            final_o = current_o
        else:
            final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

        # Remove batch dimension: [1, total_tokens, heads, dim] -> [total_tokens, heads, dim]
        return final_o.squeeze(0)

    def _chunked_decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """
        Compute decode attention with three-region GPU buffer.

        All KV is stored on CPU. Uses Compute region buffer on GPU:
        1. Load chunk to Compute region
        2. Compute attention
        3. Repeat for all chunks
        4. Finally, attend to Decode region (slot 0) which contains the new token's KV
        5. Merge all attention outputs using online softmax (LSE)

        Key: new token's KV is in Decode region (slot 0), won't be overwritten by Compute region loading.
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        # Need: [batch, seqlen, heads, dim]
        q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

        # Note: context.offload_engine is actually HybridKVCacheManager
        kvcache_manager = context.offload_engine
        seq = context.chunked_seq

        # Get all CPU blocks for this sequence
        cpu_block_table, _ = kvcache_manager.get_all_cpu_blocks(seq)
        if self.layer_id == 0:
            logger.debug(f"Decode attention: cpu_block_table={cpu_block_table}, seq.block_table={list(seq.block_table)}")
        if not cpu_block_table:
            raise RuntimeError("Chunked decode attention failed: no CPU blocks available")

        # Get the actual offload_engine for three-region operations
        offload_engine = kvcache_manager.offload_engine

        # Calculate chunk info using Compute region
        compute_size = offload_engine.num_compute_blocks
        num_chunks = (len(cpu_block_table) + compute_size - 1) // compute_size

        o_acc = None
        lse_acc = None

        for chunk_idx in range(num_chunks):
            start = chunk_idx * compute_size
            end = min(start + compute_size, len(cpu_block_table))
            num_blocks_in_chunk = end - start
            chunk_ids = cpu_block_table[start:end]

            # Load this chunk to Compute region (per-layer loading)
            # Each layer loads only its own KV, avoiding the bug where layer 0
            # loads all layers and overwrites data before other layers can read it
            offload_engine.load_to_compute_layer(self.layer_id, chunk_ids)

            # Wait for this layer's Compute region to be ready and get KV
            offload_engine.wait_compute_layer(self.layer_id)
            k_chunk, v_chunk = offload_engine.get_kv_for_compute(
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

        # Now attend to Decode region (contains accumulated decode tokens)
        # When batching offloads, decode slot accumulates multiple tokens
        # from decode_start_pos_in_block to decode_pos_in_block (inclusive)
        pos_in_block = context.decode_pos_in_block
        start_pos = context.decode_start_pos_in_block
        num_accumulated = pos_in_block - start_pos + 1

        if num_accumulated > 0:
            # Get accumulated KV in decode slot [start_pos : pos_in_block+1]
            decode_k = offload_engine.k_cache_gpu[self.layer_id, offload_engine.decode_slot, start_pos:pos_in_block+1]
            decode_v = offload_engine.v_cache_gpu[self.layer_id, offload_engine.decode_slot, start_pos:pos_in_block+1]
            decode_k = decode_k.unsqueeze(0)  # [1, num_tokens, heads, dim]
            decode_v = decode_v.unsqueeze(0)

            decode_o, decode_lse = flash_attn_with_lse(
                q_batched, decode_k, decode_v,
                softmax_scale=self.scale,
                causal=False,
            )

            # Merge with accumulated
            if o_acc is None:
                o_acc = decode_o
            else:
                o_acc, _ = merge_attention_outputs(o_acc, lse_acc, decode_o, decode_lse)

        if o_acc is None:
            raise RuntimeError("Chunked decode attention failed: no KV available")

        # Output shape: [batch, 1, heads, dim] (same as normal decode)
        return o_acc
