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
        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq if hasattr(context, 'chunked_seq') else None

        if kvcache_manager is not None and seq is not None and self.layer_id >= 0:
            # Get prefilled CPU blocks (blocks already written in previous chunks)
            cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

            if cpu_block_table:
                offload_engine = kvcache_manager.offload_engine
                # For prefill: ONLY use Prefetch region to avoid conflict with
                # current chunk's KV being written to Compute region slots
                # Use synchronous per-layer loading (async would conflict with writes)
                chunk_size = offload_engine.num_prefetch_blocks
                num_chunks = (len(cpu_block_table) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, len(cpu_block_table))
                    num_blocks_in_chunk = end - start
                    chunk_ids = cpu_block_table[start:end]

                    # Load to Prefetch region (per-layer, sync)
                    offload_engine.load_to_prefetch_layer(self.layer_id, chunk_ids)
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
        Compute decode attention with async double-buffering using Compute and Prefetch regions.

        Pipeline design:
        - Compute region: holds current chunk being computed
        - Prefetch region: async loads next chunk while current is computing
        - After computation, swap roles of the two regions

        Timeline:
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │Load C0→Comp │ │Load C1→Pref │ │Load C2→Comp │ ...
        └─────────────┘ └─────────────┘ └─────────────┘
                      ↘               ↘               ↘
                       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                       │ Compute C0  │ │ Compute C1  │ │ Compute C2  │
                       └─────────────┘ └─────────────┘ └─────────────┘
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq

        # Get all CPU blocks for this sequence
        cpu_block_table, _ = kvcache_manager.get_all_cpu_blocks(seq)
        if self.layer_id == 0:
            logger.debug(f"Decode attention: cpu_block_table={cpu_block_table}, seq.block_table={list(seq.block_table)}")
        if not cpu_block_table:
            raise RuntimeError("Chunked decode attention failed: no CPU blocks available")

        offload_engine = kvcache_manager.offload_engine

        # Use prefetch_size as chunk size for double buffering
        # This ensures both Compute and Prefetch regions can hold a full chunk
        chunk_size = offload_engine.num_prefetch_blocks
        num_chunks = (len(cpu_block_table) + chunk_size - 1) // chunk_size

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

            # Wait for current buffer to be ready
            if use_compute:
                offload_engine.wait_compute_layer(self.layer_id)
            else:
                offload_engine.wait_prefetch_layer(self.layer_id)

            # Trigger async prefetch of next chunk to the OTHER buffer
            # This overlaps transfer with current chunk's computation
            if chunk_idx + 1 < num_chunks:
                next_start = end
                next_end = min(next_start + chunk_size, len(cpu_block_table))
                next_chunk_ids = cpu_block_table[next_start:next_end]
                if use_compute:
                    # Current in Compute, prefetch next to Prefetch region
                    offload_engine.load_to_prefetch_layer(self.layer_id, next_chunk_ids)
                else:
                    # Current in Prefetch, prefetch next to Compute region
                    offload_engine.load_to_compute_layer(self.layer_id, next_chunk_ids)

            # Get KV from current buffer
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

            # Swap buffers for next iteration
            use_compute = not use_compute

        # Now attend to Decode region (contains accumulated decode tokens)
        pos_in_block = context.decode_pos_in_block
        start_pos = context.decode_start_pos_in_block
        num_accumulated = pos_in_block - start_pos + 1

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

        return o_acc
