import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


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
        Compute attention with chunked KV from CPU cache.

        For chunked prefill:
        1. Load previous KV from CPU for this layer
        2. Compute attention against previous KV (no causal mask)
        3. Compute attention against current chunk's KV (causal)
        4. Merge results using online softmax
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q, k, v shape: [total_tokens, num_heads, head_dim]
        total_tokens = q.shape[0]

        # Reshape for flash attention: [batch, seq, heads, dim]
        q_batched = q.unsqueeze(0)  # [1, total_tokens, heads, dim]
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)

        accumulated_o = None
        accumulated_lse = None

        # Load previous KV from CPU for this layer
        if context.offload_engine is not None and self.layer_id >= 0:
            # Get the kvcache_manager from context
            kvcache_manager = context.offload_engine

            # For each sequence in the chunk, load previous KV
            # Currently assuming single sequence
            if hasattr(context, 'chunked_seq') and context.chunked_seq is not None:
                prev_k, prev_v = kvcache_manager.load_prev_kv_for_layer(
                    context.chunked_seq,
                    self.layer_id,
                )

                if prev_k is not None and prev_v is not None:
                    # Compute attention against previous KV (no causal mask)
                    prev_o, prev_lse = flash_attn_with_lse(
                        q_batched,
                        prev_k,
                        prev_v,
                        softmax_scale=self.scale,
                        causal=False,  # No causal mask for previous context
                    )
                    accumulated_o = prev_o
                    accumulated_lse = prev_lse

        # Compute attention against current chunk's KV (with causal mask)
        current_o, current_lse = flash_attn_with_lse(
            q_batched,
            k_batched,
            v_batched,
            softmax_scale=self.scale,
            causal=True,  # Causal mask for current chunk
        )

        # Merge with accumulated
        if accumulated_o is None:
            final_o = current_o
        else:
            final_o, _ = merge_attention_outputs(
                accumulated_o, accumulated_lse,
                current_o, current_lse,
            )

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
        Compute decode attention with KV spread across CPU and GPU.

        For decode with chunked KV:
        1. Load all KV for this layer from CPU+GPU
        2. Compute attention (1 query token vs all KV)
        3. Return output
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse

        # q shape: [batch_size, num_heads, head_dim] (single decode token per sequence)
        # We need to attend to ALL previous tokens

        # Load all KV for this layer
        if context.offload_engine is not None and self.layer_id >= 0:
            kvcache_manager = context.offload_engine

            if hasattr(context, 'chunked_seq') and context.chunked_seq is not None:
                # Load all KV from both GPU and CPU for this layer
                k_all, v_all = kvcache_manager.load_all_kv_for_layer(
                    context.chunked_seq,
                    self.layer_id,
                )

                if k_all is not None and v_all is not None:
                    # q shape: [batch_size, num_heads, head_dim]
                    # Need: [batch, seqlen, heads, dim]
                    # Insert seqlen dimension at position 1
                    q_batched = q.unsqueeze(1)  # [batch, 1, heads, dim]

                    # k_all, v_all shape: [1, total_kv_tokens, kv_heads, head_dim]
                    # Compute attention (no causal mask for decode - we want all KV)
                    out, _ = flash_attn_with_lse(
                        q_batched,
                        k_all,
                        v_all,
                        softmax_scale=self.scale,
                        causal=False,  # No causal mask for decode
                    )

                    # Output shape: [batch, 1, heads, dim] -> [batch, heads, dim]
                    return out.squeeze(1)

        # Fallback: shouldn't reach here
        raise RuntimeError("Chunked decode attention failed: no KV available")
