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
        Compute attention with 三区域 GPU buffer for chunked prefill.

        For chunked prefill:
        1. Load previous KV from CPU using Compute/Prefetch区 (if any previous chunks)
        2. Compute attention against previous KV chunks (no causal mask)
        3. Compute attention against current chunk's KV (causal)
        4. Merge all results using online softmax

        三区域设计保证：当前chunk的KV在Compute区，previous KV从CPU加载到Prefetch区，
        不会发生写入和加载区域重叠的问题。
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        # q, k, v shape: [total_tokens, num_heads, head_dim]
        # Reshape for flash attention: [batch, seq, heads, dim]
        q_batched = q.unsqueeze(0)  # [1, total_tokens, heads, dim]
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)

        o_acc = None
        lse_acc = None

        # Load previous KV from CPU using Compute/Prefetch区
        # Note: context.offload_engine is actually HybridKVCacheManager
        kvcache_manager = context.offload_engine
        seq = context.chunked_seq if hasattr(context, 'chunked_seq') else None

        if kvcache_manager is not None and seq is not None and self.layer_id >= 0:
            # Get prefilled CPU blocks (blocks already written in previous chunks)
            cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

            if cpu_block_table:
                offload_engine = kvcache_manager.offload_engine
                # 使用 Prefetch区 来加载 previous KV（不会与当前 Compute区 冲突）
                prefetch_size = offload_engine.num_prefetch_blocks
                num_chunks = (len(cpu_block_table) + prefetch_size - 1) // prefetch_size
                use_compute = True  # 交替使用 Compute区 和 Prefetch区

                # 首先将 previous KV 加载到 Prefetch区
                # Only layer 0 triggers the load (loads ALL layers at once)
                first_chunk_end = min(prefetch_size, len(cpu_block_table))
                first_chunk_ids = cpu_block_table[:first_chunk_end]
                if self.layer_id == 0:
                    offload_engine.load_to_prefetch(first_chunk_ids)

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * prefetch_size
                    end = min(start + prefetch_size, len(cpu_block_table))
                    num_blocks_in_chunk = end - start

                    # Prefetch next chunk to other buffer (if exists)
                    # Only layer 0 triggers the load
                    if chunk_idx + 1 < num_chunks and self.layer_id == 0:
                        next_start = end
                        next_end = min(next_start + prefetch_size, len(cpu_block_table))
                        next_chunk_ids = cpu_block_table[next_start:next_end]
                        if use_compute:
                            # 当前在 Prefetch区，下一个加载到 Compute区（如果有空间）
                            # 注意：Compute区 此时已写入当前chunk的KV，不能覆盖
                            # 所以这里我们使用简单的同步策略：等待当前完成后再加载
                            pass  # 简化版本：不进行双缓冲，只用 Prefetch区
                        else:
                            offload_engine.load_to_prefetch(next_chunk_ids)

                    # Wait for Prefetch区 and get KV
                    offload_engine.wait_prefetch()
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

                    # Load next chunk to Prefetch区 (if exists)
                    if chunk_idx + 1 < num_chunks and self.layer_id == 0:
                        next_start = end
                        next_end = min(next_start + prefetch_size, len(cpu_block_table))
                        next_chunk_ids = cpu_block_table[next_start:next_end]
                        offload_engine.load_to_prefetch(next_chunk_ids)

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
        Compute decode attention with 三区域 GPU buffer.

        All KV is stored on CPU. Uses Compute区 buffer on GPU:
        1. Load chunk to Compute区
        2. Compute attention
        3. Repeat for all chunks
        4. Finally, attend to Decode区 (slot 0) which contains the new token's KV
        5. Merge all attention outputs using online softmax (LSE)

        关键：新token的KV在Decode区(slot 0)，不会被Compute区的加载覆盖。
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

        # Get the actual offload_engine for 三区域 operations
        offload_engine = kvcache_manager.offload_engine

        # Calculate chunk info using Compute区
        compute_size = offload_engine.num_compute_blocks
        num_chunks = (len(cpu_block_table) + compute_size - 1) // compute_size

        o_acc = None
        lse_acc = None

        for chunk_idx in range(num_chunks):
            start = chunk_idx * compute_size
            end = min(start + compute_size, len(cpu_block_table))
            num_blocks_in_chunk = end - start
            chunk_ids = cpu_block_table[start:end]

            # Load this chunk to Compute区
            # Only layer 0 triggers the load (loads ALL layers at once)
            if self.layer_id == 0:
                offload_engine.load_to_compute(chunk_ids)

            # Wait for Compute区 to be ready and get KV
            offload_engine.wait_compute()
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

        # Now attend to Decode区 (contains the new token's KV)
        # This is the token being decoded - only 1 token at position pos_in_block
        pos_in_block = context.decode_pos_in_block
        decode_k, decode_v = offload_engine.get_kv_for_decode_slot(self.layer_id, pos_in_block)
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
