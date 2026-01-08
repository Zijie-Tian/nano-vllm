import torch
from torch import nn

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


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
    k_cache_flat.index_copy_(0, valid_slots.long(), valid_keys_flat)
    v_cache_flat.index_copy_(0, valid_slots.long(), valid_values_flat)


class Attention(nn.Module):
    """
    Attention layer for GPU-only mode.

    For CPU offload mode, attention is computed directly in model_runner's
    run_layerwise_offload_prefill/decode methods using FlashAttention.
    """

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

        # Store KV to cache (for GPU-only mode)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            elif context.sparse_prefill_policy is not None:
                # Sparse prefill (GPU-only) - delegate to policy
                o = context.sparse_prefill_policy.sparse_prefill_attention(
                    q, k, v, self.layer_id
                )
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
