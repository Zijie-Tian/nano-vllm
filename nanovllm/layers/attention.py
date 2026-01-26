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
            chunk_idx = context.current_chunk_idx if hasattr(context, 'current_chunk_idx') else -1

            # Wait for default stream to ensure slot_mapping tensor transfer is complete
            compute_stream.wait_stream(torch.cuda.default_stream())

            with torch.cuda.stream(compute_stream):
                # Write KV to per-layer prefill buffer via offload_engine
                # k, v shape: [num_tokens, kv_heads, head_dim]
                #! GPU 2 GPU
                offload_engine.write_to_prefill_buffer(self.layer_id, k, v, chunk_idx=chunk_idx)
        elif is_chunked_offload:
            # Chunked decode mode: write KV to per-layer decode buffer via offload_engine
            # KV will be written to decode buffer in the decode branch below
            # No store_kvcache needed - all KV management goes through offload_engine
            pass
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
                offload_engine.write_to_decode_buffer(self.layer_id, pos_in_block, k.squeeze(0), v.squeeze(0))
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

        Simplified design:
        - All computation logic is delegated to sparse_policy.compute_chunked_prefill()
        - This method only handles async offload after computation

        The policy handles:
        1. Loading historical blocks from CPU
        2. Computing attention against historical KV (no causal mask)
        3. Computing attention against current KV from prefill buffer (causal)
        4. Merging all results
        """
        current_chunk_idx = context.current_chunk_idx
        torch.cuda.nvtx.range_push(f"ChunkedPrefill: L{self.layer_id} Chunk{current_chunk_idx}")

        num_tokens = k.shape[0]

        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq if hasattr(context, 'chunked_seq') else None
        offload_engine = kvcache_manager.offload_engine if kvcache_manager is not None else None

        # Get sparse policy - required for chunked prefill
        sparse_policy = kvcache_manager.sparse_policy
        if sparse_policy is None:
            raise RuntimeError("sparse_policy is required for chunked prefill")

        # Step 1: Get historical CPU blocks
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

        # Step 2: Apply select_blocks to filter blocks (before calling compute_chunked_prefill)
        selected_blocks = []
        if cpu_block_table:
            num_chunks = current_chunk_idx + 1
            policy_ctx = PolicyContext(
                query_chunk_idx=current_chunk_idx,
                num_query_chunks=num_chunks,
                layer_id=self.layer_id,
                query=q,  # Pass query for sparse policies that need it
                is_prefill=True,
                block_size=kvcache_manager.block_size,
                total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
            )
            selected_blocks = sparse_policy.select_blocks(cpu_block_table, offload_engine, policy_ctx)
            logger.debug(f"[DEBUG] select_blocks: {len(cpu_block_table)} -> {len(selected_blocks)} blocks")

        # [DEBUG] Verify execution path
        logger.debug(f"[DEBUG] Calling sparse_policy.compute_chunked_prefill, "
                     f"policy={sparse_policy}, layer={self.layer_id}, chunk={current_chunk_idx}")

        # Delegate computation to policy with pre-selected blocks
        final_o = sparse_policy.compute_chunked_prefill(
            q, k, v,
            self.layer_id,
            self.scale,
            offload_engine,
            kvcache_manager,
            current_chunk_idx,
            seq,
            num_tokens,
            selected_blocks,
        )

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

        return final_o

    def _chunked_decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """
        Compute decode attention by delegating to sparse policy.

        Simplified design:
        - All computation logic is delegated to sparse_policy.compute_chunked_decode()
        - This method only validates the policy and delegates

        The policy handles:
        1. Loading prefilled blocks from CPU via pipeline
        2. Computing attention against prefilled KV
        3. Reading accumulated decode tokens from decode buffer
        4. Merging all results
        """
        kvcache_manager = context.kvcache_manager
        seq = context.chunked_seq
        offload_engine = kvcache_manager.offload_engine

        # Get sparse policy - required for chunked decode
        sparse_policy = kvcache_manager.sparse_policy
        if sparse_policy is None:
            raise RuntimeError("sparse_policy is required for chunked decode")

        # Check if policy supports decode phase
        # If not, fallback to FullAttentionPolicy (e.g., XAttentionBSAPolicy only supports prefill)
        if not sparse_policy.supports_decode:
            from nanovllm.kvcache.sparse import FullAttentionPolicy
            sparse_policy = FullAttentionPolicy()
            logger.debug(f"[DEBUG] {kvcache_manager.sparse_policy} doesn't support decode, "
                         f"falling back to FullAttentionPolicy")

        # Step 1: Get prefilled CPU blocks
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

        # Step 2: Apply select_blocks to filter blocks (before calling compute_chunked_decode)
        selected_blocks = []
        if cpu_block_table:
            policy_ctx = PolicyContext(
                query_chunk_idx=0,
                num_query_chunks=1,
                layer_id=self.layer_id,
                query=q,  # Pass query for sparse policies that need it
                is_prefill=False,
                block_size=kvcache_manager.block_size,
                total_kv_len=len(cpu_block_table) * kvcache_manager.block_size,
            )
            selected_blocks = sparse_policy.select_blocks(cpu_block_table, offload_engine, policy_ctx)
            logger.debug(f"[DEBUG] decode select_blocks: {len(cpu_block_table)} -> {len(selected_blocks)} blocks")

        # [DEBUG] Verify execution path
        logger.debug(f"[DEBUG] Calling sparse_policy.compute_chunked_decode, "
                     f"policy={sparse_policy}, layer={self.layer_id}")

        # Delegate computation to policy with pre-selected blocks
        return sparse_policy.compute_chunked_decode(
            q,
            self.layer_id,
            self.scale,
            offload_engine,
            kvcache_manager,
            seq,
            selected_blocks,
        )
