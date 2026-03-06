"""
BLASST sparse attention policy for chunked prefill.

BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

Key insight: Use online softmax running max to dynamically skip unimportant blocks.
Skip condition: local_max - running_max < ln(λ)
Threshold formula: λ = a / L (inverse with sequence length)

Implementation notes:
- IO communication: Same as FullAttention (load all blocks)
- Compute decision: Skip merging sub-blocks based on BLASST condition
- Fine-grained: Both query and KV processed at configurable granularity (default 128)
- Running max: Extracted from LSE (log-sum-exp) of flash attention
"""

import logging
import math
import torch
from typing import List, TYPE_CHECKING

from nanovllm.ops.blasst_mask import blasst_mask_forward
from nanovllm.ops.blasst_attn import blasst_attn_forward

from .policy import SparsePolicy, PolicyContext

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class BLASSTPolicy(SparsePolicy):
    """
    BLASST sparse attention policy with fine-grained granularity.

    Implementation:
    - select_blocks: Returns all available blocks (load all, decide later)
    - compute_chunked_prefill: Dynamic skip based on running max threshold
      at configurable sub-block granularity (default 128 tokens)

    Threshold configuration:
    - a: Inverse formula parameter (λ = a / L), default 16384
    - fixed_lambda: If set, use fixed threshold instead of formula
    - granularity: Token granularity for skip decisions (default 128)

    Fine-grained processing:
    - Query chunk is divided into sub-chunks of 'granularity' tokens
    - Each KV block is divided into sub-blocks of 'granularity' tokens
    - BLASST skip condition applied at (q_sub, kv_sub) pair level
    - This allows more precise sparsity compared to block-level skipping
    """

    # BLASST supports both prefill and decode
    supports_prefill = True
    supports_decode = True

    def __init__(self, a: int = 16384, fixed_lambda: float = None, granularity: int = 128):
        """
        Initialize BLASST policy.

        Args:
            a: Inverse formula numerator (λ = a / L). Default 16384 gives λ=0.5 at 32K.
            fixed_lambda: If set, use fixed threshold instead of inverse formula.
            granularity: Token granularity for BLASST skip decisions (default 128).
                        Both query and KV are processed at this granularity.
        """
        self.a = a
        self.fixed_lambda = fixed_lambda
        self.granularity = granularity
        self._stats_num_chunks = 0
        self._stats_skipped_subblocks = 0
        self._stats_total_subblocks = 0

    def _get_lambda(self, seq_len: int) -> float:
        """
        Compute threshold λ based on sequence length.

        Formula: λ = a / L (inverse relationship)

        Args:
            seq_len: Current sequence length

        Returns:
            Threshold value λ
        """
        if self.fixed_lambda is not None:
            return self.fixed_lambda
        # Inverse formula: longer sequences → smaller threshold
        return self.a / max(seq_len, 1)

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select blocks - currently returns all blocks (FullAttention behavior).

        TODO: Implement BLASST block selection logic.
        For now, this just tests the chunked prefill flow with full attention.
        """
        # For now, return all blocks to test basic chunked prefill flow
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1
            logger.debug(f"[BLASST] chunk={ctx.query_chunk_idx}, "
                        f"available={len(available_blocks)}, selected={len(available_blocks)}")
        return available_blocks

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats_num_chunks = 0
        self._stats_skipped_subblocks = 0
        self._stats_total_subblocks = 0

    def get_stats(self) -> dict:
        """Get statistics."""
        skip_rate = 0.0
        if self._stats_total_subblocks > 0:
            skip_rate = self._stats_skipped_subblocks / self._stats_total_subblocks
        return {
            "num_chunks": self._stats_num_chunks,
            "skipped_subblocks": self._stats_skipped_subblocks,
            "total_subblocks": self._stats_total_subblocks,
            "skip_rate": skip_rate,
        }

    # ========================================================================
    # GPU-only methods (non-chunked) - Not supported for now
    # ========================================================================

    def compute_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        layer_id: int,
        block_tables=None,
    ) -> torch.Tensor:
        """GPU-only prefill - not supported, use FullAttentionPolicy instead."""
        raise NotImplementedError(
            "BLASST policy only supports chunked prefill mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables=None,
    ) -> torch.Tensor:
        """GPU-only decode - not supported, use FullAttentionPolicy instead."""
        raise NotImplementedError(
            "BLASST policy only supports chunked decode mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    # ========================================================================
    # Chunked offload methods
    # ========================================================================

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
        selected_blocks: List[int],
    ) -> torch.Tensor:
        """
        Compute attention for chunked prefill with BLASST dynamic skipping.

        BLASST Algorithm (Fine-grained version):
        1. Divide query chunk into sub-chunks (default 128 tokens)
        2. Load all KV blocks (same as FullAttention)
        3. Divide each KV block into sub-blocks (default 128 tokens)
        4. For each (query_sub, kv_sub) pair:
           - Compute attention and extract local_max from LSE
           - Apply skip condition: local_max - running_max < ln(λ)
           - Skip merging if condition met
        5. Merge results from all query sub-chunks

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
            selected_blocks: List of CPU block IDs to load

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        # Use FlashInfer-based implementations
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        # Calculate sequence length for threshold computation
        total_seq_len = len(seq) if seq else num_tokens
        lambda_val = self._get_lambda(total_seq_len)
        ln_lambda = math.log(lambda_val)

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        q_len = q.shape[0]
        granularity = self.granularity
        compute_stream = offload_engine.compute_stream

        # Divide query into sub-chunks
        num_q_subchunks = (q_len + granularity - 1) // granularity

        # Prepare accumulators for each query sub-chunk
        q_sub_outputs = []  # List of (o_acc, lse_acc) for each sub-chunk
        q_sub_running_max = []  # List of running_max tensors [num_heads, sub_len]
        num_heads = q.shape[1]

        for q_sub_idx in range(num_q_subchunks):
            q_start = q_sub_idx * granularity
            q_end = min((q_sub_idx + 1) * granularity, q_len)
            sub_len = q_end - q_start

            # Initialize accumulator and running_max for this sub-chunk
            q_sub_outputs.append((None, None))  # (o_acc, lse_acc)
            q_sub_running_max.append(torch.full(
                (num_heads, sub_len),
                float('-inf'),
                device=q.device,
                dtype=torch.float32
            ))

        # Track skip statistics
        skipped_subblocks = 0
        total_subblocks = 0

        # Get block size from offload engine
        block_size = offload_engine.block_size
        num_kv_subblocks = (block_size + granularity - 1) // granularity

        # Load and compute attention on historical blocks with BLASST skipping
        cpu_block_table = selected_blocks

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            # Log at layer 0 for debugging
            if layer_id == 0:
                logger.info(f"[BLASST] Chunk {current_chunk_idx}: "
                           f"seq_len={total_seq_len}, λ={lambda_val:.4f}, "
                           f"blocks={num_blocks}, granularity={granularity}")

            # Collect all KV blocks using ring buffer pipeline
            k_blocks = []
            v_blocks = []

            if len(load_slots) == 1:
                # Only 1 slot - use synchronous mode
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id)
                    offload_engine.wait_slot_layer(slot)

                    prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
                    # Reshape: [1, block_size, num_kv_heads, head_dim] -> [num_kv_heads, block_size, head_dim]
                    k_block = prev_k.squeeze(0).permute(1, 0, 2).contiguous()
                    v_block = prev_v.squeeze(0).permute(1, 0, 2).contiguous()
                    k_blocks.append(k_block)
                    v_blocks.append(v_block)
                    offload_engine.record_slot_compute_done(slot)
            else:
                # Multiple slots - use pipeline
                num_slots = len(load_slots)
                num_preload = min(num_slots, num_blocks)
                for i in range(num_preload):
                    cpu_block_id = cpu_block_table[i]
                    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_id, chunk_idx=cpu_block_id)

                for block_idx in range(num_blocks):
                    current_slot = load_slots[block_idx % num_slots]
                    offload_engine.wait_slot_layer(current_slot)

                    prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                    # Reshape: [1, block_size, num_kv_heads, head_dim] -> [num_kv_heads, block_size, head_dim]
                    k_block = prev_k.squeeze(0).permute(1, 0, 2).contiguous()
                    v_block = prev_v.squeeze(0).permute(1, 0, 2).contiguous()
                    k_blocks.append(k_block)
                    v_blocks.append(v_block)
                    offload_engine.record_slot_compute_done(current_slot)

                    # Issue next transfer
                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id, chunk_idx=next_cpu_block_id)

            # Concatenate all KV blocks into single tensors
            # Kernel will process all blocks internally with BLASST skip logic
            k_all = torch.cat(k_blocks, dim=1)  # [num_kv_heads, total_kv_len, head_dim]
            v_all = torch.cat(v_blocks, dim=1)  # [num_kv_heads, total_kv_len, head_dim]

            # Process all query sub-chunks with the concatenated KV tensors
            for q_sub_idx in range(num_q_subchunks):
                q_start = q_sub_idx * granularity
                q_end = min((q_sub_idx + 1) * granularity, q_len)

                # [1, sub_len, num_heads, head_dim] -> [num_heads, sub_len, head_dim]
                q_sub = q_batched[:, q_start:q_end, :, :]
                q_sub_kernel = q_sub.squeeze(0).permute(1, 0, 2).contiguous()

                with torch.cuda.stream(compute_stream):
                    # Step 1: Compute skip mask for precise statistics
                    skip_mask = blasst_mask_forward(
                        q=q_sub_kernel,
                        k=k_all,
                        running_max=q_sub_running_max[q_sub_idx],
                        ln_lambda=ln_lambda,
                        softmax_scale=softmax_scale,
                        granularity=granularity,
                    )

                    # Update skip statistics - precise counting from mask
                    # skip_mask shape: [num_heads, sub_len, num_kv_blocks]
                    total_subblocks += skip_mask.numel()
                    skipped_subblocks += int(skip_mask.sum().item())

                    # Step 2: Compute attention using skip mask
                    fused_o, new_running_max, block_lse = blasst_attn_forward(
                        q=q_sub_kernel,
                        k=k_all,
                        v=v_all,
                        skip_mask=skip_mask,
                        running_max=q_sub_running_max[q_sub_idx],
                        softmax_scale=softmax_scale,
                        granularity=granularity,
                    )

                    # Convert kernel output to flash-attn format for merging with current chunk
                    sub_o = fused_o.permute(1, 0, 2).unsqueeze(0).contiguous()
                    sub_lse = block_lse.unsqueeze(0)

                    # Store output for later merging with current (causal) chunk
                    q_sub_outputs[q_sub_idx] = (sub_o, sub_lse)
                    q_sub_running_max[q_sub_idx] = new_running_max

            # Update statistics
            self._stats_skipped_subblocks += skipped_subblocks
            self._stats_total_subblocks += total_subblocks

            if layer_id == 0:
                skip_rate = skipped_subblocks / total_subblocks if total_subblocks > 0 else 0.0
                logger.info(f"[BLASST] Chunk {current_chunk_idx}: skipped {skipped_subblocks}/{total_subblocks} "
                           f"({skip_rate:.1%}) sub-blocks (granularity={granularity})")

        # Process current chunk (causal mask) for each query sub-chunk
        final_outputs = []
        for q_sub_idx in range(num_q_subchunks):
            q_start = q_sub_idx * granularity
            q_end = min((q_sub_idx + 1) * granularity, q_len)
            q_sub = q_batched[:, q_start:q_end, :, :]

            with torch.cuda.stream(compute_stream):
                k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
                current_o, current_lse = flash_attn_with_lse(
                    q_sub, k_curr, v_curr,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

                # Merge historical and current attention for this sub-chunk
                o_acc, lse_acc = q_sub_outputs[q_sub_idx]
                if o_acc is None:
                    final_sub_o = current_o
                else:
                    final_sub_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

            final_outputs.append(final_sub_o)

        # Concatenate all sub-chunk outputs
        with torch.cuda.stream(compute_stream):
            final_o = torch.cat(final_outputs, dim=1)  # [1, seq_len, num_heads, head_dim]

        # Sync default stream with compute_stream before returning
        torch.cuda.default_stream().wait_stream(compute_stream)

        # Remove batch dimension
        return final_o.squeeze(0)

    def compute_chunked_decode(
        self,
        q: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        seq: "Sequence",
        selected_blocks: List[int],
    ) -> torch.Tensor:
        """
        Compute attention for chunked decode.

        BLASST uses full attention on all prefilled blocks during decode
        (since select_blocks returns all available blocks).

        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            seq: Sequence object
            selected_blocks: List of CPU block IDs to process (already filtered)

        Returns:
            Attention output [batch_size, 1, num_heads, head_dim]
        """
        # BLASST uses FullAttentionPolicy for decode computation
        # (select_blocks already returns all blocks)
        from .full_policy import FullAttentionPolicy

        logger.debug(f"[BLASST] compute_chunked_decode using FullAttentionPolicy, "
                    f"layer={layer_id}, selected_blocks={len(selected_blocks)}")

        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq, selected_blocks
        )

    def __repr__(self) -> str:
        if self.fixed_lambda is not None:
            return f"BLASSTPolicy(fixed_lambda={self.fixed_lambda}, granularity={self.granularity})"
        return f"BLASSTPolicy(a={self.a}, granularity={self.granularity})"
