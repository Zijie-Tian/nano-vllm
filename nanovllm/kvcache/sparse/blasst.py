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
    - Historical KV is divided into sub-blocks of `granularity` tokens
    - The Triton kernel iterates over the full query chunk internally
    - BLASST skip condition is still applied per (query, kv_sub_block) pair
    - The current causal chunk is merged back in granularity-sized slices
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

        Uses ring buffer pipeline for IO-compute overlap:
        1. Load KV blocks one at a time through ring buffer
        2. For each block, compute BLASST mask + attention with streaming running_max
        3. Online merge results using LSE
        4. Memory bounded by ring buffer size, not total KV length

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
            selected_blocks: List of CPU block IDs to process

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
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
        num_heads = q.shape[1]
        compute_stream = offload_engine.compute_stream

        # Prepare query in BLASST layout [batch, num_heads, q_len, head_dim]
        q_input = q_batched.transpose(1, 2).contiguous()

        # Accumulators for historical attention
        historical_o = None
        historical_lse = None

        cpu_block_table = selected_blocks

        # Density tracking per chunk for the first layer
        collect_density = (layer_id == 0)
        compute_density_sum = 0.0
        required_kv_density_sum = 0.0
        num_density_measurements = 0

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            if layer_id == 0:
                logger.info(f"[BLASST] Chunk {current_chunk_idx}: "
                           f"seq_len={total_seq_len}, λ={lambda_val:.4f}, "
                           f"blocks={num_blocks}")

            from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill

            # Determine mask buffer size if collecting density
            # TRITON_BLOCK_M = 128, TRITON_BLOCK_N = 64
            TRITON_BLOCK_M, TRITON_BLOCK_N = 128, 64
            grid_0 = (q_len + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
            grid_1 = 1 * num_heads # batch=1
            num_kv_subblocks = kvcache_manager.block_size // TRITON_BLOCK_N
            
            def get_mask_buffer():
                if collect_density:
                    return torch.zeros((grid_0, grid_1, num_kv_subblocks), device=q.device, dtype=torch.int8)
                return None

            if len(load_slots) == 1:
                # Only 1 slot - synchronous mode
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id)
                    offload_engine.wait_slot_layer(slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
                        k_input = prev_k.transpose(1, 2).contiguous()
                        v_input = prev_v.transpose(1, 2).contiguous()

                        mask_buffer = get_mask_buffer()
                        out, lse = blasst_chunked_prefill(
                            q=q_input, k=k_input, v=v_input,
                            threshold_ln_lambda=ln_lambda,
                            mask_buffer=mask_buffer
                        )
                        
                        if mask_buffer is not None:
                            compute_stream.synchronize()
                            # 1. Compute Density (Average skip rate)
                            compute_density_sum += mask_buffer.float().mean().item()
                            # 2. Required KV Density (Logical OR along Q-axis)
                            # Shape: [grid_1, num_kv_subblocks]
                            required_kv_mask = mask_buffer.any(dim=0)
                            required_kv_density_sum += required_kv_mask.float().mean().item()
                            num_density_measurements += 1

                        block_o = out.transpose(1, 2).contiguous()
                        block_lse = lse

                        if historical_o is None:
                            historical_o, historical_lse = block_o, block_lse
                        else:
                            historical_o, historical_lse = merge_attention_outputs(
                                historical_o, historical_lse, block_o, block_lse
                            )

                        offload_engine.record_slot_compute_done(slot)
            else:
                # Multiple slots - pipeline mode
                num_slots = len(load_slots)
                num_preload = min(num_slots, num_blocks)

                for i in range(num_preload):
                    cpu_block_id = cpu_block_table[i]
                    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_id, chunk_idx=cpu_block_id)

                for block_idx in range(num_blocks):
                    current_slot = load_slots[block_idx % num_slots]
                    offload_engine.wait_slot_layer(current_slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                        k_input = prev_k.transpose(1, 2).contiguous()
                        v_input = prev_v.transpose(1, 2).contiguous()

                        mask_buffer = get_mask_buffer()
                        out, lse = blasst_chunked_prefill(
                            q=q_input, k=k_input, v=v_input,
                            threshold_ln_lambda=ln_lambda,
                            mask_buffer=mask_buffer
                        )
                        
                        if mask_buffer is not None:
                            compute_stream.synchronize()
                            compute_density_sum += mask_buffer.float().mean().item()
                            required_kv_mask = mask_buffer.any(dim=0)
                            required_kv_density_sum += required_kv_mask.float().mean().item()
                            num_density_measurements += 1

                        block_o = out.transpose(1, 2).contiguous()
                        block_lse = lse

                        if historical_o is None:
                            historical_o, historical_lse = block_o, block_lse
                        else:
                            historical_o, historical_lse = merge_attention_outputs(
                                historical_o, historical_lse, block_o, block_lse
                            )

                        offload_engine.record_slot_compute_done(current_slot)

                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id, chunk_idx=next_cpu_block_id)

        # Log density if collected
        if num_density_measurements > 0:
            avg_compute_density = compute_density_sum / num_density_measurements
            avg_required_kv_density = required_kv_density_sum / num_density_measurements
            logger.info(f"[BLASST] Chunk {current_chunk_idx} Stats: "
                       f"Compute Density={avg_compute_density*100:.2f}%, "
                       f"Required KV Density={avg_required_kv_density*100:.2f}%")

        # Process current chunk (causal mask)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            current_o, current_lse = flash_attn_with_lse(
                q_batched, k_curr, v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

            if historical_o is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(historical_o, historical_lse, current_o, current_lse)

        torch.cuda.default_stream().wait_stream(compute_stream)
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
        """Compute attention for chunked decode."""
        from .full_policy import FullAttentionPolicy
        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq, selected_blocks
        )

    def __repr__(self) -> str:
        if self.fixed_lambda is not None:
            return f"BLASSTPolicy(fixed_lambda={self.fixed_lambda}, granularity={self.granularity})"
        return f"BLASSTPolicy(a={self.a}, granularity={self.granularity})"
