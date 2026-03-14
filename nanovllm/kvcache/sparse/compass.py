"""
COMPASS sparse attention policy for chunked prefill.

Two-tier sparse attention pipeline:
- Tier 1 (CPU, coarse): Torch FP32 GEMM on FP16 k_cache_cpu filters which
  4096-token blocks to load
- Tier 2 (GPU, fine): BLASST dynamically prunes sub-blocks within loaded blocks

select_blocks uses CPU torch GEMM prediction to filter available blocks.
compute_chunked_prefill uses BLASST-like attention with dynamic pruning on
the CPU-pre-filtered block subset.
"""

import logging
import math
import torch
from typing import List, Optional, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class COMPASSPolicy(SparsePolicy):
    """
    COMPASS sparse attention policy with two-tier sparsity.

    Tier 1 (CPU): Torch FP32 GEMM in select_blocks filters coarse 4096-token blocks.
    Tier 2 (GPU): BLASST dynamic pruning in compute_chunked_prefill on loaded blocks.
    """

    supports_prefill = True
    supports_decode = True

    # Estimation granularity (tokens per fine-grained sub-block)
    FINE_GRAIN = 128

    def __init__(
        self,
        lambda_threshold: float = 0.001,
        **kwargs,
    ):
        """Initialize with parameters for CPU estimation and BLASST GPU attention."""
        self._stats_num_chunks = 0
        self._stats_selected_blocks = 0
        self._stats_total_blocks = 0
        self.lambda_threshold = lambda_threshold

        # Model dimensions (set during alloc_policy_metadata)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        # Q buffer for metadata tracking
        self._q_buffer: Optional[torch.Tensor] = None
        self._q_chunk_sizes: list[int] = []

    # ========================================================================
    # Initialization
    # ========================================================================

    def alloc_policy_metadata(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        enable_cpu_offload: bool = False,
        num_layers: int = 64,
    ) -> None:
        """Allocate CPU buffers for COMPASS estimation."""
        if not enable_cpu_offload:
            return

        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        logger.info("[COMPASS] Allocating metadata buffers (torch GEMM mode)...")

        # Q buffer: [num_layers, max_seq_len, num_heads, head_dim]
        self._q_buffer = torch.zeros(
            (num_layers, max_seq_len, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True,
        )

        q_mb = self._q_buffer.numel() * self._q_buffer.element_size() / (1024 * 1024)
        logger.info(f"[COMPASS] Allocated Q buffer: {q_mb:.1f} MB (Pinned CPU)")
        logger.info("[COMPASS] Using torch FP32 CPU GEMM for block estimation "
                     "(reads k_cache_cpu directly from OffloadEngine)")

        self._q_chunk_sizes = []

    # ========================================================================
    # CPU Torch GEMM Estimation
    # ========================================================================

    def _estimate_blocks_torch(
        self,
        layer_id: int,
        q_cpu: torch.Tensor,
        available_blocks: list,
        offload_engine: "OffloadEngine",
        block_size: int,
        ln_lambda: float,
    ) -> list:
        """
        Estimate which blocks to keep using torch FP32 CPU GEMM.

        For each available block, reads FP16 K data from offload_engine.k_cache_cpu,
        computes Q @ K^T per KV head per Q-group, and applies BLASST-style
        relative threshold to select important sub-blocks.

        Args:
            layer_id: Current layer index.
            q_cpu: Query tensor on CPU, shape [q_len, num_heads, head_dim].
            available_blocks: List of CPU block IDs to evaluate.
            offload_engine: OffloadEngine with k_cache_cpu.
            block_size: Tokens per block (4096).
            ln_lambda: ln(lambda_threshold) for BLASST threshold.

        Returns:
            List of selected block IDs.
        """
        fine_grain = self.FINE_GRAIN  # 128
        num_q_groups = q_cpu.shape[0] // fine_grain
        num_fine_per_block = block_size // fine_grain  # 32
        heads_per_group = self._num_heads // self._num_kv_heads

        # Build sub-block index: [(block_id, sub_idx_in_block), ...]
        # Each sub-block is fine_grain (128) tokens
        subblock_info = []
        for bid in available_blocks:
            for si in range(num_fine_per_block):
                subblock_info.append((bid, si))

        if not subblock_info:
            return available_blocks

        num_subblocks = len(subblock_info)

        # Pre-load all K blocks from CPU cache: [num_blocks, block_size, kv_heads, head_dim]
        # k_cache_cpu shape: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
        k_blocks = torch.stack([
            offload_engine.k_cache_cpu[layer_id, bid]
            for bid in available_blocks
        ], dim=0).float()  # [num_blocks, block_size, kv_heads, head_dim]

        # Reshape to sub-blocks: [num_blocks * num_fine_per_block, fine_grain, kv_heads, head_dim]
        num_blocks = len(available_blocks)
        k_subblocks = k_blocks.reshape(
            num_blocks * num_fine_per_block, fine_grain,
            self._num_kv_heads, self._head_dim,
        )  # [total_subs, fine_grain, kv_heads, head_dim]

        q_float = q_cpu.float()  # [q_len, num_heads, head_dim]

        G = num_q_groups
        H = self._num_kv_heads
        S = num_subblocks       # total sub-blocks
        F = fine_grain          # 128
        D = self._head_dim      # 128

        # Truncate Q to aligned length (drop trailing tokens < fine_grain)
        aligned_len = G * F
        if aligned_len == 0:
            return available_blocks
        q_float = q_float[:aligned_len]  # [G*F, num_heads, head_dim]

        # --- BMM: compute all (q_group, kv_head) scores in one shot ---

        # 1. Pre-compute Q representatives for all q-groups and kv-heads
        #    GQA layout: num_heads = num_kv_heads * heads_per_group
        #    e.g., heads [0,1,2,3] → kv_head 0, heads [4,5,6,7] → kv_head 1, etc.
        q_groups = q_float.reshape(G, F, H, heads_per_group, D)
        q_repr_all = q_groups.mean(dim=(1, 3))  # [G, H, D]

        # 2. Prepare K: [S, F, H, D] → [H, S*F, D]
        k_flat = k_subblocks.permute(2, 0, 1, 3).reshape(H, S * F, D)

        # 3. Prepare Q: [G, H, D] → [H, D, G]
        q_flat = q_repr_all.permute(1, 2, 0)  # [H, D, G]

        # 4. Single BMM: [H, S*F, D] @ [H, D, G] → [H, S*F, G]
        attn_all = torch.bmm(k_flat, q_flat)

        # 5. Reshape and reduce: [H, S, F, G] → per-sub-block scores [H, S, G]
        attn_all = attn_all.reshape(H, S, F, G)
        scores = attn_all.abs().amax(dim=2)  # [H, S, G]

        # 6. BLASST threshold: per (head, q-group) pair
        m_globals = scores.amax(dim=1, keepdim=True)  # [H, 1, G]
        thresholds = m_globals + ln_lambda

        # 7. Union across all heads and q-groups → selected sub-blocks
        selected_mask = (scores >= thresholds).any(dim=(0, 2))  # [S]

        # Aggregate: sub-block index → block ID
        selected_block_set = set()
        for idx in selected_mask.nonzero(as_tuple=True)[0].tolist():
            bid, _ = subblock_info[idx]
            selected_block_set.add(bid)

        return [b for b in available_blocks if b in selected_block_set]

    # ========================================================================
    # Block selection
    # ========================================================================

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select blocks using CPU torch GEMM prediction during prefill.

        Splits Q chunk (4096 tokens) into Q-groups of 128 tokens.
        Each Q-group independently estimates importance of all K sub-blocks
        (128 tokens each) using torch FP32 GEMM with BLASST relative threshold:
            score >= m_global + ln(λ)

        Takes union across all Q-groups, then aggregates to 4096-token IO blocks.
        During decode, returns all blocks.
        """
        # Save Q to buffer for metadata tracking
        if q is not None and self._q_buffer is not None:
            start_idx = ctx.total_kv_len
            seq_len = q.shape[0]
            self._q_buffer[ctx.layer_id, start_idx:start_idx + seq_len].copy_(q, non_blocking=True)
            if ctx.layer_id == 0:
                self._q_chunk_sizes.append(seq_len)

        # Track statistics (across ALL layers for correct IO reduction)
        self._stats_total_blocks += len(available_blocks)
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1

        # During decode or no blocks: return all
        if not ctx.is_prefill or not available_blocks:
            self._stats_selected_blocks += len(available_blocks)
            return available_blocks

        # --- CPU Torch GEMM Sparse Estimation ---
        # Synchronize to ensure offload has written K data to k_cache_cpu
        torch.cuda.synchronize()

        ln_lambda = math.log(self.lambda_threshold)
        q_cpu = q.cpu()

        selected = self._estimate_blocks_torch(
            layer_id=ctx.layer_id,
            q_cpu=q_cpu,
            available_blocks=available_blocks,
            offload_engine=offload_engine,
            block_size=ctx.block_size,
            ln_lambda=ln_lambda,
        )

        # Logging
        self._stats_selected_blocks += len(selected)
        density_block = len(selected) / max(len(available_blocks), 1) * 100
        logger.info(
            f"[COMPASS] layer={ctx.layer_id}, chunk={ctx.query_chunk_idx}, "
            f"blocks: {len(selected)}/{len(available_blocks)} ({density_block:.1f}%), "
            f"λ={self.lambda_threshold}"
        )

        return selected

    # ========================================================================
    # Statistics
    # ========================================================================

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats_num_chunks = 0
        self._stats_selected_blocks = 0
        self._stats_total_blocks = 0

    def get_stats(self) -> dict:
        """Get statistics including IO reduction."""
        select_rate = 0.0
        io_reduction = 0.0
        if self._stats_total_blocks > 0:
            select_rate = self._stats_selected_blocks / self._stats_total_blocks
            io_reduction = 1.0 - select_rate
        return {
            "num_chunks": self._stats_num_chunks,
            "selected_blocks": self._stats_selected_blocks,
            "total_blocks": self._stats_total_blocks,
            "select_rate": select_rate,
            "io_reduction": io_reduction,
        }

    # ========================================================================
    # GPU-only methods (non-chunked) - Not supported
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
            "COMPASS policy only supports chunked prefill mode. "
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
            "COMPASS policy only supports chunked decode mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    # ========================================================================
    # Chunked offload methods — BLASST-like GPU attention
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
        Compute attention with BLASST-like dynamic pruning on CPU-pre-filtered blocks.

        The selected_blocks have already been filtered by CPU torch GEMM prediction.
        This method applies BLASST dynamic pruning on top for fine-grained sparsity.
        """
        from nanovllm.ops.chunked_attention import (
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )
        from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill

        total_seq_len = len(seq) if seq else num_tokens
        lambda_val = self.lambda_threshold
        ln_lambda = math.log(lambda_val)

        q_len = q.shape[0]
        num_heads = q.shape[1]
        compute_stream = offload_engine.compute_stream
        q_input = q.unsqueeze(0).transpose(1, 2).contiguous()  # [1, num_heads, q_len, head_dim]

        historical_o = None
        historical_lse = None
        historical_m_global = None  # Running max for BLASST pruning

        TRITON_BLOCK_M, TRITON_BLOCK_N = 128, 64
        grid_0 = (q_len + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
        grid_1 = num_heads
        num_kv_subblocks = kvcache_manager.block_size // TRITON_BLOCK_N

        def get_mask_buffer(is_causal=False, kv_len_override=None):
            num_sub = (
                num_kv_subblocks
                if kv_len_override is None
                else (kv_len_override + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
            )
            mask = torch.ones(
                (grid_0, grid_1, num_sub), device=q.device, dtype=torch.int8
            )
            if is_causal:
                for q_idx in range(grid_0):
                    q_end_pos = (q_idx + 1) * TRITON_BLOCK_M
                    for kv_idx in range(num_sub):
                        kv_start_pos = kv_idx * TRITON_BLOCK_N
                        if kv_start_pos >= q_end_pos:
                            mask[q_idx, :, kv_idx] = 0
            return mask

        logger.debug(
            f"[COMPASS] compute_chunked_prefill: layer={layer_id}, "
            f"chunk={current_chunk_idx}, num_tokens={num_tokens}, "
            f"selected_blocks={len(selected_blocks)}"
        )

        # 1. Process CPU-pre-filtered historical blocks with BLASST pruning
        cpu_block_table = selected_blocks
        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_slots = len(load_slots)
            num_blocks = len(cpu_block_table)

            num_preload = min(num_slots, num_blocks)
            for i in range(num_preload):
                offload_engine.load_to_slot_layer(
                    load_slots[i], layer_id, cpu_block_table[i]
                )

            for block_idx in range(num_blocks):
                current_slot = load_slots[block_idx % num_slots]
                offload_engine.wait_slot_layer(current_slot)

                with torch.cuda.stream(compute_stream):
                    prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                    k_input = prev_k.transpose(1, 2).contiguous()
                    v_input = prev_v.transpose(1, 2).contiguous()

                    mask_buffer = get_mask_buffer(is_causal=False)

                    out, lse, m_global_out = blasst_chunked_prefill(
                        q=q_input,
                        k=k_input,
                        v=v_input,
                        threshold_ln_lambda=ln_lambda,
                        m_global_in=historical_m_global,
                        mask_buffer=mask_buffer,
                    )

                    # Update running max for next kernel call
                    if historical_m_global is None:
                        historical_m_global = m_global_out
                    else:
                        historical_m_global = torch.maximum(historical_m_global, m_global_out)

                    compute_stream.synchronize()

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
                    offload_engine.load_to_slot_layer(
                        load_slots[next_block_idx % num_slots],
                        layer_id,
                        cpu_block_table[next_block_idx],
                    )

        # 2. Process current prefill chunk (GPU buffer, causal)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(
                layer_id, num_tokens
            )
            k_curr_input = k_curr.transpose(1, 2).contiguous()
            v_curr_input = v_curr.transpose(1, 2).contiguous()

            kv_offset = len(selected_blocks) * kvcache_manager.block_size

            curr_mask_buffer = get_mask_buffer(
                is_causal=False, kv_len_override=num_tokens
            )

            out_curr, lse_curr, _ = blasst_chunked_prefill(
                q=q_input,
                k=k_curr_input,
                v=v_curr_input,
                threshold_ln_lambda=ln_lambda,
                m_global_in=historical_m_global,
                mask_buffer=curr_mask_buffer,
                is_causal=True,
                kv_offset=kv_offset,
            )

            compute_stream.synchronize()

            block_o = out_curr.transpose(1, 2).contiguous()
            block_lse = lse_curr

            if historical_o is None:
                final_o = block_o
            else:
                final_o, _ = merge_attention_outputs(
                    historical_o, historical_lse, block_o, block_lse
                )

        # 3. Finalize
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
        """
        Compute attention for chunked decode.
        Falls back to FullAttentionPolicy for decode phase.
        """
        from .full_policy import FullAttentionPolicy

        logger.debug(
            f"[COMPASS] compute_chunked_decode using FullAttentionPolicy, "
            f"layer={layer_id}, selected_blocks={len(selected_blocks)}"
        )

        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q,
            layer_id,
            softmax_scale,
            offload_engine,
            kvcache_manager,
            seq,
            selected_blocks,
        )

    # ========================================================================
    # Offload hooks
    # ========================================================================

    def offload_prefill_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
        **kwargs,
    ) -> None:
        """Offload prefill chunk. K data is saved to k_cache_cpu by OffloadEngine."""
        # No additional processing needed — k_cache_cpu is populated by the
        # offload engine's standard D2H path, and we read from it directly
        # in _estimate_blocks_torch.
        super().offload_prefill_chunk(
            offload_engine, layer_id, cpu_block_id, num_tokens, **kwargs
        )

    def offload_decode_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        **kwargs,
    ) -> None:
        super().offload_decode_chunk(offload_engine, layer_id, cpu_block_id, **kwargs)

    def __repr__(self) -> str:
        return f"COMPASSPolicy(lambda={self.lambda_threshold})"
