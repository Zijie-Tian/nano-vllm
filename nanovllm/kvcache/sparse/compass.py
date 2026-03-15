"""
COMPASS sparse attention policy for chunked prefill.

Two-tier sparse attention pipeline:
- Tier 1 (CPU, coarse): Pooled cosine similarity + softmax + top-p selection
  produces a block-sparse mask at 128-token sub-block granularity
- Tier 2 (GPU, fine): BLASST dynamically prunes within the CPU-selected sub-blocks

Key design:
- IO granularity: 4096-token chunks (offload_engine block_size)
- Selection granularity: 128-token sub-blocks (FINE_GRAIN)
- CPU produces a sub-block mask → converted to BLASST mask_buffer
- BLASST further prunes sub-blocks dynamically via softmax thresholding
"""

import logging
import math
import time
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext, SubBlockSelection

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class COMPASSPolicy(SparsePolicy):
    """
    COMPASS sparse attention policy with two-tier sparsity.

    Tier 1 (CPU): Pooled cosine similarity + softmax + top-p → sub-block mask
                   at 128-token granularity. Passed to BLASST as mask_buffer.
    Tier 2 (GPU): BLASST dynamic pruning on top of the CPU mask.

    Selection granularity: 128 tokens (sub-block)
    IO granularity: 4096 tokens (chunk)
    """

    supports_prefill = True
    supports_decode = True

    # Selection granularity (tokens per sub-block for CPU estimation)
    FINE_GRAIN = 128

    def __init__(
        self,
        lambda_threshold: float = 0.001,
        top_p: float = 0.9,
        **kwargs,
    ):
        self._stats_num_chunks = 0
        self._stats_selected_subblocks = 0
        self._stats_total_subblocks = 0
        self.lambda_threshold = lambda_threshold
        self.top_p = top_p

        # Model dimensions (set during alloc_policy_metadata)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        # Q buffer for metadata tracking
        self._q_buffer: Optional[torch.Tensor] = None
        self._q_chunk_sizes: list[int] = []

        # Pooled K cache: layer_id -> {cpu_block_id -> pooled_k tensor}
        # Each pooled_k has shape [num_subblocks, kv_heads, head_dim] in FP32
        self._k_pooled_cache: Dict[int, Dict[int, torch.Tensor]] = {}

        # Per-block sub-block mask from CPU estimation
        # layer_id -> {cpu_block_id -> bool tensor [num_subblocks]}
        self._block_masks: Dict[int, Dict[int, torch.Tensor]] = {}

        # Sub-block level selections for gather-based transfer
        # layer_id -> SubBlockSelection
        self._compacted_selections: Dict[int, SubBlockSelection] = {}

        # Profiling accumulators
        self._prof_sync = 0.0
        self._prof_q_cpu = 0.0
        self._prof_q_pool = 0.0
        self._prof_k_collect = 0.0
        self._prof_matmul = 0.0
        self._prof_topp = 0.0
        self._prof_mask_build = 0.0
        self._prof_calls = 0

        # BLASST second-stage pruning stats
        self._blasst_total_blocks = 0
        self._blasst_computed_blocks = 0

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
        if not enable_cpu_offload:
            return

        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        logger.info("[COMPASS] Allocating metadata buffers (pooled top-p mode)...")

        self._q_buffer = torch.zeros(
            (num_layers, max_seq_len, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True,
        )

        # Dedicated stream for metadata GPU→CPU transfers (non-blocking)
        self._metadata_stream = torch.cuda.Stream()

        # Pre-allocated pinned CPU buffer for pooled Q
        # Shape: [max_q_groups, num_kv_heads, head_dim] in FP32
        max_q_groups = max_seq_len // self.FINE_GRAIN + 1
        self._q_pooled_cpu_buf = torch.zeros(
            (max_q_groups, num_kv_heads, head_dim),
            dtype=torch.float32, device='cpu',
        ).pin_memory()

        q_mb = self._q_buffer.numel() * self._q_buffer.element_size() / (1024 * 1024)
        qp_kb = self._q_pooled_cpu_buf.numel() * 4 / 1024
        logger.info(f"[COMPASS] Allocated Q buffer: {q_mb:.1f} MB (Pinned CPU)")
        logger.info(f"[COMPASS] Allocated pooled Q buffer: {qp_kb:.1f} KB (Pinned CPU)")
        logger.info(f"[COMPASS] Metadata stream created")
        logger.info(f"[COMPASS] Selection granularity: {self.FINE_GRAIN} tokens, "
                     f"top_p={self.top_p}")

        self._k_pooled_cache = {lid: {} for lid in range(num_layers)}
        self._block_masks = {lid: {} for lid in range(num_layers)}
        self._q_chunk_sizes = []

    # ========================================================================
    # Pooling & Estimation Helpers
    # ========================================================================

    @staticmethod
    def _pool_and_normalize(
        x: torch.Tensor,
        pool_size: int = 128,
    ) -> torch.Tensor:
        """Mean pool tokens into groups and L2 normalize for cosine similarity.

        Args:
            x: [seq_len, heads, dim] tensor.
            pool_size: Number of tokens per pool group.

        Returns:
            [num_groups, heads, dim] pooled & L2-normalized tensor in FP32.
        """
        seq_len, heads, dim = x.shape
        num_groups = seq_len // pool_size
        if num_groups == 0:
            pooled = x.float().mean(dim=0, keepdim=True)
        else:
            aligned = num_groups * pool_size
            x_aligned = x[:aligned].float()
            x_grouped = x_aligned.reshape(num_groups, pool_size, heads, dim)
            pooled = x_grouped.mean(dim=1)

        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def _precompute_pooled_k_gpu(
        self,
        k_cache_gpu: torch.Tensor,
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
    ) -> None:
        """Compute pooled & normalized K on GPU, then store tiny result on CPU.

        Called from on_prefill_offload BEFORE D2H copy, using GPU-resident k_cache.
        The pooled result is ~32KB per block (vs ~2MB for full K), so the
        GPU→CPU transfer of the pooled tensor is negligible.
        """
        # k_cache_gpu: [block_size, kv_heads, head_dim] on GPU
        k_block = k_cache_gpu[:num_tokens]  # [num_tokens, kv_heads, head_dim]
        # Compute on GPU: mean pool + L2 normalize
        pooled_k_gpu = self._pool_and_normalize(k_block, self.FINE_GRAIN)
        # Transfer only the tiny pooled result to CPU (~32 × H × D × 4 = ~32KB)
        self._k_pooled_cache[layer_id][cpu_block_id] = pooled_k_gpu.cpu()

    # ========================================================================
    # CPU Pooled Estimation (128-token sub-block granularity)
    # ========================================================================

    def _estimate_subblock_mask(
        self,
        layer_id: int,
        q_pooled: torch.Tensor,
        available_blocks: list,
        block_size: int,
    ) -> Dict[int, torch.Tensor]:
        """Estimate which 128-token sub-blocks to keep using pooled cosine
        similarity + softmax + top-p.

        Algorithm:
        1. (done by caller) Pool Q on GPU, GQA fold, L2 normalize, transfer to CPU
        2. Collect pre-computed pooled K for available blocks → [G_k, H_kv, D]
        3. Per KV head: Q_h @ K_h^T → [G_q, G_k] (cosine similarity)
        4. Softmax per row → probability distribution
        5. Top-p per row → selected K sub-blocks
        6. Union across all rows and heads → sub-block mask per IO block

        Args:
            layer_id: Current layer index.
            q_pooled: Pre-pooled Q on CPU [G_q, num_kv_heads, head_dim] (FP32).
            available_blocks: List of CPU block IDs.
            block_size: Tokens per IO block (4096).

        Returns:
            Dict mapping cpu_block_id → bool tensor [num_subblocks_in_block]
            True = keep sub-block, False = skip
        """
        fine_grain = self.FINE_GRAIN  # 128
        num_fine_per_block = block_size // fine_grain  # 32

        G_q = q_pooled.shape[0]

        if G_q == 0:
            # Return all-True masks
            return {bid: torch.ones(num_fine_per_block, dtype=torch.bool)
                    for bid in available_blocks}

        # 2. Collect pooled K
        t_start = time.perf_counter()
        pooled_k_list = []
        subblock_to_block = []  # Maps global sub-block idx → (block_id, local_sub_idx)

        for bid in available_blocks:
            if bid not in self._k_pooled_cache.get(layer_id, {}):
                logger.warning(f"[COMPASS] Pooled K cache miss for layer={layer_id}, block={bid}")
                return {bid: torch.ones(num_fine_per_block, dtype=torch.bool)
                        for bid in available_blocks}

            pk = self._k_pooled_cache[layer_id][bid]
            pooled_k_list.append(pk)
            for si in range(pk.shape[0]):
                subblock_to_block.append((bid, si))

        if not pooled_k_list:
            return {}

        k_pooled = torch.cat(pooled_k_list, dim=0)  # [G_k, kv_heads, D]
        G_k = k_pooled.shape[0]
        H = self._num_kv_heads
        t_k_collect = time.perf_counter()
        self._prof_k_collect += t_k_collect - t_start

        # 3. Compute cosine similarity: [H, G_q, G_k]
        #    Then AGGREGATE across Q sub-blocks (mean) to get [H, G_k]
        #    This avoids the problem where union of per-Q top-p sets covers everything.
        q_t = q_pooled.permute(1, 0, 2)  # [H, G_q, D]
        k_t = k_pooled.permute(1, 2, 0)  # [H, D, G_k]
        scores = torch.bmm(q_t, k_t)     # [H, G_q, G_k]
        avg_scores = scores.mean(dim=1)   # [H, G_k]  — average over Q groups
        probs = torch.softmax(avg_scores, dim=-1)  # [H, G_k]
        t_matmul = time.perf_counter()
        self._prof_matmul += t_matmul - t_k_collect

        # 4. Vectorized top-p selection on averaged probabilities
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)  # [H, G_k]

        # Mask: include elements until cumsum exceeds top_p (+1 boundary element)
        mask_in_sorted = torch.cat([
            torch.ones(H, 1, dtype=torch.bool),
            cumsum[:, :-1] <= self.top_p,
        ], dim=-1)  # [H, G_k]

        # Scatter back to original index space
        selected_mask = torch.zeros_like(probs, dtype=torch.bool)  # [H, G_k]
        selected_mask.scatter_(1, sorted_indices, mask_in_sorted)

        # Union across heads only (each head votes independently)
        overall_selected = selected_mask.any(dim=0)  # [G_k]
        t_topp = time.perf_counter()
        self._prof_topp += t_topp - t_matmul

        # 5. Build per-block sub-block masks
        block_masks: Dict[int, torch.Tensor] = {}
        for bid in available_blocks:
            pk = self._k_pooled_cache[layer_id][bid]
            num_sub = pk.shape[0]
            block_masks[bid] = torch.ones(num_sub, dtype=torch.bool)

        for idx in range(G_k):
            bid, si = subblock_to_block[idx]
            block_masks[bid][si] = overall_selected[idx]

        t_mask = time.perf_counter()
        self._prof_mask_build += t_mask - t_topp

        # Stats
        total_sub = G_k
        selected_sub = overall_selected.sum().item()
        self._stats_total_subblocks += total_sub
        self._stats_selected_subblocks += selected_sub

        return block_masks

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
        """Select blocks and compute sub-block masks at 128-token granularity.

        The sub-block masks are stored and later applied as mask_buffer
        in compute_chunked_prefill.

        Returns all available_blocks for IO (since IO is at 4096-token level),
        but sets internal masks at 128-token sub-block level.
        """
        # Save Q to buffer
        if q is not None and self._q_buffer is not None:
            start_idx = ctx.total_kv_len
            seq_len = q.shape[0]
            self._q_buffer[ctx.layer_id, start_idx:start_idx + seq_len].copy_(q, non_blocking=True)
            if ctx.layer_id == 0:
                self._q_chunk_sizes.append(seq_len)

        if ctx.layer_id == 0:
            self._stats_num_chunks += 1

        # During decode or no blocks: return all, no mask
        if not ctx.is_prefill or not available_blocks:
            return available_blocks

        self._prof_calls += 1
        heads_per_group = self._num_heads // self._num_kv_heads

        # 1. Synchronize to ensure previous GPU work (including pooled K) is done
        t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        self._prof_sync += t1 - t0

        # 2. GPU-side Q pooling + GQA fold (fast on GPU)
        q_pooled_gpu = self._pool_and_normalize(q, self.FINE_GRAIN)
        G_q = q_pooled_gpu.shape[0]
        # GQA fold on GPU: [G_q, H, D] → [G_q, H_kv, hpg, D] → mean → [G_q, H_kv, D]
        q_pooled_gpu = q_pooled_gpu.reshape(G_q, self._num_kv_heads, heads_per_group, self._head_dim)
        q_pooled_gpu = q_pooled_gpu.mean(dim=2)  # [G_q, H_kv, D]
        t2 = time.perf_counter()
        self._prof_q_pool += t2 - t1

        # 3. Async transfer pooled Q to pinned CPU buffer (~128KB vs 32MB)
        with torch.cuda.stream(self._metadata_stream):
            self._q_pooled_cpu_buf[:G_q].copy_(q_pooled_gpu, non_blocking=True)
        self._metadata_stream.synchronize()
        q_pooled_cpu = self._q_pooled_cpu_buf[:G_q].clone()  # snapshot
        t3 = time.perf_counter()
        self._prof_q_cpu += t3 - t2

        # 4. CPU-side estimation using pre-pooled Q
        block_masks = self._estimate_subblock_mask(
            layer_id=ctx.layer_id,
            q_pooled=q_pooled_cpu,
            available_blocks=available_blocks,
            block_size=ctx.block_size,
        )
        t4 = time.perf_counter()

        # Store masks for use in compute_chunked_prefill
        self._block_masks[ctx.layer_id] = block_masks

        # Compute and log statistics
        total_sub = sum(m.numel() for m in block_masks.values())
        selected_sub = sum(m.sum().item() for m in block_masks.values())
        density = selected_sub / max(total_sub, 1) * 100

        logger.info(
            f"[COMPASS] layer={ctx.layer_id}, chunk={ctx.query_chunk_idx}, "
            f"sub-blocks: {int(selected_sub)}/{total_sub} ({density:.1f}%), "
            f"top_p={self.top_p}"
        )

        # Return ALL blocks for IO (sub-block filtering is via mask_buffer)
        # Optimization: skip blocks where ALL sub-blocks are masked
        io_blocks = [bid for bid in available_blocks
                     if bid in block_masks and block_masks[bid].any()]

        # Build compacted selections: (block_id, [selected sub-block indices])
        selection_entries = []
        for bid in io_blocks:
            mask = block_masks[bid]
            selected_indices = mask.nonzero(as_tuple=True)[0].tolist()
            if selected_indices:
                selection_entries.append((bid, selected_indices))

        self._compacted_selections[ctx.layer_id] = SubBlockSelection(
            entries=selection_entries,
            sub_block_size=self.FINE_GRAIN,
        )

        return io_blocks

    # ========================================================================
    # Statistics
    # ========================================================================

    def reset_stats(self) -> None:
        self._stats_num_chunks = 0
        self._stats_selected_subblocks = 0
        self._stats_total_subblocks = 0
        self._blasst_total_blocks = 0
        self._blasst_computed_blocks = 0

    def get_stats(self) -> dict:
        select_rate = 0.0
        io_reduction = 0.0
        if self._stats_total_subblocks > 0:
            select_rate = self._stats_selected_subblocks / self._stats_total_subblocks
            io_reduction = 1.0 - select_rate
        blasst_density = 0.0
        if self._blasst_total_blocks > 0:
            blasst_density = self._blasst_computed_blocks / self._blasst_total_blocks
        return {
            "num_chunks": self._stats_num_chunks,
            "selected_subblocks": self._stats_selected_subblocks,
            "total_subblocks": self._stats_total_subblocks,
            "select_rate": select_rate,
            "io_reduction": io_reduction,
            "blasst_total_blocks": self._blasst_total_blocks,
            "blasst_computed_blocks": self._blasst_computed_blocks,
            "blasst_density": blasst_density,
            "prof_calls": self._prof_calls,
            "prof_sync": self._prof_sync,
            "prof_q_cpu": self._prof_q_cpu,
            "prof_q_pool": self._prof_q_pool,
            "prof_k_collect": self._prof_k_collect,
            "prof_matmul": self._prof_matmul,
            "prof_topp": self._prof_topp,
            "prof_mask_build": self._prof_mask_build,
        }

    # ========================================================================
    # GPU-only methods - Not supported
    # ========================================================================

    def compute_prefill(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale, layer_id, block_tables=None,
    ) -> torch.Tensor:
        raise NotImplementedError("COMPASS only supports chunked prefill mode.")

    def compute_decode(
        self, q, k_cache, v_cache, cache_seqlens, softmax_scale, layer_id, block_tables=None,
    ) -> torch.Tensor:
        raise NotImplementedError("COMPASS only supports chunked decode mode.")

    # ========================================================================
    # Chunked offload methods — BLASST-like GPU attention with CPU mask
    # ========================================================================

    def _build_mask_buffer(
        self,
        block_id: int,
        layer_id: int,
        q_len: int,
        kv_len: int,
        num_heads: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Convert 128-token sub-block mask to BLASST mask_buffer format.

        BLASST mask_buffer shape: [grid_0, grid_1, num_kv_blocks]
        - grid_0 = ceil(q_len / BLOCK_M), where BLOCK_M = 128
        - grid_1 = num_heads (batch=1)
        - num_kv_blocks = ceil(kv_len / BLOCK_N), where BLOCK_N = 64

        Our sub-block mask is at 128-token granularity.
        Each 128-token sub-block → 2 BLASST BLOCK_N's (64 tokens each).
        """
        BLOCK_M = 128
        BLOCK_N = 64
        grid_0 = (q_len + BLOCK_M - 1) // BLOCK_M
        num_kv_blocks = (kv_len + BLOCK_N - 1) // BLOCK_N

        masks = self._block_masks.get(layer_id, {})
        if block_id not in masks:
            return None

        submask = masks[block_id]  # [num_128_subblocks] bool
        num_subblocks = submask.shape[0]

        # Expand 128-token mask → 64-token BLASST blocks (each sub-block = 2 BLOCK_N's)
        expanded = submask.repeat_interleave(self.FINE_GRAIN // BLOCK_N)
        # Pad or truncate to match num_kv_blocks
        if expanded.shape[0] < num_kv_blocks:
            expanded = F.pad(expanded, (0, num_kv_blocks - expanded.shape[0]), value=True)
        elif expanded.shape[0] > num_kv_blocks:
            expanded = expanded[:num_kv_blocks]

        # Build [grid_0, grid_1, num_kv_blocks] mask
        # Same mask for all Q groups and all heads (CPU selects globally)
        mask = expanded.to(torch.int8).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(grid_0, num_heads, num_kv_blocks).contiguous().to(device)
        return mask

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
        """Compute attention with sub-block gather + BLASST dynamic pruning.

        Uses gather-based transfer: only selected sub-blocks are compacted into
        staging buffer and bulk-transferred to GPU, reducing H2D traffic.
        """
        from nanovllm.ops.chunked_attention import (
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )
        from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill

        lambda_val = self.lambda_threshold
        ln_lambda = math.log(lambda_val)

        q_len = q.shape[0]
        num_heads = q.shape[1]
        compute_stream = offload_engine.compute_stream
        q_input = q.unsqueeze(0).transpose(1, 2).contiguous()

        historical_o = None
        historical_lse = None
        historical_m_global = None

        logger.debug(
            f"[COMPASS] compute_chunked_prefill: layer={layer_id}, "
            f"chunk={current_chunk_idx}, num_tokens={num_tokens}, "
            f"selected_blocks={len(selected_blocks)}"
        )

        # ---- Gather-based historical block processing ----
        selection = self._compacted_selections.get(layer_id)
        if selection is not None and selection.total_subblocks > 0:
            fine_grain = selection.sub_block_size
            block_size = kvcache_manager.block_size
            max_subblocks_per_slot = block_size // fine_grain  # 32 for 4096/128

            # Flatten all selected sub-blocks into a list for batching
            all_subblocks = []  # [(block_id, sub_idx), ...]
            for bid, sub_indices in selection.entries:
                for si in sub_indices:
                    all_subblocks.append((bid, si))

            total_selected = len(all_subblocks)
            logger.info(
                f"[COMPASS] layer={layer_id}, chunk={current_chunk_idx}: "
                f"gather {total_selected} sub-blocks "
                f"({total_selected * fine_grain} tokens) from "
                f"{len(selection.entries)} blocks"
            )

            # Batch into slot-sized groups
            load_slots = list(range(offload_engine.num_ring_slots))
            num_slots = len(load_slots)
            batches = []
            for start in range(0, total_selected, max_subblocks_per_slot):
                end = min(start + max_subblocks_per_slot, total_selected)
                batch_items = all_subblocks[start:end]
                # Group by block_id for gather_subblocks_to_staging
                batch_grouped = {}  # block_id -> [sub_indices]
                for bid, si in batch_items:
                    batch_grouped.setdefault(bid, []).append(si)
                batch_selections = list(batch_grouped.items())
                batch_tokens = len(batch_items) * fine_grain
                batches.append((batch_selections, batch_tokens))

            num_batches = len(batches)

            # IMPORTANT: staging buffer is shared — only ONE gather+H2D can be
            # in-flight at a time. Preload only the first batch, then pipeline:
            # after waiting for current slot, preload the next batch.
            if num_batches > 0:
                batch_sel, batch_tok = batches[0]
                offload_engine.gather_subblocks_to_staging(
                    layer_id, batch_sel, fine_grain
                )
                offload_engine.load_staging_to_slot(
                    load_slots[0], batch_tok, layer_id=layer_id
                )

            # Process batches with ring buffer pipeline
            for batch_idx in range(num_batches):
                current_slot = load_slots[batch_idx % num_slots]
                _, batch_tokens = batches[batch_idx]

                offload_engine.wait_slot_layer(current_slot)

                with torch.cuda.stream(compute_stream):
                    # Get only the valid portion of the slot
                    prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                    prev_k = prev_k[:, :batch_tokens, :, :]
                    prev_v = prev_v[:, :batch_tokens, :, :]
                    k_input = prev_k.transpose(1, 2).contiguous()
                    v_input = prev_v.transpose(1, 2).contiguous()

                    # Create all-ones mask to track BLASST pruning
                    BLOCK_M, BLOCK_N = 128, 64
                    grid_0 = (q_input.shape[2] + BLOCK_M - 1) // BLOCK_M
                    grid_1 = q_input.shape[0] * q_input.shape[1]  # batch * heads
                    num_kv_blocks = (k_input.shape[2] + BLOCK_N - 1) // BLOCK_N
                    mask_buf = torch.ones(
                        (grid_0, grid_1, num_kv_blocks),
                        dtype=torch.int8, device=q_input.device,
                    )

                    out, lse, m_global_out = blasst_chunked_prefill(
                        q=q_input,
                        k=k_input,
                        v=v_input,
                        threshold_ln_lambda=ln_lambda,
                        m_global_in=historical_m_global,
                        mask_buffer=mask_buf,
                    )

                    # Track BLASST pruning stats
                    self._blasst_total_blocks += mask_buf.numel()
                    self._blasst_computed_blocks += int(mask_buf.sum().item())

                    if historical_m_global is None:
                        historical_m_global = m_global_out
                    else:
                        historical_m_global = torch.maximum(
                            historical_m_global, m_global_out
                        )

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

                # Pipeline: preload next batch (only 1 ahead due to shared staging)
                next_batch_idx = batch_idx + 1
                if next_batch_idx < num_batches:
                    next_slot = load_slots[next_batch_idx % num_slots]
                    next_sel, next_tok = batches[next_batch_idx]
                    offload_engine.gather_subblocks_to_staging(
                        layer_id, next_sel, fine_grain
                    )
                    offload_engine.load_staging_to_slot(
                        next_slot, next_tok, layer_id=layer_id
                    )

        # ---- Current prefill chunk (causal, no gather) ----
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(
                layer_id, num_tokens
            )
            k_curr_input = k_curr.transpose(1, 2).contiguous()
            v_curr_input = v_curr.transpose(1, 2).contiguous()

            # kv_offset: total historical tokens for causal masking
            # Use actual selected token count, not block count × block_size
            kv_offset = (
                selection.total_tokens
                if selection is not None
                else len(selected_blocks) * kvcache_manager.block_size
            )

            out_curr, lse_curr, _ = blasst_chunked_prefill(
                q=q_input,
                k=k_curr_input,
                v=v_curr_input,
                threshold_ln_lambda=ln_lambda,
                m_global_in=historical_m_global,
                mask_buffer=None,
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
        from .full_policy import FullAttentionPolicy
        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq, selected_blocks,
        )

    # ========================================================================
    # Offload hooks
    # ========================================================================

    def on_prefill_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """Hook called BEFORE GPU→CPU copy with GPU-resident k_cache.

        Computes pooled K on GPU and stores the tiny result on CPU.
        This avoids waiting for D2H completion and leverages GPU speed.
        """
        self._precompute_pooled_k_gpu(k_cache, layer_id, cpu_block_id, num_valid_tokens)

    def offload_prefill_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
        **kwargs,
    ) -> None:
        # on_prefill_offload is called inside super() with GPU k_cache data
        # → pooled K is computed on GPU before D2H copy
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
        return f"COMPASSPolicy(top_p={self.top_p}, lambda={self.lambda_threshold})"
