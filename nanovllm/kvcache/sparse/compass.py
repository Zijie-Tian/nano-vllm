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
import nvtx
from typing import List, Optional, Dict, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext, SubBlockSelection, PerHeadSubBlockSelection

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
        
        logger.info(f"[COMPASS] Initialized policy with top_p={self.top_p}, lambda={self.lambda_threshold}")

        # Model dimensions (set during alloc_policy_metadata)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        # Q buffer for metadata tracking
        self._q_buffer: Optional[torch.Tensor] = None
        self._q_chunk_sizes: list[int] = []

        # Pooled K cache: layer_id -> {cpu_block_id -> pooled_k tensor [G_k, H_kv, D]}
        self._k_pooled_cache: Dict[int, Dict[int, torch.Tensor]] = {}

        # Per-head sub-block selections: layer_id -> PerHeadSubBlockSelection
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
                     f"top_p={self.top_p}, lambda={self.lambda_threshold}")

        self._k_pooled_cache = {lid: {} for lid in range(num_layers)}
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
        # k_cache_gpu: [kv_heads, max_tokens, head_dim] on GPU if Head-First
        if k_cache_gpu.shape[0] == self._num_kv_heads:
            k_block = k_cache_gpu[:, :num_tokens, :].transpose(0, 1)  # [num_tokens, kv_heads, head_dim]
        else:
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
    ):
        """Estimate which 128-token sub-blocks to keep via pooled cosine sim + top-p.

        Returns:
            (selected_mask, subblock_to_block): ([H, G_k] bool, [(bid, si)])
        """
        fine_grain = self.FINE_GRAIN  # 128
        num_fine_per_block = block_size // fine_grain  # 32

        G_q = q_pooled.shape[0]

        if G_q == 0:
            # Return all-True masks
            return {bid: torch.ones(num_fine_per_block, dtype=torch.bool)
                    for bid in available_blocks}

        # 2. Collect pooled K
        nvtx.push_range("compass_k_collect", color="green")
        t_start = time.perf_counter()
        pooled_k_list = []
        subblock_to_block = []  # Maps global sub-block idx → (block_id, local_sub_idx)

        for bid in available_blocks:
            if bid not in self._k_pooled_cache.get(layer_id, {}):
                logger.warning(f"[COMPASS] Pooled K cache miss for layer={layer_id}, block={bid}")
                nvtx.pop_range()
                return {bid: torch.ones(num_fine_per_block, dtype=torch.bool)
                        for bid in available_blocks}

            pk = self._k_pooled_cache[layer_id][bid]
            pooled_k_list.append(pk)
            for si in range(pk.shape[0]):
                subblock_to_block.append((bid, si))

        if not pooled_k_list:
            nvtx.pop_range()
            return {}

        k_pooled = torch.cat(pooled_k_list, dim=0)  # [G_k, kv_heads, D]
        G_k = k_pooled.shape[0]
        H = self._num_kv_heads
        t_k_collect = time.perf_counter()
        self._prof_k_collect += t_k_collect - t_start
        nvtx.pop_range()

        # 3. Compute cosine similarity: [H, G_q, G_k]
        #    Then AGGREGATE across Q sub-blocks (mean) to get [H, G_k]
        #    This avoids the problem where union of per-Q top-p sets covers everything.
        nvtx.push_range("compass_matmul", color="blue")
        q_t = q_pooled.permute(1, 0, 2)  # [H, G_q, D]
        k_t = k_pooled.permute(1, 2, 0)  # [H, D, G_k]
        scores = torch.bmm(q_t, k_t)     # [H, G_q, G_k]
        avg_scores = scores.mean(dim=1)   # [H, G_k]  — average over Q groups
        probs = torch.softmax(avg_scores, dim=-1)  # [H, G_k]
        t_matmul = time.perf_counter()
        self._prof_matmul += t_matmul - t_k_collect
        nvtx.pop_range()

        # 4. Vectorized top-p selection on averaged probabilities
        nvtx.push_range("compass_topp", color="red")
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

        # Per-head selection: keep per-head mask, do NOT union
        t_topp = time.perf_counter()
        self._prof_topp += t_topp - t_matmul
        nvtx.pop_range()

        # Stats: per-head selected sub-blocks
        nvtx.push_range("compass_mask_build", color="yellow")
        per_head_counts = selected_mask.sum(dim=1)  # [H]
        overall_selected = selected_mask.any(dim=0)  # [G_k] for IO/stats
        total_sub = G_k * H  # total = G_k per head × H heads
        selected_sub = int(per_head_counts.sum().item())
        self._stats_total_subblocks += total_sub
        self._stats_selected_subblocks += selected_sub

        t_mask = time.perf_counter()
        self._prof_mask_build += t_mask - t_topp
        nvtx.pop_range()

        return selected_mask, subblock_to_block  # [H, G_k], [(bid, si)]

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
        nvtx.push_range("compass_cuda_sync", color="gray")
        t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        self._prof_sync += t1 - t0
        nvtx.pop_range()

        # 2. GPU-side Q pooling + GQA fold (fast on GPU)
        nvtx.push_range("compass_q_pool_gpu", color="magenta")
        q_pooled_gpu = self._pool_and_normalize(q, self.FINE_GRAIN)
        G_q = q_pooled_gpu.shape[0]
        # GQA fold on GPU: [G_q, H, D] → [G_q, H_kv, hpg, D] → mean → [G_q, H_kv, D]
        q_pooled_gpu = q_pooled_gpu.reshape(G_q, self._num_kv_heads, heads_per_group, self._head_dim)
        q_pooled_gpu = q_pooled_gpu.mean(dim=2)  # [G_q, H_kv, D]
        t2 = time.perf_counter()
        self._prof_q_pool += t2 - t1
        nvtx.pop_range()

        # 3. Async transfer pooled Q to pinned CPU buffer (~128KB vs 32MB)
        nvtx.push_range("compass_q_d2h", color="cyan")
        with torch.cuda.stream(self._metadata_stream):
            self._q_pooled_cpu_buf[:G_q].copy_(q_pooled_gpu, non_blocking=True)
        self._metadata_stream.synchronize()
        q_pooled_cpu = self._q_pooled_cpu_buf[:G_q].clone()  # snapshot
        t3 = time.perf_counter()
        self._prof_q_cpu += t3 - t2
        nvtx.pop_range()

        # 4. CPU-side estimation using pre-pooled Q
        result = self._estimate_subblock_mask(
            layer_id=ctx.layer_id,
            q_pooled=q_pooled_cpu,
            available_blocks=available_blocks,
            block_size=ctx.block_size,
        )
        selected_mask, subblock_to_block = result  # [H, G_k], [(bid, si)]
        H = self._num_kv_heads
        G_k = selected_mask.shape[1]
        t4 = time.perf_counter()

        nvtx.push_range("compass_build_entries", color="purple")
        # Build per-head selections
        per_head_entries = [[] for _ in range(H)]  # [H] -> [(bid, [si...])]
        per_head_grouped = [{} for _ in range(H)]  # [H] -> {bid: [si...]}
        for gk in range(G_k):
            bid, si = subblock_to_block[gk]
            for h in range(H):
                if selected_mask[h, gk]:
                    per_head_grouped[h].setdefault(bid, []).append(si)
        for h in range(H):
            per_head_entries[h] = list(per_head_grouped[h].items())

        selection = PerHeadSubBlockSelection(
            per_head_entries=per_head_entries,
            sub_block_size=self.FINE_GRAIN,
            num_kv_heads=H,
        )
        self._compacted_selections[ctx.layer_id] = selection
        nvtx.pop_range()

        nvtx.push_range("compass_log_io", color="brown")
        # Log per-head statistics
        per_head_counts = selection.per_head_num_subblocks
        total_ph = sum(per_head_counts)
        # Union for IO block list
        overall_selected = selected_mask.any(dim=0)
        union_count = int(overall_selected.sum().item())

        head_strs = ", ".join(f"H{h}:{per_head_counts[h]}" for h in range(H))
        io_density = (union_count / G_k * 100) if G_k > 0 else 0
        compute_density = (total_ph / (G_k * H) * 100) if G_k > 0 else 0
        logger.info(
            f"[COMPASS] layer={ctx.layer_id}, seq_chunk={ctx.query_chunk_idx}: "
            f"IO_density={io_density:.1f}% ({union_count}/{G_k}), "
            f"Compute_density={compute_density:.1f}% ({total_ph}/{G_k*H}), "
            f"per-head: [{head_strs}]"
        )

        # IO blocks: any block with at least one head selecting a sub-block
        io_blocks = set()
        for h_entries in per_head_entries:
            for bid, _ in h_entries:
                io_blocks.add(bid)
        io_blocks = sorted(io_blocks)
        nvtx.pop_range()

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
        """Compute attention with per-head sub-block gather + BLASST dynamic pruning.

        Pipeline: CPU gather → staging → H2D → GPU BLASST (per KV head) → merge.
        """
        from nanovllm.ops.chunked_attention import (
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )
        from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill

        ln_lambda = math.log(self.lambda_threshold)
        q_len, num_heads = q.shape[0], q.shape[1]
        compute_stream = offload_engine.compute_stream
        q_input = q.unsqueeze(0).transpose(1, 2).contiguous()  # [1, H, q_len, D]

        historical_o = None
        historical_lse = None
        historical_m_global = None

        # ---- Per-head historical block processing ----
        selection = self._compacted_selections.get(layer_id)

        if (selection is not None
                and isinstance(selection, PerHeadSubBlockSelection)
                and selection.total_subblocks > 0):
            fine_grain = selection.sub_block_size
            kv_heads = selection.num_kv_heads
            gqa_ratio = num_heads // kv_heads
            block_size = kvcache_manager.block_size
            max_subs_per_batch = block_size // fine_grain

            nvtx.push_range("compass_v2_regroup", color="purple")
            # 1. Collect all unique selected block IDs across all heads
            unique_bids = set()
            for h in range(selection.num_kv_heads):
                for bid, _ in selection.per_head_entries[h]:
                    unique_bids.add(bid)
            unique_bids = sorted(list(unique_bids))

            # 2. Divide blocks into chunks (e.g., 2 blocks = 8192 tokens max per chunk)
            blocks_per_chunk = 2
            bid_chunks = [unique_bids[i:i + blocks_per_chunk] for i in range(0, len(unique_bids), blocks_per_chunk)]
            nvtx.pop_range()

            nvtx.push_range("compass_prefill_log", color="brown")
            logger.info(
                f"[COMPASS] layer={layer_id}, seq_chunk={current_chunk_idx}: "
                f"total {selection.total_subblocks} sub-blocks "
                f"-> pipelined into {len(bid_chunks)} pieces (max {blocks_per_chunk} blks/piece) "
                f"for overlap"
            )
            nvtx.pop_range()

            # 3. Intra-layer Pipelining: Loop through chunks
            for chunk_iter_idx, active_bids in enumerate(bid_chunks):
                active_bids_set = set(active_bids)

                grouped_per_head = []
                for h in range(selection.num_kv_heads):
                    flat_subs = [
                        (bid, si)
                        for bid, sub_indices in selection.per_head_entries[h]
                        if bid in active_bids_set
                        for si in sub_indices
                    ]
                    grouped: Dict[int, list] = {}
                    for bid, si in flat_subs:
                        grouped.setdefault(bid, []).append(si)
                    grouped_per_head.append(list(grouped.items()))

                # Multi-buffer slot assignment for intra-layer ping-pong overlap
                slot_idx = chunk_iter_idx % offload_engine.num_jagged_slots

                nvtx.push_range(f"COMPASS L{layer_id}: jagged_slot_sync", color="yellow")
                # Wait for previous computation on this slot to complete before overwriting staging buffers
                offload_engine.jagged_compute_done[slot_idx].synchronize()
                nvtx.pop_range()

                nvtx.push_range(f"COMPASS L{layer_id}: V2 Packed Gather", color="orange")
                max_tok, kv_indptr_cpu = offload_engine.gather_packed_subblocks_all_heads(
                    layer_id, grouped_per_head, fine_grain, slot_idx=slot_idx
                )
                if max_tok == 0:
                    nvtx.pop_range()
                    continue
                nvtx.pop_range()

                transfer_stream = offload_engine.slot_transfer_streams[slot_idx]
                with torch.cuda.stream(transfer_stream):
                    nvtx.push_range(f"COMPASS L{layer_id}: V2 H2D Packed", color="green")
                    
                    # Must copy offsets tensor to GPU
                    kv_indptr_gpu = kv_indptr_cpu.to(q.device, non_blocking=True)
                    
                    offload_engine.load_packed_staging_to_jagged_gpu(
                        total_tokens=max_tok, stream=transfer_stream, slot_idx=slot_idx
                    )
                    
                    offload_engine.ring_slot_ready[slot_idx].record(transfer_stream)
                    nvtx.pop_range()

                from nanovllm.ops.compass import compass_jagged_chunked_prefill
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(offload_engine.ring_slot_ready[slot_idx])
                    
                    nvtx.push_range(f"COMPASS L{layer_id}: V2 Jagged Kernel", color="blue")
                    
                    # Fetch as 1D from dedicated jagged buffers for this slot
                    k_packed = offload_engine.jagged_k_gpu[slot_idx]
                    v_packed = offload_engine.jagged_v_gpu[slot_idx]
                    
                    sm_scale = 1.0 / math.sqrt(self._head_dim)
                    o, lse = compass_jagged_chunked_prefill(
                        q.unsqueeze(0), k_packed, v_packed, kv_indptr_gpu, 
                        sm_scale=sm_scale, causal=False
                    )
                    
                    if historical_o is None:
                        historical_o = o
                        historical_lse = lse
                    else:
                        historical_o, historical_lse = merge_attention_outputs(
                            historical_o, historical_lse, o, lse
                        )
                    
                    # Record completion of the GPU kernel so CPU knows when it's safe to overwrite the slot
                    offload_engine.jagged_compute_done[slot_idx].record(compute_stream)
                    
                    nvtx.pop_range()
                    
            # m_global was used in V1 batching across seqlen loop. V2 processes all tokens simultaneously without slicing seqs.
            historical_m_global = None
            
        # ---- Current prefill chunk (causal) ----
        nvtx.push_range(f"COMPASS L{layer_id}: current_chunk_causal {num_tokens}tok", color="cyan")
        with torch.cuda.stream(compute_stream):
            # No concat necessary: historical_o and historical_lse are natively shaped [1, q_len, H, D] and [1, H, q_len]

            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            k_curr_input = k_curr.transpose(1, 2).contiguous()
            v_curr_input = v_curr.transpose(1, 2).contiguous()

            # kv_offset for causal masking
            if selection is not None and isinstance(selection, PerHeadSubBlockSelection):
                kv_offset = max(selection.per_head_tokens) if selection.per_head_tokens else 0
                # historical_m_global is established as None from the jagged logic skip
            else:
                kv_offset = len(selected_blocks) * kvcache_manager.block_size

            out_curr, lse_curr, _ = blasst_chunked_prefill(
                q=q_input, k=k_curr_input, v=v_curr_input,
                threshold_ln_lambda=ln_lambda,
                m_global_in=historical_m_global,
                is_causal=True, kv_offset=kv_offset,
            )
            compute_stream.synchronize()

            curr_o = out_curr.transpose(1, 2).contiguous()
            if historical_o is None:
                final_o = curr_o
            else:
                nvtx.push_range(f"COMPASS L{layer_id}: merge_final", color="purple")
                final_o, _ = merge_attention_outputs(
                    historical_o, historical_lse, curr_o, lse_curr
                )
                nvtx.pop_range()

        torch.cuda.default_stream().wait_stream(compute_stream)
        nvtx.pop_range()
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



    def __repr__(self) -> str:
        return f"COMPASSPolicy(top_p={self.top_p}, lambda={self.lambda_threshold})"
