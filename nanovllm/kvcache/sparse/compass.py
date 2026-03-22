"""
COMPASS sparse attention policy for chunked prefill.

Two-tier sparse attention pipeline:
- Tier 1 (CPU, coarse): Pooled cosine similarity + softmax + top-p selection
  produces a block-sparse mask at 128-token sub-block granularity.
  Self-cosine similarity (SpargeAttention) detects blocks with high internal
  token variance and force-selects them, preventing over-pruning.
- Tier 2 (GPU, fine): BLASST dynamically prunes within the CPU-selected sub-blocks

Key design:
- IO granularity: 4096-token chunks (offload_engine block_size)
- Selection granularity: 128-token sub-blocks (FINE_GRAIN)
- CPU produces a sub-block mask, BLASST further prunes dynamically
- Self-cosine (SpargeAttention arXiv:2502.18137): CosSim(X) = sum(gram(L2norm(X))) / BS^2
  If CosSim < theta, block tokens are diverse, mean pool is unreliable, force-select.
"""

import logging
import math
import time
import torch
import torch.nn.functional as F
import nvtx
from typing import List, Optional, Dict, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext, SubBlockSelection, PerHeadSubBlockSelection, TensorSelection

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
        theta: float = 0.6,
        **kwargs,
    ):
        self._stats_num_chunks = 0
        self._stats_selected_subblocks = 0
        self._stats_total_subblocks = 0
        self._stats_force_selected_k = 0
        self._stats_force_selected_q_heads = 0
        self._stats_persistent_k_blocks = 0
        self._stats_persistent_k_total = 0
        self.lambda_threshold = lambda_threshold
        self.top_p = top_p
        self.theta = theta
        
        logger.info(f"[COMPASS] Initialized policy with top_p={self.top_p}, lambda={self.lambda_threshold}, theta={self.theta}")

        # Model dimensions (set during alloc_policy_metadata)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        # Q buffer for metadata tracking
        self._q_buffer: Optional[torch.Tensor] = None
        self._q_chunk_sizes: list[int] = []

        # Pooled K cache: layer_id -> {cpu_block_id -> pooled_k tensor [G_k, H_kv, D]}
        self._k_pooled_cache: Dict[int, Dict[int, torch.Tensor]] = {}

        # Per-head sub-block selections: layer_id -> TensorSelection
        self._compacted_selections: Dict[int, SubBlockSelection] = {}

        # Profiling accumulators
        self._prof_sync = 0.0
        self._prof_q_cpu = 0.0
        self._prof_q_pool = 0.0
        self._prof_k_collect = 0.0
        self._prof_matmul = 0.0
        self._prof_topp = 0.0
        self._prof_mask_build = 0.0
        self._prof_self_cos = 0.0
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

        # Pre-allocate mask_buffer for L2 density tracking (avoids per-piece torch.ones)
        # grid_0 = max Q blocks, max_kv_blocks = blocks_per_chunk * subs_per_block
        BLOCK_M = 64
        max_grid_0 = (max_seq_len + BLOCK_M - 1) // BLOCK_M
        blocks_per_chunk = 2  # must match compute_chunked_prefill
        max_kv_blocks_per_piece = blocks_per_chunk * (4096 // self.FINE_GRAIN)
        self._mask_buffer = torch.empty(
            (max_grid_0, num_heads, max_kv_blocks_per_piece),
            device=device, dtype=torch.int8
        )
        mask_kb = self._mask_buffer.numel() / 1024
        logger.info(f"[COMPASS] Pre-allocated mask_buffer: {mask_kb:.1f} KB (GPU)")

        # Pre-allocate TensorSelection buffers (tiny: ~2KB)
        block_size_tokens = 4096  # offload_engine block_size
        max_blocks = max_seq_len // block_size_tokens + 1
        subs_per_block = block_size_tokens // self.FINE_GRAIN  # 32
        self._sel_mask = torch.zeros(
            (num_kv_heads, max_blocks, subs_per_block), dtype=torch.bool, device='cpu'
        )
        self._sel_block_ids = torch.zeros(max_blocks, dtype=torch.int32, device='cpu')
        sel_bytes = self._sel_mask.numel() + self._sel_block_ids.numel() * 4
        logger.info(f"[COMPASS] Pre-allocated TensorSelection buffers: {sel_bytes} bytes (CPU)")

        # Pre-allocated pinned CPU buffers for async K pooled DtoH
        max_k_groups = block_size_tokens // self.FINE_GRAIN  # 32 sub-blocks per block
        self._k_pooled_buf = torch.zeros(
            (num_layers, max_blocks, max_k_groups, num_kv_heads, head_dim),
            dtype=torch.float32, device='cpu',
        ).pin_memory()
        self._k_self_cos_buf = torch.zeros(
            (num_layers, max_blocks, max_k_groups, num_kv_heads),
            dtype=torch.float32, device='cpu',
        ).pin_memory()
        k_pool_mb = self._k_pooled_buf.numel() * 4 / (1024 * 1024)
        k_cos_kb = self._k_self_cos_buf.numel() * 4 / 1024
        logger.info(f"[COMPASS] Allocated K pooled buffer: {k_pool_mb:.1f} MB (Pinned CPU)")
        logger.info(f"[COMPASS] Allocated K self-cos buffer: {k_cos_kb:.1f} KB (Pinned CPU)")

        # Pre-allocated pinned CPU buffer for Q self-cosine DtoH
        self._q_self_cos_buf = torch.zeros(
            (max_q_groups, num_kv_heads),
            dtype=torch.float32, device='cpu',
        ).pin_memory()

        # Event for tracking async K pooled DtoH completion
        self._k_dtoh_event = None

        self._k_pooled_cache = {lid: {} for lid in range(num_layers)}
        self._q_chunk_sizes = []

    # ========================================================================
    # Pooling & Estimation Helpers
    # ========================================================================

    @staticmethod
    def _mean_pool(
        x: torch.Tensor,
        pool_size: int = 128,
    ) -> torch.Tensor:
        """Mean pool tokens into groups (no L2 normalization).

        Following SpargeAttn: raw mean pooling preserves scale symmetry
        between Q and K.  Scaling is applied later as * d^{-0.5}.

        Args:
            x: [seq_len, heads, dim] tensor.
            pool_size: Number of tokens per pool group.

        Returns:
            [num_groups, heads, dim] mean-pooled tensor in FP32.
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

        return pooled

    @staticmethod
    def _compute_self_cosine(
        x: torch.Tensor,
        pool_size: int = 128,
    ) -> torch.Tensor:
        """Compute self-cosine similarity per sub-block (SpargeAttention).

        Measures how similar tokens within a sub-block are to each other.
        High value = tokens are similar, mean pooling is accurate.
        Low value = tokens are diverse, mean pooling is unreliable.

        Algorithm (from SpargeAttn Triton kernel triton_bmm_pool_sim_simmean):
            x_norm = L2_normalize(x, dim=-1)   # normalize each token row
            gram = x_norm @ x_norm^T            # [pool_size, pool_size]
            self_cos = sum(gram) / (pool_size^2)

        Args:
            x: [seq_len, heads, dim] tensor.
            pool_size: Number of tokens per pool group.

        Returns:
            [num_groups, heads] self-cosine values in FP32.
        """
        seq_len, heads, dim = x.shape
        num_groups = seq_len // pool_size
        if num_groups == 0:
            # Single group: always treat as homogeneous (high self-cosine)
            return torch.ones(1, heads, dtype=torch.float32, device=x.device)

        aligned = num_groups * pool_size
        x_aligned = x[:aligned].float()  # [aligned, heads, dim]
        x_grouped = x_aligned.reshape(num_groups, pool_size, heads, dim)
        # Permute to [num_groups, heads, pool_size, dim]
        x_grouped = x_grouped.permute(0, 2, 1, 3)
        # Reshape to [num_groups * heads, pool_size, dim] for batched matmul
        GH = num_groups * heads
        x_flat = x_grouped.reshape(GH, pool_size, dim)
        # L2 normalize each token row
        x_norm = F.normalize(x_flat, p=2, dim=-1)  # [GH, pool_size, dim]
        # Gram matrix: [GH, pool_size, pool_size]
        gram = torch.bmm(x_norm, x_norm.transpose(1, 2))
        # Self-cosine: mean of all gram elements per (group, head)
        self_cos = gram.sum(dim=(1, 2)) / (pool_size * pool_size)  # [GH]
        # Reshape back to [num_groups, heads]
        self_cos = self_cos.reshape(num_groups, heads)
        return self_cos

    def _precompute_pooled_k_gpu(
        self,
        k_cache_gpu: torch.Tensor,
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
    ) -> None:
        """Compute pooled K and self-cosine on GPU, async DtoH via _metadata_stream.

        Called from on_prefill_offload BEFORE GPU→CPU KV copy.
        GPU compute runs on current stream, DtoH runs on _metadata_stream.
        Consumers must wait on _k_dtoh_event before reading the data.
        """
        # k_cache_gpu: [kv_heads, max_tokens, head_dim] on GPU if Head-First
        if k_cache_gpu.shape[0] == self._num_kv_heads:
            k_block = k_cache_gpu[:, :num_tokens, :].transpose(0, 1)
        else:
            k_block = k_cache_gpu[:num_tokens]
        # Compute on GPU (current stream): mean pool (no L2 normalize, following SpargeAttn)
        pooled_k_gpu = self._mean_pool(k_block, self.FINE_GRAIN)
        # Compute self-cosine similarity per K sub-block on GPU (current stream)
        self_cos_k_gpu = self._compute_self_cosine(k_block, self.FINE_GRAIN)
        G_k = pooled_k_gpu.shape[0]

        # Async DtoH via _metadata_stream (non-blocking, no CPU stall)
        with torch.cuda.stream(self._metadata_stream):
            self._metadata_stream.wait_stream(torch.cuda.current_stream())
            self._k_pooled_buf[layer_id, cpu_block_id, :G_k].copy_(
                pooled_k_gpu, non_blocking=True
            )
            self._k_self_cos_buf[layer_id, cpu_block_id, :G_k].copy_(
                self_cos_k_gpu, non_blocking=True
            )
        # Record event so select_blocks can precisely wait for this DtoH
        self._k_dtoh_event = self._metadata_stream.record_event()

        # Store pinned buffer views in cache (data valid after event sync)
        self._k_pooled_cache[layer_id][cpu_block_id] = (
            self._k_pooled_buf[layer_id, cpu_block_id, :G_k],
            self._k_self_cos_buf[layer_id, cpu_block_id, :G_k],
        )

    # ========================================================================
    # CPU Pooled Estimation (128-token sub-block granularity)
    # ========================================================================

    def _estimate_subblock_mask(
        self,
        layer_id: int,
        q_pooled: torch.Tensor,
        available_blocks: list,
        block_size: int,
        q_self_cos: torch.Tensor = None,
    ):
        """Estimate which 128-token sub-blocks to keep via pooled cosine sim + top-p.

        SpargeAttention self-cosine override:
        - K blocks with self-cosine < theta are force-selected (diverse tokens)
        - Q groups with self-cosine < theta force-select all K blocks for that head

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

        # 2. Collect pooled K and K self-cosine
        nvtx.push_range("compass_k_collect", color="green")
        t_start = time.perf_counter()
        pooled_k_list = []
        self_cos_k_list = []
        subblock_to_block = []  # Maps global sub-block idx -> (block_id, local_sub_idx)

        for bid in available_blocks:
            if bid not in self._k_pooled_cache.get(layer_id, {}):
                logger.warning(f"[COMPASS] Pooled K cache miss for layer={layer_id}, block={bid}")
                nvtx.pop_range()
                return {bid: torch.ones(num_fine_per_block, dtype=torch.bool)
                        for bid in available_blocks}

            cache_entry = self._k_pooled_cache[layer_id][bid]
            # Support tuple format (pooled_k, self_cos_k)
            if isinstance(cache_entry, tuple):
                pk, sc_k = cache_entry
            else:
                pk = cache_entry
                sc_k = None
            pooled_k_list.append(pk)
            if sc_k is not None:
                self_cos_k_list.append(sc_k)
            for si in range(pk.shape[0]):
                subblock_to_block.append((bid, si))

        if not pooled_k_list:
            nvtx.pop_range()
            return {}

        k_pooled = torch.cat(pooled_k_list, dim=0)  # [G_k, kv_heads, D]
        k_self_cos = torch.cat(self_cos_k_list, dim=0) if self_cos_k_list else None  # [G_k, kv_heads]
        G_k = k_pooled.shape[0]
        H = self._num_kv_heads
        t_k_collect = time.perf_counter()
        self._prof_k_collect += t_k_collect - t_start
        nvtx.pop_range()

        # 3. Compute scaled dot-product: [H, G_q, G_k]
        #    Following SpargeAttn: scores = Q_pool @ K_pool^T * d^{-0.5}
        nvtx.push_range("compass_matmul", color="blue")
        q_t = q_pooled.permute(1, 0, 2)  # [H, G_q, D]
        k_t = k_pooled.permute(1, 2, 0)  # [H, D, G_k]
        scores = torch.bmm(q_t, k_t) * (self._head_dim ** -0.5)  # [H, G_q, G_k]

        # SpargeAttn Step 1: mask diverse K blocks BEFORE softmax (L337)
        #   sim_k = True means homogeneous (high self-cos >= theta)
        #   ~sim_k = diverse → set to -inf so they don't consume top_p budget
        #   They get force-selected separately later.
        diverse_k = None
        if k_self_cos is not None and self.theta is not None:
            # k_self_cos: [G_k, H] → .t() → [H, G_k]
            diverse_k = (k_self_cos.t() < self.theta)  # [H, G_k] True=diverse
            # Expand to [H, G_q, G_k] and mask
            scores = scores.masked_fill(diverse_k.unsqueeze(1), -float('inf'))

        # SpargeAttn Step 2: per-Q-block softmax (L344)
        probs = torch.softmax(scores, dim=-1)  # [H, G_q, G_k]

        t_matmul = time.perf_counter()
        self._prof_matmul += t_matmul - t_k_collect
        nvtx.pop_range()

        # SpargeAttn Step 3: per-Q-block top_p via cumsum (L345-351)
        nvtx.push_range("compass_topp", color="red")
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)  # [H, G_q, G_k]

        # searchsorted: find how many blocks needed per (H, G_q) to reach top_p
        # cdf is [H, G_q, G_k], we need searchsorted along last dim
        top_p_tensor = torch.tensor(self.top_p, device=cdf.device, dtype=cdf.dtype)
        # Reshape for searchsorted: [H*G_q, G_k]
        cdf_flat = cdf.reshape(-1, G_k)
        num_to_select = torch.searchsorted(cdf_flat, top_p_tensor.expand(cdf_flat.shape[0], 1), right=True)
        num_to_select = (num_to_select.squeeze(-1) + 1).clamp(min=1)  # at least 1, +1 for right boundary
        num_to_select = num_to_select.reshape(H, G_q)  # [H, G_q]

        # SpargeAttn Step 4: build final_map [H, G_q, G_k] (L355-358)
        final_map = torch.zeros(H, G_q, G_k, dtype=torch.bool, device=probs.device)

        # Force-select diverse K blocks for ALL Q blocks (SpargeAttn L356)
        persistent_k_io = 0
        if diverse_k is not None:
            final_map |= diverse_k.unsqueeze(1)  # broadcast [H, 1, G_k] → [H, G_q, G_k]
            force_k_count = int(diverse_k.sum().item())
            self._stats_force_selected_k += force_k_count
            persistent_k_io = int(diverse_k.any(dim=0).sum().item())
            self._stats_persistent_k_blocks += persistent_k_io
            self._stats_persistent_k_total += G_k

        # Force-select all K for diverse Q blocks (SpargeAttn L357)
        if q_self_cos is not None and self.theta is not None:
            # q_self_cos: [G_q, H] → diverse_q: [H, G_q] True=diverse
            diverse_q = (q_self_cos < self.theta).t()  # [H, G_q]
            # For each (h, gq) where diverse_q is True → all K selected
            diverse_q_expanded = diverse_q.unsqueeze(-1).expand_as(final_map)  # [H, G_q, G_k]
            final_map |= diverse_q_expanded
            self._stats_force_selected_q_heads += int(diverse_q.sum().item())

        # Scatter top_p selected blocks into final_map (SpargeAttn L358)
        # For each (h, gq), select the top num_to_select[h, gq] blocks
        for h in range(H):
            for gq in range(G_q):
                n = int(num_to_select[h, gq].item())
                n = min(n, G_k)
                top_indices = sorted_indices[h, gq, :n]
                final_map[h, gq, top_indices] = True

        # SpargeAttn Step 5: attention sink (L362-363)
        # Always select block 0 (first sub-block in first available block)
        final_map[:, :, 0] = True

        # SpargeAttn Step 6: union across Q → per-K mask [H, G_k]
        selected_mask = final_map.any(dim=1)  # [H, G_k]

        t_topp = time.perf_counter()
        self._prof_topp += t_topp - t_matmul
        nvtx.pop_range()

        # Stats: per-head selected sub-blocks
        nvtx.push_range("compass_mask_build", color="yellow")
        per_head_counts = selected_mask.sum(dim=1)  # [H]
        overall_selected = selected_mask.any(dim=0)  # [G_k] for IO/stats
        total_sub = G_k * H
        selected_sub = int(per_head_counts.sum().item())
        self._stats_total_subblocks += total_sub
        self._stats_selected_subblocks += selected_sub

        t_mask = time.perf_counter()
        self._prof_mask_build += t_mask - t_topp
        nvtx.pop_range()

        return selected_mask, subblock_to_block, persistent_k_io  # [H, G_k], [(bid, si)], int

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

        # 1. Wait for previous K pooled DtoH on _metadata_stream (precise event wait)
        nvtx.push_range("compass_k_dtoh_wait", color="gray")
        t0 = time.perf_counter()
        if self._k_dtoh_event is not None:
            self._k_dtoh_event.synchronize()  # Only wait for K DtoH, not entire stream
        t1 = time.perf_counter()
        self._prof_sync += t1 - t0
        nvtx.pop_range()

        # 2. GPU-side Q pooling + GQA fold (fast on GPU)
        nvtx.push_range("compass_q_pool_gpu", color="magenta")
        q_pooled_gpu = self._mean_pool(q, self.FINE_GRAIN)
        G_q = q_pooled_gpu.shape[0]
        # GQA fold on GPU: [G_q, H, D] -> [G_q, H_kv, hpg, D] -> mean -> [G_q, H_kv, D]
        q_pooled_gpu = q_pooled_gpu.reshape(G_q, self._num_kv_heads, heads_per_group, self._head_dim)
        q_pooled_gpu = q_pooled_gpu.mean(dim=2)  # [G_q, H_kv, D]
        t2 = time.perf_counter()
        self._prof_q_pool += t2 - t1
        nvtx.pop_range()

        # 2b. GPU-side Q self-cosine similarity (SpargeAttention)
        nvtx.push_range("compass_q_self_cos", color="olive")
        q_self_cos_gpu = self._compute_self_cosine(q, self.FINE_GRAIN)  # [G_q, H]
        # GQA fold self-cosine: [G_q, H] -> [G_q, H_kv] via mean over head groups
        q_self_cos_gpu = q_self_cos_gpu.reshape(G_q, self._num_kv_heads, heads_per_group).mean(dim=2)
        t2b = time.perf_counter()
        self._prof_self_cos += t2b - t2
        nvtx.pop_range()

        # 3. Async transfer pooled Q + Q self-cos to pinned CPU buffers
        nvtx.push_range("compass_q_d2h", color="cyan")
        with torch.cuda.stream(self._metadata_stream):
            self._metadata_stream.wait_stream(torch.cuda.current_stream())
            self._q_pooled_cpu_buf[:G_q].copy_(q_pooled_gpu, non_blocking=True)
            self._q_self_cos_buf[:G_q].copy_(q_self_cos_gpu, non_blocking=True)
        self._metadata_stream.synchronize()
        q_pooled_cpu = self._q_pooled_cpu_buf[:G_q].clone()  # snapshot
        q_self_cos_cpu = self._q_self_cos_buf[:G_q]  # pinned buf, valid after sync
        t3 = time.perf_counter()
        self._prof_q_cpu += t3 - t2b
        nvtx.pop_range()

        # 4. CPU-side estimation using pre-pooled Q + self-cosine
        result = self._estimate_subblock_mask(
            layer_id=ctx.layer_id,
            q_pooled=q_pooled_cpu,
            available_blocks=available_blocks,
            block_size=ctx.block_size,
            q_self_cos=q_self_cos_cpu,
        )
        selected_mask, subblock_to_block, persistent_k_io = result  # [H, G_k], [(bid, si)], int
        H = self._num_kv_heads
        G_k = selected_mask.shape[1]
        t4 = time.perf_counter()

        nvtx.push_range("compass_build_entries", color="purple")
        # Vectorized build: O(1) tensor reshape replaces O(G_k × H) Python loop
        num_blocks = len(available_blocks)
        subs_per_block = ctx.block_size // self.FINE_GRAIN
        mask_3d = selected_mask.view(H, num_blocks, subs_per_block)
        self._sel_mask[:H, :num_blocks, :subs_per_block] = mask_3d
        for i, bid in enumerate(available_blocks):  # tiny loop: ~7 iters
            self._sel_block_ids[i] = bid

        selection = TensorSelection(
            mask=self._sel_mask,
            block_ids=self._sel_block_ids,
            num_valid_blocks=num_blocks,
            sub_block_size=self.FINE_GRAIN,
            num_kv_heads=H,
            subs_per_block=subs_per_block,
        )
        self._compacted_selections[ctx.layer_id] = selection
        nvtx.pop_range()

        nvtx.push_range("compass_log_io", color="brown")
        # Vectorized stats: tensor ops instead of Python iteration
        per_head_counts = selection.per_head_num_subblocks  # List[int]
        total_ph = sum(per_head_counts)
        # Union for IO: any head selected any sub-block in each block
        any_selected_per_block = mask_3d.any(dim=(0, 2))  # [num_blocks] bool
        union_count = int(any_selected_per_block.sum().item())

        head_strs = ", ".join(f"H{h}:{per_head_counts[h]/G_k*100:.1f}%" for h in range(H))
        io_density = (union_count / num_blocks * 100) if num_blocks > 0 else 0
        l1_density = (total_ph / (G_k * H) * 100) if G_k > 0 else 0
        persistent_ratio = (persistent_k_io / num_blocks * 100) if num_blocks > 0 else 0
        logger.info(
            f"[COMPASS] layer={ctx.layer_id}, seq_chunk={ctx.query_chunk_idx}: "
            f"IO_density={io_density:.1f}% ({union_count}/{num_blocks}), "
            f"L1_density={l1_density:.1f}% ({total_ph}/{G_k*H}), "
            f"Persistent_K={persistent_ratio:.1f}% ({persistent_k_io}/{num_blocks}), "
            f"per-head: [{head_strs}]"
        )

        # IO blocks: vectorized — blocks where any head has any selection
        io_block_indices = any_selected_per_block.nonzero(as_tuple=True)[0]
        io_blocks = self._sel_block_ids[io_block_indices].tolist()
        nvtx.pop_range()

        return io_blocks

    # ========================================================================
    # Statistics
    # ========================================================================

    def reset_stats(self) -> None:
        self._stats_num_chunks = 0
        self._stats_selected_subblocks = 0
        self._stats_total_subblocks = 0
        self._stats_force_selected_k = 0
        self._stats_force_selected_q_heads = 0
        self._stats_persistent_k_blocks = 0
        self._stats_persistent_k_total = 0
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
            "prof_self_cos": self._prof_self_cos,
            "force_selected_k": self._stats_force_selected_k,
            "force_selected_q_heads": self._stats_force_selected_q_heads,
            "persistent_k_blocks": self._stats_persistent_k_blocks,
            "persistent_k_total": self._stats_persistent_k_total,
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
                and isinstance(selection, TensorSelection)
                and selection.total_subblocks > 0):
            fine_grain = selection.sub_block_size
            kv_heads = selection.num_kv_heads
            gqa_ratio = num_heads // kv_heads
            block_size = kvcache_manager.block_size
            max_subs_per_batch = block_size // fine_grain
            nv = selection.num_valid_blocks

            nvtx.push_range("compass_v2_regroup", color="purple")
            # 1. Vectorized unique block detection
            any_selected = selection.mask[:, :nv, :].any(dim=(0, 2))  # [nv] bool
            unique_bid_local = any_selected.nonzero(as_tuple=True)[0]  # indices into block_ids
            unique_bids_tensor = selection.block_ids[unique_bid_local]  # actual cpu_block_ids

            # 2. Divide blocks into chunks (e.g., 2 blocks = 8192 tokens max per chunk)
            blocks_per_chunk = 2
            num_unique = unique_bid_local.shape[0]
            bid_chunks_indices = [unique_bid_local[i:i + blocks_per_chunk]
                                  for i in range(0, num_unique, blocks_per_chunk)]
            nvtx.pop_range()

            # Compute density tracking
            TRITON_BLOCK_N = 64
            TRITON_BLOCK_M = 64
            grid_0 = (q_len + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
            total_full_compute_pairs = 0
            ph_tokens = selection.per_head_tokens
            for h in range(kv_heads):
                h_tokens = ph_tokens[h] if h < len(ph_tokens) else 0
                h_kv_blocks = (h_tokens + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
                total_full_compute_pairs += grid_0 * h_kv_blocks * gqa_ratio
            total_l2_computed_pairs = 0

            nvtx.push_range("compass_prefill_log", color="brown")
            logger.info(
                f"[COMPASS] layer={layer_id}, seq_chunk={current_chunk_idx}: "
                f"total {selection.total_subblocks} sub-blocks "
                f"-> pipelined into {len(bid_chunks_indices)} pieces (max {blocks_per_chunk} blks/piece) "
                f"for overlap"
            )
            nvtx.pop_range()

            # Deferred density tracking: collect per-piece data, process AFTER pipeline loop
            deferred_density_data = []  # [(mask_buffer, kv_indptr_cpu), ...]

            # 3. Intra-layer Pipelining: Loop through chunks
            for chunk_iter_idx, active_local_indices in enumerate(bid_chunks_indices):
                # Slice mask for active blocks: [H_kv, num_active, subs_per_block]
                active_mask = selection.mask[:, active_local_indices, :]
                active_block_ids = selection.block_ids[active_local_indices]

                # Multi-buffer slot assignment for intra-layer ping-pong overlap
                slot_idx = chunk_iter_idx % offload_engine.num_jagged_slots

                nvtx.push_range(f"COMPASS L{layer_id}: jagged_slot_sync", color="yellow")
                # Wait for previous computation on this slot to complete before overwriting staging buffers
                offload_engine.jagged_compute_done[slot_idx].synchronize()
                nvtx.pop_range()

                nvtx.push_range(f"COMPASS L{layer_id}: V2 Packed Gather", color="orange")
                max_tok, kv_indptr_cpu = offload_engine.gather_packed_from_mask(
                    layer_id, active_mask, active_block_ids, fine_grain, slot_idx=slot_idx
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

                # Reuse pre-allocated mask_buffer (avoid torch.ones per piece)
                max_kv_blocks = 0
                for h in range(kv_heads):
                    h_start = kv_indptr_cpu[h].item()
                    h_end = kv_indptr_cpu[h + 1].item()
                    h_blocks = (h_end - h_start + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
                    max_kv_blocks = max(max_kv_blocks, h_blocks)
                if max_kv_blocks > 0:
                    mask_buffer = self._mask_buffer[:grid_0, :num_heads, :max_kv_blocks]
                    mask_buffer.fill_(1)
                else:
                    mask_buffer = None

                from nanovllm.ops.compass import compass_jagged_chunked_prefill, merge_attention_inplace
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(offload_engine.ring_slot_ready[slot_idx])
                    
                    nvtx.push_range(f"COMPASS L{layer_id}: V2 Jagged Kernel", color="blue")
                    
                    # Fetch as 1D from dedicated jagged buffers for this slot
                    k_packed = offload_engine.jagged_k_gpu[slot_idx]
                    v_packed = offload_engine.jagged_v_gpu[slot_idx]
                    
                    sm_scale = 1.0 / math.sqrt(self._head_dim)

                    o, lse, m_global_out = compass_jagged_chunked_prefill(
                        q.unsqueeze(0), k_packed, v_packed, kv_indptr_gpu, 
                        sm_scale=sm_scale, causal=False,
                        threshold_ln_lambda=ln_lambda,
                        m_global_in=historical_m_global,
                        mask_buffer=mask_buffer,
                    )

                    # Update running m_global across pipeline pieces (GPU async op, no CPU sync)
                    if historical_m_global is None:
                        historical_m_global = m_global_out
                    else:
                        historical_m_global = torch.maximum(historical_m_global, m_global_out)

                    # Defer density calculation — NO .item() here to avoid breaking pipeline overlap
                    if mask_buffer is not None:
                        deferred_density_data.append((mask_buffer, kv_indptr_cpu))
                    
                    if historical_o is None:
                        historical_o = o
                        historical_lse = lse
                    else:
                        merge_attention_inplace(historical_o, historical_lse, o, lse)
                    
                    # Record completion of the GPU kernel so CPU knows when it's safe to overwrite the slot
                    offload_engine.jagged_compute_done[slot_idx].record(compute_stream)
                    
                    nvtx.pop_range()

            # Compute density AFTER pipeline loop (single sync point)
            if deferred_density_data and total_full_compute_pairs > 0:
                compute_stream.synchronize()  # single sync to ensure all mask_buffers are ready
                total_l2_computed_pairs = 0
                for mask_buf, indptr_cpu in deferred_density_data:
                    for h_kv in range(kv_heads):
                        h_start = indptr_cpu[h_kv].item()
                        h_end = indptr_cpu[h_kv + 1].item()
                        h_actual_blocks = (h_end - h_start + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
                        if h_actual_blocks > 0:
                            for g in range(gqa_ratio):
                                q_head = h_kv * gqa_ratio + g
                                total_l2_computed_pairs += int(
                                    mask_buf[:, q_head, :h_actual_blocks].sum().item()
                                )
                compute_density = total_l2_computed_pairs / total_full_compute_pairs * 100
                logger.info(
                    f"[COMPASS] layer={layer_id}, seq_chunk={current_chunk_idx}: "
                    f"Compute_density={compute_density:.1f}% "
                    f"({total_l2_computed_pairs}/{total_full_compute_pairs} compute-pairs after L1+L2)"
                )
            
        # ---- Current prefill chunk (causal) ----
        nvtx.push_range(f"COMPASS L{layer_id}: current_chunk_causal {num_tokens}tok", color="cyan")
        with torch.cuda.stream(compute_stream):
            # No concat necessary: historical_o and historical_lse are natively shaped [1, q_len, H, D] and [1, H, q_len]

            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            k_curr_input = k_curr.transpose(1, 2).contiguous()
            v_curr_input = v_curr.transpose(1, 2).contiguous()

            # kv_offset for causal masking
            if selection is not None and isinstance(selection, TensorSelection):
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
        return f"COMPASSPolicy(top_p={self.top_p}, lambda={self.lambda_threshold}, theta={self.theta})"
