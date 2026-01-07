"""
MInference sparse attention policy.

Implements vertical + slash sparse pattern estimation using the last 64 query tokens.
Reference: MInference paper (https://arxiv.org/abs/2407.02490)
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext


class MInferencePolicy(SparsePolicy):
    """
    MInference sparse prefill policy using vertical + slash pattern.

    This policy estimates sparse attention patterns by analyzing attention
    scores from the last 64 query tokens, then selects:
    - Vertical: Key positions that are important across all queries
    - Slash: Diagonal bands (local context)

    The estimated pattern is then used to compute sparse attention.

    Note: This policy is designed for GPU-only prefill. For CPU offload,
    the pattern estimation and sparse attention will be handled differently.
    """

    supports_prefill = True
    supports_decode = False  # MInference is prefill-only sparse strategy

    def __init__(
        self,
        vertical_size: int = 1000,
        slash_size: int = 6096,
        adaptive_budget: Optional[float] = 0.3,
        num_sink_tokens: int = 30,
        num_recent_diags: int = 100,
    ):
        """
        Initialize MInference policy.

        Args:
            vertical_size: Number of vertical (column) positions to keep
            slash_size: Number of diagonal bands to keep
            adaptive_budget: If set, compute budget as fraction of seq_len
                            (overrides vertical_size and slash_size)
            num_sink_tokens: Number of initial sink tokens to always keep
            num_recent_diags: Number of recent diagonals to always keep
        """
        self.vertical_size = vertical_size
        self.slash_size = slash_size
        self.adaptive_budget = adaptive_budget
        self.num_sink_tokens = num_sink_tokens
        self.num_recent_diags = num_recent_diags

        # Cache for last-q causal mask
        self._last_q_mask_cache: dict = {}

    def _get_causal_mask(self, last_q: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get causal mask for last-q attention."""
        cache_key = (last_q, seq_len, device)
        if cache_key not in self._last_q_mask_cache:
            # Create mask where last_q queries can attend to all previous positions
            # Shape: [last_q, seq_len]
            mask = torch.ones(last_q, seq_len, device=device, dtype=torch.bool)
            # Apply causal constraint for the last last_q positions
            # Query i (from last_q) can only attend to positions <= (seq_len - last_q + i)
            for i in range(last_q):
                mask[i, seq_len - last_q + i + 1:] = False
            self._last_q_mask_cache[cache_key] = mask
        return self._last_q_mask_cache[cache_key]

    def estimate_pattern(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate vertical + slash sparse pattern using last 64 query tokens.
        Memory-optimized for long sequences (64K+).

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            layer_id: Current layer index (for potential layer-specific patterns)

        Returns:
            Tuple of (vertical_indices, slash_indices):
            - vertical_indices: [num_heads, vertical_size] - important K positions
            - slash_indices: [num_heads, slash_size] - diagonal offsets
        """
        seq_len = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k.shape[1]

        # Adaptive budget
        if self.adaptive_budget is not None:
            budget = int(seq_len * self.adaptive_budget)
            vertical_size = max(self.num_sink_tokens + 1, int(budget * 0.2))
            slash_size = max(self.num_recent_diags + 1, int(budget * 0.8))
        else:
            vertical_size = self.vertical_size
            slash_size = self.slash_size

        # Use last 64 Q tokens for estimation
        last_q = min(64, seq_len)
        q_last = q[-last_q:]  # [last_q, heads, dim] - this is a view, not a copy

        # Handle GQA: if num_kv_heads < num_heads, we need to expand K
        if num_kv_heads < num_heads:
            num_groups = num_heads // num_kv_heads
            k_work = k.repeat_interleave(num_groups, dim=1)
        else:
            k_work = k

        # Compute attention scores: [heads, last_q, seq_len]
        scale = 1.0 / math.sqrt(head_dim)
        qk = torch.einsum('qhd,khd->hqk', q_last, k_work) * scale

        # Free k_work if it was a copy
        if num_kv_heads < num_heads:
            del k_work

        # Apply causal mask for last positions (in-place)
        causal_mask = self._get_causal_mask(last_q, seq_len, q.device)
        qk.masked_fill_(~causal_mask.unsqueeze(0), float('-inf'))

        # Softmax (in-place where possible)
        qk = F.softmax(qk, dim=-1, dtype=torch.float32)

        # === Vertical pattern ===
        # Sum across query dimension -> importance of each K position
        vertical_scores = qk.sum(dim=1)  # [heads, seq_len]

        # Force keep first num_sink_tokens (attention sinks) - in-place
        vertical_scores[:, :self.num_sink_tokens] = float('inf')

        # Select top-k
        actual_vertical = min(vertical_size, seq_len)
        vertical_indices = vertical_scores.topk(actual_vertical, dim=-1).indices
        vertical_indices = vertical_indices.sort(dim=-1).values
        del vertical_scores

        # === Slash pattern ===
        # Create diagonal index matrix: [last_q, seq_len] with int32 to save memory
        q_indices = torch.arange(last_q, device=q.device, dtype=torch.int32).unsqueeze(1)
        k_indices = torch.arange(seq_len, device=q.device, dtype=torch.int32).unsqueeze(0)
        diag_indices = (seq_len - last_q + q_indices) - k_indices  # [last_q, seq_len]
        del q_indices

        # Create causal mask for slash computation
        q_pos = seq_len - last_q + torch.arange(last_q, device=q.device, dtype=torch.int32).unsqueeze(1)
        slash_causal_mask = k_indices <= q_pos
        del q_pos, k_indices

        # Clamp diagonal indices to valid range
        diag_indices = diag_indices.clamp(0, seq_len - 1)

        # Apply causal mask to qk (in-place) for slash computation
        qk[:, ~slash_causal_mask] = 0
        del slash_causal_mask

        # Accumulate scores per diagonal - process in batches to save memory
        slash_scores = torch.zeros(num_heads, seq_len, device=q.device, dtype=torch.float32)

        # Process heads in chunks to reduce peak memory for diag_indices_expanded
        chunk_size = min(8, num_heads)  # Process 8 heads at a time
        for h_start in range(0, num_heads, chunk_size):
            h_end = min(h_start + chunk_size, num_heads)
            n_heads_chunk = h_end - h_start

            # Expand diag_indices only for this chunk
            diag_chunk = diag_indices.unsqueeze(0).expand(n_heads_chunk, -1, -1).long()
            qk_chunk = qk[h_start:h_end]

            slash_scores[h_start:h_end].scatter_add_(
                1,
                diag_chunk.reshape(n_heads_chunk, -1),
                qk_chunk.reshape(n_heads_chunk, -1)
            )
            del diag_chunk, qk_chunk

        del diag_indices, qk

        # Force keep first num_recent_diags (in-place)
        slash_scores[:, :self.num_recent_diags] = float('inf')

        # Select top-k diagonal indices
        actual_slash = min(slash_size, seq_len)
        slash_indices = slash_scores.topk(actual_slash, dim=-1).indices
        slash_indices = slash_indices.sort(dim=-1).values
        del slash_scores

        return vertical_indices, slash_indices

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Select blocks for chunked CPU offload mode.

        For MInference in GPU-only mode, this method is not used.
        In CPU offload mode, it would select blocks based on the sparse pattern.

        For now, return all blocks (full attention fallback).
        """
        # MInference pattern is computed in attention.forward()
        # For CPU offload integration (Phase B), this would use the pattern
        return available_blocks

    def reset(self) -> None:
        """Reset policy state."""
        self._last_q_mask_cache.clear()

    def sparse_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Compute MInference sparse attention for prefill.

        Uses vertical + slash pattern to compute sparse attention efficiently.
        Memory-optimized to handle long sequences (64K+) by freeing intermediate tensors.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            layer_id: Current transformer layer index

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        from minference.ops.pit_sparse_flash_attention_v2 import _triton_mixed_sparse_attention
        from minference.cuda import convert_vertical_slash_indexes

        seq_len = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k.shape[1]

        # Estimate sparse pattern (uses temporary memory for qk scores)
        vertical_indices, slash_indices = self.estimate_pattern(q, k, layer_id)
        # Free any cached memory from pattern estimation
        torch.cuda.empty_cache()

        # Triton sparse attention kernel parameters
        block_size_M = 64
        block_size_N = 64

        # Calculate padding
        pad = (block_size_M - seq_len) & (block_size_M - 1)
        need_head_pad = head_dim not in [16, 32, 64, 128, 256, 512]
        head_pad = (2 ** math.ceil(math.log2(head_dim)) - head_dim) if need_head_pad else 0

        # Handle GQA: expand K/V to match query heads
        # Do this BEFORE creating batched tensors to avoid double copies
        if num_kv_heads < num_heads:
            num_groups = num_heads // num_kv_heads
            # Use repeat_interleave for memory-efficient expansion
            k_work = k.repeat_interleave(num_groups, dim=1)
            v_work = v.repeat_interleave(num_groups, dim=1)
        else:
            k_work = k
            v_work = v

        # Transform Q to [batch, heads, seq, dim] format with padding in one step
        # This avoids creating intermediate copies
        if pad > 0 or head_pad > 0:
            q_batched = torch.nn.functional.pad(
                q.unsqueeze(0).transpose(1, 2),
                [0, head_pad, 0, pad, 0, 0, 0, 0]
            ).contiguous()
        else:
            q_batched = q.unsqueeze(0).transpose(1, 2).contiguous()

        # Transform K to batched format
        if pad > 0 or head_pad > 0:
            k_batched = torch.nn.functional.pad(
                k_work.unsqueeze(0).transpose(1, 2),
                [0, head_pad, 0, pad, 0, 0, 0, 0]
            ).contiguous()
        else:
            k_batched = k_work.unsqueeze(0).transpose(1, 2).contiguous()

        # Free k_work if it was a copy (GQA case)
        if num_kv_heads < num_heads:
            del k_work

        # Transform V to batched format
        if pad > 0 or head_pad > 0:
            v_batched = torch.nn.functional.pad(
                v_work.unsqueeze(0).transpose(1, 2),
                [0, head_pad, 0, pad, 0, 0, 0, 0]
            ).contiguous()
        else:
            v_batched = v_work.unsqueeze(0).transpose(1, 2).contiguous()

        # Free v_work if it was a copy (GQA case)
        if num_kv_heads < num_heads:
            del v_work
            torch.cuda.empty_cache()

        # Prepare indices for Triton kernel
        v_idx = vertical_indices.to(torch.int32).reshape((1, num_heads, -1))
        v_idx = v_idx.sort(dim=-1, descending=False)[0].contiguous()
        del vertical_indices

        s_idx = slash_indices.to(torch.int32).reshape((1, num_heads, -1))
        s_idx = s_idx.sort(dim=-1, descending=True)[0].contiguous()
        del slash_indices

        seqlens = torch.tensor([seq_len], dtype=torch.int32, device=q.device)
        sm_scale = head_dim ** -0.5

        # Convert vertical+slash indices to block sparse format
        block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
            seqlens, v_idx, s_idx, seq_len, block_size_M, block_size_N,
        )
        del v_idx, s_idx

        # Call Triton mixed sparse attention kernel
        o = _triton_mixed_sparse_attention(
            q_batched, k_batched, v_batched, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, block_size_M, block_size_N,
        )

        # Free input tensors immediately after kernel call
        del q_batched, k_batched, v_batched
        del block_count, block_offset, column_count, column_index

        # Remove padding and convert back to [seq_len, num_heads, head_dim]
        o = o[..., :seq_len, :head_dim]
        o = o.transpose(1, 2).squeeze(0).contiguous()

        return o

    def __repr__(self) -> str:
        return (f"MInferencePolicy("
                f"adaptive_budget={self.adaptive_budget}, "
                f"vertical_size={self.vertical_size}, "
                f"slash_size={self.slash_size})")
