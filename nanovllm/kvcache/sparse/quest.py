"""
Quest-style sparse attention policy.

Uses min/max key bounds per block to estimate attention scores
and select Top-K blocks most relevant to the current query.

Reference: Quest paper on query-aware KV cache selection.
"""

import logging
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .policy import SparsePolicy, PolicyContext

logger = logging.getLogger(__name__)


class BlockMetadataManager:
    """
    Manages per-block metadata for Quest-style sparse selection.

    Stores min/max key values for each block, which are used to
    compute upper bounds on attention scores without loading the
    full KV cache.

    Memory usage: 2 * num_blocks * num_layers * num_kv_heads * head_dim * dtype_size
    Example: 1000 blocks, 28 layers, 4 heads, 128 dim, bf16 = ~57 MB
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        """
        Initialize metadata storage.

        Args:
            num_blocks: Maximum number of CPU blocks
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension per head
            dtype: Data type for metadata storage
            device: Device for metadata storage (default: CUDA if available)
        """
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per-block min/max key values: [num_blocks, num_layers, num_heads, head_dim]
        # Stored on GPU for efficient score computation during decode
        shape = (num_blocks, num_layers, num_kv_heads, head_dim)
        self.key_min = torch.zeros(shape, dtype=dtype, device=self.device)
        self.key_max = torch.zeros(shape, dtype=dtype, device=self.device)

        # Track which blocks have valid metadata
        self.valid_blocks = torch.zeros(num_blocks, dtype=torch.bool, device=self.device)

    def update_metadata(
        self,
        block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """
        Update min/max key bounds for a block.

        Called BEFORE offload to CPU, while k_cache is still on GPU.

        Args:
            block_id: CPU block ID
            layer_id: Layer index
            k_cache: Key cache tensor [block_size, num_kv_heads, head_dim] (on GPU)
            num_valid_tokens: Number of valid tokens in this block
        """
        if num_valid_tokens == 0:
            return

        # Get valid keys only (k_cache is on GPU, metadata is on GPU)
        k_valid = k_cache[:num_valid_tokens]  # [num_tokens, heads, dim]

        # Compute min/max across token dimension (all on GPU)
        self.key_min[block_id, layer_id] = k_valid.min(dim=0).values
        self.key_max[block_id, layer_id] = k_valid.max(dim=0).values
        self.valid_blocks[block_id] = True

    def get_block_metadata(
        self,
        block_ids: List[int],
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get min/max keys for specified blocks.

        Args:
            block_ids: List of CPU block IDs
            layer_id: Layer index

        Returns:
            Tuple of (key_min, key_max) tensors
            Shape: [num_blocks, num_kv_heads, head_dim]
        """
        key_min = self.key_min[block_ids, layer_id]
        key_max = self.key_max[block_ids, layer_id]
        return key_min, key_max

    def reset(self) -> None:
        """Reset all metadata."""
        self.key_min.zero_()
        self.key_max.zero_()
        self.valid_blocks.zero_()


@dataclass
class QuestConfig:
    """Configuration for QuestPolicy."""

    topk_blocks: int = 8
    """Number of top blocks to select based on estimated attention scores."""

    threshold_blocks: int = 4
    """If total blocks <= threshold, load all (no scoring needed)."""

    include_sink_blocks: int = 0
    """Always include this many sink blocks (first N blocks), in addition to Top-K."""

    include_recent_blocks: int = 0
    """Always include this many recent blocks (last N blocks), in addition to Top-K."""


class QuestPolicy(SparsePolicy):
    """
    Quest-style Top-K block selection using min/max key bounds.

    For each query, computes an upper bound on attention scores for
    each block using the stored min/max keys, then selects the Top-K
    blocks with highest estimated scores.

    Score computation:
        score(q, block) = max(q · key_min, q · key_max)

    This upper bound is derived from the fact that for any key k in
    the block: min_k <= k <= max_k (element-wise), so the actual
    attention score is bounded by the maximum of the two extremes.

    Note: This is a decode-only policy. For prefill, use FullAttentionPolicy.
    """

    # Quest is decode-only
    supports_prefill = False
    supports_decode = True
    requires_block_selection = True  # Quest affects KV load strategy (selective block loading)

    def __init__(self, config: QuestConfig):
        """
        Initialize Quest policy.

        Args:
            config: QuestConfig with selection parameters
        """
        self.config = config
        self.metadata: Optional[BlockMetadataManager] = None

    def initialize(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
        device: torch.device = None,
    ) -> None:
        """Create BlockMetadataManager for storing min/max keys on GPU."""
        self.metadata = BlockMetadataManager(
            num_blocks=num_cpu_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Select Top-K blocks based on query-key similarity bounds.

        If query is not available (some prefill scenarios), falls back
        to loading all blocks.
        """
        if self.metadata is None:
            raise RuntimeError(
                "QuestPolicy not initialized. Call initialize() first or "
                "let the framework call it during KV cache allocation."
            )

        n = len(available_blocks)

        # If below threshold or no query, load all
        if n <= self.config.threshold_blocks:
            return available_blocks

        if ctx.query is None:
            # No query available - cannot compute scores
            return available_blocks

        # Get metadata for available blocks (already on GPU)
        key_min, key_max = self.metadata.get_block_metadata(
            available_blocks, ctx.layer_id
        )

        # Metadata is already on GPU, same device as query
        device = ctx.query.device

        # Compute upper bound scores
        # query shape: [1, num_heads, head_dim] or [1, seq_len, num_heads, head_dim]
        q = ctx.query
        if q.dim() == 4:
            # Prefill: use mean over sequence length
            q = q.mean(dim=1)  # [1, num_heads, head_dim]
        q = q.squeeze(0)  # [num_q_heads, head_dim]

        # Handle GQA: query may have more heads than KV
        # key_min/key_max: [num_blocks, num_kv_heads, head_dim]
        num_q_heads = q.shape[0]
        num_kv_heads = key_min.shape[1]

        if num_q_heads != num_kv_heads:
            # GQA: group query heads and average per KV group
            # Reshape q: [num_q_heads, head_dim] -> [num_kv_heads, group_size, head_dim]
            group_size = num_q_heads // num_kv_heads
            q = q.view(num_kv_heads, group_size, -1).mean(dim=1)  # [num_kv_heads, head_dim]

        # Score: max(q·k_min, q·k_max) averaged over heads
        # key_min/key_max: [num_blocks, num_kv_heads, head_dim]
        # q: [num_kv_heads, head_dim]
        score_min = torch.einsum('hd,bhd->bh', q, key_min)  # [num_blocks, kv_heads]
        score_max = torch.einsum('hd,bhd->bh', q, key_max)
        scores = torch.maximum(score_min, score_max).mean(dim=-1)  # [num_blocks]

        # Build selection set
        selected_indices = set()

        # Always include sink blocks
        for i in range(min(self.config.include_sink_blocks, n)):
            selected_indices.add(i)

        # Always include recent blocks
        for i in range(max(0, n - self.config.include_recent_blocks), n):
            selected_indices.add(i)

        # Top-K selection from remaining
        remaining_k = max(0, self.config.topk_blocks - len(selected_indices))
        if remaining_k > 0:
            # Mask out already selected
            mask = torch.ones(n, dtype=torch.bool, device=device)
            for idx in selected_indices:
                mask[idx] = False

            if mask.any():
                masked_scores = scores.clone()
                masked_scores[~mask] = float('-inf')
                topk_count = min(remaining_k, mask.sum().item())
                if topk_count > 0:
                    topk_indices = masked_scores.topk(topk_count).indices.cpu().tolist()
                    selected_indices.update(topk_indices)

        # Return in sequential order for better memory access
        result = [available_blocks[i] for i in sorted(selected_indices)]

        # Log selection info (only for layer 0 to avoid spam)
        if ctx.layer_id == 0:
            logger.debug(
                f"Quest select: {len(result)}/{n} blocks "
                f"(topk={self.config.topk_blocks}, sink={self.config.include_sink_blocks}, "
                f"recent={self.config.include_recent_blocks})"
            )

        return result

    def on_prefill_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """Update min/max key metadata during prefill offload."""
        if self.metadata is not None:
            self.metadata.update_metadata(cpu_block_id, layer_id, k_cache, num_valid_tokens)

    def on_decode_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """Update min/max key metadata during decode offload (for new blocks)."""
        if self.metadata is not None:
            self.metadata.update_metadata(cpu_block_id, layer_id, k_cache, num_valid_tokens)

    def reset(self) -> None:
        """Reset metadata."""
        if self.metadata is not None:
            self.metadata.reset()

    def __repr__(self) -> str:
        return (
            f"QuestPolicy(topk={self.config.topk_blocks}, "
            f"threshold={self.config.threshold_blocks}, "
            f"sink={self.config.include_sink_blocks}, "
            f"recent={self.config.include_recent_blocks})"
        )
