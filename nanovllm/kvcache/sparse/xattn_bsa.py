"""
XAttention Block Sparse Attention (BSA) Policy for nano-vllm.

This module implements XAttention-inspired block sparse attention for chunked prefill.
Current implementation loads all historical blocks (FULL strategy).

Sparse selection to be implemented in next phase.
"""

import torch
from typing import List, Optional, Tuple

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.utils.context import get_context


class XAttentionBSAPolicy(SparsePolicy):
    """
    XAttention Block Sparse Attention policy for chunked prefill.

    This policy uses block-level estimation to determine which KV blocks
    are important for the current chunk's queries, enabling sparse computation.

    Note: Current implementation loads all historical chunks (FULL strategy).
    Sparse selection to be implemented in next phase.
    """

    supports_prefill = False  # Uses standard select_blocks interface
    supports_decode = False  # BSA is prefill-only
    requires_block_selection = False  # Selection happens at chunk level, not block level

    def __init__(
        self,
        block_size: int = 128,
        samples_per_chunk: int = 128,
        threshold: float = 0.9,
    ):
        """
        Initialize XAttention BSA policy.

        Args:
            block_size: Number of tokens per block (default: 128)
            samples_per_chunk: Number of tokens to sample from each historical chunk for estimation
            threshold: Cumulative attention threshold for chunk selection (0-1)
        """
        self.block_size = block_size
        self.samples_per_chunk = samples_per_chunk
        self.threshold = threshold

    def select_blocks(self, available_blocks: List[int], ctx: PolicyContext) -> List[int]:
        """
        Select blocks to load from CPU.

        Current implementation returns all blocks (FULL strategy).
        Sparse selection to be implemented in next phase.

        Args:
            available_blocks: List of all available CPU block IDs
            ctx: Policy context with query info, chunk index, etc.

        Returns:
            List of selected block IDs to load
        """
        # Current: Return all blocks (FULL strategy)
        # TODO: Implement sparse selection based on query attention estimation
        return available_blocks

    def reset(self) -> None:
        """Reset policy state."""
        pass
