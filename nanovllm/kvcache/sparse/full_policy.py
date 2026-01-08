"""
Full attention policy - loads all blocks (no sparsity).

This serves as a baseline and default policy when sparse
attention is not needed.
"""

from typing import List
from .policy import SparsePolicy, PolicyContext


class FullAttentionPolicy(SparsePolicy):
    """
    Full attention policy that loads all available blocks.

    This is the default behavior with no sparsity - all previous
    KV cache blocks are loaded for each query chunk.

    Use this as:
    - A baseline for comparing sparse policies
    - When you need full attention accuracy
    - For short sequences where sparsity isn't beneficial
    """

    # Full attention supports both prefill and decode
    supports_prefill = True
    supports_decode = True
    requires_block_selection = False  # Load all blocks, no selective loading

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """Return all blocks - no sparsity."""
        return available_blocks

    def __repr__(self) -> str:
        return "FullAttentionPolicy()"
