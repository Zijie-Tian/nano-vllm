"""
Base class for sparse attention policies.

Sparse attention policies determine which KV cache blocks to load
from CPU for each query chunk during chunked attention computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
import torch


@dataclass
class PolicyContext:
    """
    Context passed to sparse policy for block selection.

    This dataclass contains all information needed by a sparse policy
    to decide which blocks to load for the current query chunk.
    """

    query_chunk_idx: int
    """Index of the current query chunk (0-indexed)."""

    num_query_chunks: int
    """Total number of query chunks in this prefill."""

    layer_id: int
    """Current transformer layer index."""

    query: Optional[torch.Tensor]
    """
    Query tensor for current chunk.
    Shape: [1, num_heads, head_dim] for decode, [1, seq_len, num_heads, head_dim] for prefill.
    May be None if not available (e.g., some prefill scenarios).
    """

    is_prefill: bool
    """True if in prefill phase, False if in decode phase."""

    block_size: int = 1024
    """Number of tokens per block."""

    total_kv_len: int = 0
    """Total KV sequence length so far (for reference)."""


class SparsePolicy(ABC):
    """
    Abstract base class for sparse attention policies.

    Subclass this and implement select_blocks() to create custom
    sparse attention patterns. The policy receives context about
    the current query chunk and returns which KV blocks to load.

    Example:
        class MySparsePolicy(SparsePolicy):
            def select_blocks(self, available_blocks, ctx):
                # Load first block and last 2 blocks
                if len(available_blocks) <= 3:
                    return available_blocks
                return [available_blocks[0]] + available_blocks[-2:]
    """

    @abstractmethod
    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Select which KV blocks to load for the current query chunk.

        This is the core method that defines the sparse attention pattern.
        The returned blocks will be loaded from CPU to GPU for attention
        computation against the current query chunk.

        Args:
            available_blocks: List of CPU block IDs that contain KV cache
                             from previous chunks. These are ordered by
                             their position in the sequence.
            ctx: PolicyContext with information about the current query
                 chunk, layer, phase (prefill/decode), etc.

        Returns:
            List of block IDs to load (must be a subset of available_blocks).
            The order may affect performance (sequential access is faster).
            Returning [] means no previous blocks will be loaded.
        """
        pass

    def on_block_offloaded(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """
        Hook called when a block is offloaded from GPU to CPU.

        Override this to collect metadata about blocks (e.g., min/max keys
        for Quest-style selection). Default implementation does nothing.

        Args:
            cpu_block_id: The CPU block ID that was written
            layer_id: Transformer layer index
            k_cache: Key cache tensor [block_size, num_kv_heads, head_dim]
            num_valid_tokens: Number of valid tokens in this block
        """
        pass

    def reset(self) -> None:
        """
        Reset policy state.

        Called when starting a new sequence or clearing state.
        Default implementation does nothing.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
