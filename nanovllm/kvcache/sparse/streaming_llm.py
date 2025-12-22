"""
StreamingLLM sparse attention policy.

Only keeps sink tokens (beginning) + recent tokens (end).
Intermediate context is discarded. This enables infinite-length
generation but loses intermediate context.

Reference: StreamingLLM paper on attention sinks.
"""

from dataclasses import dataclass
from typing import List
from .policy import SparsePolicy, PolicyContext


@dataclass
class StreamingLLMConfig:
    """Configuration for StreamingLLMPolicy."""

    num_sink_blocks: int = 1
    """Number of blocks at the beginning to always include (attention sinks)."""

    num_recent_blocks: int = 3
    """Number of most recent blocks to include (sliding window)."""


class StreamingLLMPolicy(SparsePolicy):
    """
    StreamingLLM pattern: sink tokens + recent tokens only.

    This is the most aggressive sparsity pattern - only keeps a small
    fixed window of context. Suitable for:
    - Very long streaming generation
    - When intermediate context can be safely discarded
    - Maximizing throughput over accuracy

    Pattern visualization:
    ```
    Blocks: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
             ↑              ×    ×    ×    ↑    ↑    ↑
           sink      (discarded)        recent window
    ```

    Warning: This loses information from intermediate blocks!
    Use only when this trade-off is acceptable.
    """

    def __init__(self, config: StreamingLLMConfig = None):
        self.config = config or StreamingLLMConfig()

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Select sink blocks + recent blocks only.

        Intermediate blocks are not loaded (effectively discarded).
        """
        n = len(available_blocks)

        # If total blocks fit in sink + recent, load all
        total_keep = self.config.num_sink_blocks + self.config.num_recent_blocks
        if n <= total_keep:
            return available_blocks

        selected_indices = set()

        # Sink blocks (first N)
        for i in range(min(self.config.num_sink_blocks, n)):
            selected_indices.add(i)

        # Recent blocks (last M)
        for i in range(max(0, n - self.config.num_recent_blocks), n):
            selected_indices.add(i)

        return [available_blocks[i] for i in sorted(selected_indices)]

    def __repr__(self) -> str:
        return (
            f"StreamingLLMPolicy(sink={self.config.num_sink_blocks}, "
            f"recent={self.config.num_recent_blocks})"
        )
