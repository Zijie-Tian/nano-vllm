"""
Vertical-Slash sparse attention policy (MInference-style).

Selects sink blocks (beginning of sequence) + local window blocks
(near the current query position). This pattern captures:
- Important initial context (system prompt, instructions)
- Recent context (relevant for local dependencies)
"""

from dataclasses import dataclass
from typing import List
from .policy import SparsePolicy, PolicyContext


@dataclass
class VerticalSlashConfig:
    """Configuration for VerticalSlashPolicy."""

    num_sink_blocks: int = 1
    """Number of blocks at the beginning to always include (sink tokens)."""

    local_window_blocks: int = 2
    """Number of blocks in the local window near current query position."""

    threshold_blocks: int = 4
    """If total blocks <= threshold, load all (no sparsity applied)."""


class VerticalSlashPolicy(SparsePolicy):
    """
    Vertical-Slash pattern: sink tokens + local window.

    This pattern is inspired by MInference and observations that:
    1. Initial tokens (sink) often receive high attention
    2. Local context (recent tokens) is important for dependencies

    Pattern visualization:
    ```
    Blocks: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
             ↑                        ↑    ↑    ↑
           sink                   local window (for query at block 9)
    ```

    For prefill chunk K, the local window is blocks [K-window, K-1].
    For decode, the local window is the last N blocks.
    """

    def __init__(self, config: VerticalSlashConfig = None):
        self.config = config or VerticalSlashConfig()

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Select sink blocks + local window blocks.

        For prefill: local window is relative to current chunk position.
        For decode: local window is the most recent blocks.
        """
        n = len(available_blocks)

        # If below threshold, load all
        if n <= self.config.threshold_blocks:
            return available_blocks

        selected_indices = set()

        # Sink blocks (first N blocks)
        for i in range(min(self.config.num_sink_blocks, n)):
            selected_indices.add(i)

        # Local window
        if ctx.is_prefill:
            # For prefill chunk K, local window is blocks [K-window, K-1]
            # (blocks before current chunk, not including current)
            window_end = min(ctx.query_chunk_idx, n)
            window_start = max(0, window_end - self.config.local_window_blocks)
            for i in range(window_start, window_end):
                selected_indices.add(i)
        else:
            # For decode, local window is the last M blocks
            for i in range(max(0, n - self.config.local_window_blocks), n):
                selected_indices.add(i)

        # Return blocks in order (maintains sequential access pattern)
        return [available_blocks[i] for i in sorted(selected_indices)]

    def __repr__(self) -> str:
        return (
            f"VerticalSlashPolicy(sink={self.config.num_sink_blocks}, "
            f"window={self.config.local_window_blocks}, "
            f"threshold={self.config.threshold_blocks})"
        )
