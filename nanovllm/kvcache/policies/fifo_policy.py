"""
FIFO (First In, First Out) eviction policy.

Evicts the block that was allocated earliest.
Simple policy that ignores access patterns.
"""

from collections import OrderedDict
from typing import Set

from nanovllm.kvcache.policies.base_policy import EvictionPolicy


class FIFOPolicy(EvictionPolicy):
    """
    First In, First Out (FIFO) eviction policy.

    Evicts blocks in the order they were allocated,
    regardless of access patterns.

    Properties:
    - O(1) operations for all methods
    - Simple and predictable behavior
    - Good for streaming workloads where older data
      is naturally less relevant
    - Does not adapt to access patterns (unlike LRU)
    """

    def __init__(self):
        # OrderedDict maintains insertion order
        # Key: block_id, Value: allocation_step
        # Oldest (first allocated) is at the front
        self.allocation_order: OrderedDict[int, int] = OrderedDict()

    def on_block_allocated(self, block_id: int, step: int) -> None:
        """Record allocation order (does not change on access)."""
        if block_id not in self.allocation_order:
            self.allocation_order[block_id] = step

    def on_block_access(self, block_id: int, step: int) -> None:
        """
        FIFO ignores access patterns.

        This is the key difference from LRU - we don't
        update the position based on access.
        """
        pass  # Intentionally empty

    def select_victim(self, candidates: Set[int]) -> int:
        """
        Select the earliest allocated block from candidates.
        """
        if not candidates:
            raise ValueError("Cannot select victim from empty candidate set")

        # Iterate from oldest (front) to newest (back)
        for block_id in self.allocation_order:
            if block_id in candidates:
                return block_id

        # Fallback: return any candidate
        return next(iter(candidates))

    def on_block_evicted(self, block_id: int) -> None:
        """Remove block from tracking."""
        self.allocation_order.pop(block_id, None)

    def on_block_prefetched(self, block_id: int, step: int) -> None:
        """
        When prefetched, treat as new allocation.

        This moves the block to the end of the queue,
        giving it more time before eviction.
        """
        # Remove old entry if exists
        self.allocation_order.pop(block_id, None)
        # Add as new allocation
        self.allocation_order[block_id] = step

    def on_block_deallocated(self, block_id: int) -> None:
        """Remove block from tracking."""
        self.allocation_order.pop(block_id, None)

    def reset(self) -> None:
        """Clear all tracking data."""
        self.allocation_order.clear()

    def get_eviction_order(self, candidates: Set[int], count: int) -> list:
        """
        Get multiple blocks to evict in FIFO order.
        """
        result = []
        for block_id in self.allocation_order:
            if block_id in candidates:
                result.append(block_id)
                if len(result) >= count:
                    break
        return result

    def __repr__(self) -> str:
        return f"FIFOPolicy(tracked_blocks={len(self.allocation_order)})"
