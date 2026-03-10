"""
LRU (Least Recently Used) eviction policy.

Evicts the block that was accessed least recently.
This is the default and recommended policy for most use cases.
"""

from collections import OrderedDict
from typing import Set

from nanovllm.kvcache.policies.base_policy import EvictionPolicy


class LRUPolicy(EvictionPolicy):
    """
    Least Recently Used (LRU) eviction policy.

    Maintains an ordered dictionary of block access times.
    When eviction is needed, selects the block that was
    accessed least recently.

    Properties:
    - O(1) access tracking
    - O(n) victim selection in worst case, but typically fast
      due to OrderedDict iteration order
    - Good for workloads with temporal locality
    """

    def __init__(self):
        # OrderedDict maintains insertion/update order
        # Key: block_id, Value: last_access_step
        # Oldest (least recently used) is at the front
        self.access_order: OrderedDict[int, int] = OrderedDict()

    def on_block_allocated(self, block_id: int, step: int) -> None:
        """Record allocation as an access."""
        # Move to end (most recently used)
        self.access_order[block_id] = step
        self.access_order.move_to_end(block_id)

    def on_block_access(self, block_id: int, step: int) -> None:
        """Update access time and move to end."""
        if block_id in self.access_order:
            self.access_order[block_id] = step
            self.access_order.move_to_end(block_id)

    def select_victim(self, candidates: Set[int]) -> int:
        """
        Select the least recently used block from candidates.

        Iterates from oldest to newest in access order,
        returns the first one that's in the candidate set.
        """
        if not candidates:
            raise ValueError("Cannot select victim from empty candidate set")

        # Iterate from oldest (front) to newest (back)
        for block_id in self.access_order:
            if block_id in candidates:
                return block_id

        # Fallback: return any candidate (shouldn't happen normally)
        return next(iter(candidates))

    def on_block_evicted(self, block_id: int) -> None:
        """Remove block from tracking."""
        self.access_order.pop(block_id, None)

    def on_block_deallocated(self, block_id: int) -> None:
        """Remove block from tracking."""
        self.access_order.pop(block_id, None)

    def reset(self) -> None:
        """Clear all tracking data."""
        self.access_order.clear()

    def get_eviction_order(self, candidates: Set[int], count: int) -> list:
        """
        Efficiently get multiple blocks to evict in LRU order.

        Optimized for batch eviction - iterates through access_order
        once instead of calling select_victim() multiple times.
        """
        result = []
        for block_id in self.access_order:
            if block_id in candidates:
                result.append(block_id)
                if len(result) >= count:
                    break
        return result

    def __repr__(self) -> str:
        return f"LRUPolicy(tracked_blocks={len(self.access_order)})"
