"""
Base class for eviction policies.

Users can implement custom policies by subclassing EvictionPolicy
and overriding the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Set


class EvictionPolicy(ABC):
    """
    Abstract base class for KV cache eviction policies.

    An eviction policy determines which GPU blocks to evict to CPU
    when GPU memory is full and new blocks need to be allocated.

    Lifecycle:
    1. on_block_allocated() - called when a new block is allocated
    2. on_block_access() - called each time a block is accessed (e.g., in attention)
    3. select_victim() - called when a block needs to be evicted
    4. on_block_evicted() - called after a block is evicted

    Example custom policy:
    ```python
    class MyCustomPolicy(EvictionPolicy):
        def __init__(self):
            self.priorities = {}

        def on_block_allocated(self, block_id: int, step: int):
            self.priorities[block_id] = step

        def on_block_access(self, block_id: int, step: int):
            # Custom access tracking
            pass

        def select_victim(self, candidates: Set[int]) -> int:
            # Return block with lowest priority
            return min(candidates, key=lambda b: self.priorities.get(b, 0))

        def on_block_evicted(self, block_id: int):
            self.priorities.pop(block_id, None)
    ```
    """

    @abstractmethod
    def on_block_allocated(self, block_id: int, step: int) -> None:
        """
        Called when a new block is allocated on GPU.

        Args:
            block_id: The GPU block ID that was allocated
            step: Current inference step (monotonically increasing)
        """
        pass

    @abstractmethod
    def on_block_access(self, block_id: int, step: int) -> None:
        """
        Called when a block is accessed during attention computation.

        Args:
            block_id: The GPU block ID being accessed
            step: Current inference step
        """
        pass

    @abstractmethod
    def select_victim(self, candidates: Set[int]) -> int:
        """
        Select a block to evict from the candidate set.

        This is called when GPU memory is full and a new block
        needs to be allocated. The returned block will be evicted
        to CPU.

        Args:
            candidates: Set of GPU block IDs that can be evicted
                       (blocks not currently being used)

        Returns:
            Block ID to evict

        Raises:
            ValueError: If candidates is empty
        """
        pass

    @abstractmethod
    def on_block_evicted(self, block_id: int) -> None:
        """
        Called after a block is evicted from GPU to CPU.

        Args:
            block_id: The GPU block ID that was evicted
        """
        pass

    def on_block_prefetched(self, block_id: int, step: int) -> None:
        """
        Called when a block is prefetched from CPU back to GPU.

        Default implementation calls on_block_allocated().
        Override for custom behavior.

        Args:
            block_id: The GPU block ID that was prefetched to
            step: Current inference step
        """
        self.on_block_allocated(block_id, step)

    def on_block_deallocated(self, block_id: int) -> None:
        """
        Called when a block is fully deallocated (sequence finished).

        Default implementation calls on_block_evicted().
        Override for custom behavior.

        Args:
            block_id: The GPU block ID being deallocated
        """
        self.on_block_evicted(block_id)

    def reset(self) -> None:
        """
        Reset policy state.

        Called when the inference engine is reset.
        Default implementation does nothing.
        """
        pass

    def get_eviction_order(self, candidates: Set[int], count: int) -> list:
        """
        Get multiple blocks to evict in order of priority.

        Default implementation calls select_victim() repeatedly.
        Override for more efficient batch selection.

        Args:
            candidates: Set of candidate block IDs
            count: Number of blocks to evict

        Returns:
            List of block IDs to evict, in order
        """
        result = []
        remaining = set(candidates)
        for _ in range(min(count, len(remaining))):
            if not remaining:
                break
            victim = self.select_victim(remaining)
            result.append(victim)
            remaining.remove(victim)
        return result
