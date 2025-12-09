"""Tests for eviction policies."""

import pytest
from nanovllm.kvcache.policies.lru_policy import LRUPolicy
from nanovllm.kvcache.policies.fifo_policy import FIFOPolicy
from nanovllm.kvcache.policies import get_policy


class TestLRUPolicy:
    """Tests for LRU eviction policy."""

    def test_basic_eviction(self):
        """Test that LRU evicts least recently used block."""
        policy = LRUPolicy()

        # Allocate blocks 0, 1, 2 in order
        policy.on_block_allocated(0, step=1)
        policy.on_block_allocated(1, step=2)
        policy.on_block_allocated(2, step=3)

        # Access block 0 (makes it most recently used)
        policy.on_block_access(0, step=4)

        # Should evict block 1 (least recently used)
        candidates = {0, 1, 2}
        victim = policy.select_victim(candidates)
        assert victim == 1, f"Expected block 1, got {victim}"

    def test_access_updates_order(self):
        """Test that access updates LRU order."""
        policy = LRUPolicy()

        policy.on_block_allocated(0, step=1)
        policy.on_block_allocated(1, step=2)
        policy.on_block_allocated(2, step=3)

        # Access all in reverse order
        policy.on_block_access(2, step=4)
        policy.on_block_access(1, step=5)
        policy.on_block_access(0, step=6)

        # Block 2 is now LRU (accessed earliest after allocation update)
        candidates = {0, 1, 2}
        victim = policy.select_victim(candidates)
        assert victim == 2, f"Expected block 2, got {victim}"

    def test_eviction_removes_from_tracking(self):
        """Test that evicted blocks are removed from tracking."""
        policy = LRUPolicy()

        policy.on_block_allocated(0, step=1)
        policy.on_block_allocated(1, step=2)

        policy.on_block_evicted(0)

        # Only block 1 should be a candidate
        candidates = {0, 1}
        victim = policy.select_victim(candidates)
        assert victim == 1, "Should select block 1 since 0 was evicted"

    def test_batch_eviction_order(self):
        """Test get_eviction_order returns blocks in LRU order."""
        policy = LRUPolicy()

        for i in range(5):
            policy.on_block_allocated(i, step=i)

        # Access blocks 2 and 4
        policy.on_block_access(2, step=10)
        policy.on_block_access(4, step=11)

        candidates = {0, 1, 2, 3, 4}
        order = policy.get_eviction_order(candidates, count=3)

        # Should be 0, 1, 3 (in that order, skipping 2 and 4 until needed)
        assert order == [0, 1, 3], f"Expected [0, 1, 3], got {order}"


class TestFIFOPolicy:
    """Tests for FIFO eviction policy."""

    def test_basic_eviction(self):
        """Test that FIFO evicts oldest allocated block."""
        policy = FIFOPolicy()

        policy.on_block_allocated(0, step=1)
        policy.on_block_allocated(1, step=2)
        policy.on_block_allocated(2, step=3)

        # Access doesn't change FIFO order
        policy.on_block_access(0, step=4)

        candidates = {0, 1, 2}
        victim = policy.select_victim(candidates)
        assert victim == 0, f"Expected block 0 (oldest), got {victim}"

    def test_access_does_not_update_order(self):
        """Test that FIFO ignores access patterns."""
        policy = FIFOPolicy()

        policy.on_block_allocated(0, step=1)
        policy.on_block_allocated(1, step=2)
        policy.on_block_allocated(2, step=3)

        # Multiple accesses to block 0
        for i in range(10):
            policy.on_block_access(0, step=10 + i)

        # Block 0 should still be evicted first (FIFO order)
        candidates = {0, 1, 2}
        victim = policy.select_victim(candidates)
        assert victim == 0, f"Expected block 0, got {victim}"

    def test_prefetch_resets_order(self):
        """Test that prefetch moves block to end of queue."""
        policy = FIFOPolicy()

        policy.on_block_allocated(0, step=1)
        policy.on_block_allocated(1, step=2)
        policy.on_block_allocated(2, step=3)

        # Prefetch block 0 (moves to end)
        policy.on_block_prefetched(0, step=4)

        candidates = {0, 1, 2}
        victim = policy.select_victim(candidates)
        assert victim == 1, f"Expected block 1 (now oldest), got {victim}"

    def test_batch_eviction_order(self):
        """Test get_eviction_order returns blocks in FIFO order."""
        policy = FIFOPolicy()

        for i in range(5):
            policy.on_block_allocated(i, step=i)

        candidates = {0, 1, 2, 3, 4}
        order = policy.get_eviction_order(candidates, count=3)

        assert order == [0, 1, 2], f"Expected [0, 1, 2], got {order}"


class TestGetPolicy:
    """Tests for policy factory function."""

    def test_get_lru(self):
        """Test getting LRU policy by name."""
        policy = get_policy("lru")
        assert isinstance(policy, LRUPolicy)

    def test_get_fifo(self):
        """Test getting FIFO policy by name."""
        policy = get_policy("fifo")
        assert isinstance(policy, FIFOPolicy)

    def test_get_by_class_path(self):
        """Test getting policy by full class path."""
        policy = get_policy("nanovllm.kvcache.policies.lru_policy.LRUPolicy")
        assert isinstance(policy, LRUPolicy)

    def test_invalid_policy_name(self):
        """Test that invalid policy name raises error."""
        with pytest.raises((ValueError, ImportError)):
            get_policy("invalid_policy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
