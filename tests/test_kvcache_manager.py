"""Tests for KV cache managers."""

import pytest
import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.kvcache.gpu_manager import GPUOnlyManager


class MockSequence:
    """Mock sequence for testing block allocation."""

    def __init__(self, token_ids: list[int], block_size: int = 256):
        self._token_ids = token_ids
        self._block_size = block_size
        self.block_table: list[int] = []
        self.num_cached_tokens = 0

    def __len__(self):
        return len(self._token_ids)

    @property
    def num_blocks(self) -> int:
        return (len(self) + self._block_size - 1) // self._block_size

    def block(self, i: int) -> list[int]:
        start = i * self._block_size
        end = min((i + 1) * self._block_size, len(self))
        return self._token_ids[start:end]


class TestGPUOnlyManager:
    """Tests for GPU-only KV cache manager."""

    @pytest.fixture
    def manager(self):
        """Create a small manager for testing."""
        return GPUOnlyManager(num_blocks=16, block_size=256)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.block_size == 256
        assert manager.num_free_blocks == 16
        assert len(manager.blocks) == 16

    def test_allocate_cache(self, manager):
        """Test cache allocation."""
        manager.allocate_cache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            dtype=torch.float16,
        )

        assert manager.kv_cache is not None
        assert manager.kv_cache.shape == (2, 4, 16, 256, 8, 64)
        assert manager.kv_cache.device.type == "cuda"

    def test_get_layer_cache(self, manager):
        """Test getting layer cache."""
        manager.allocate_cache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            dtype=torch.float16,
        )

        k_cache, v_cache = manager.get_layer_cache(0)
        assert k_cache.shape == (16, 256, 8, 64)
        assert v_cache.shape == (16, 256, 8, 64)

    def test_can_allocate(self, manager):
        """Test allocation check."""
        seq = MockSequence([0] * 300)  # Needs 2 blocks
        assert manager.can_allocate(seq)

        # Fill up all blocks with unique tokens to avoid prefix caching
        for i in range(8):
            # Each sequence has unique tokens to prevent prefix cache hits
            s = MockSequence([i * 1000 + j for j in range(300)])
            manager.allocate(s)

        # Now should not be able to allocate
        new_seq = MockSequence([9999] * 300)
        assert not manager.can_allocate(new_seq)

    def test_allocate_and_deallocate(self, manager):
        """Test block allocation and deallocation."""
        seq = MockSequence([0] * 600)  # Needs 3 blocks
        initial_free = manager.num_free_blocks

        manager.allocate(seq)
        assert len(seq.block_table) == 3
        assert manager.num_free_blocks == initial_free - 3

        manager.deallocate(seq)
        assert len(seq.block_table) == 0
        assert manager.num_free_blocks == initial_free

    def test_can_append(self, manager):
        """Test append check."""
        seq = MockSequence([0] * 256)  # Exactly 1 block
        manager.allocate(seq)

        # Can append without new block (still in same block)
        seq._token_ids = [0] * 257
        assert manager.can_append(seq)

    def test_prepare_for_attention_noop(self, manager):
        """Test that prepare_for_attention is a no-op for GPU-only."""
        seq = MockSequence([0] * 100)
        manager.allocate(seq)

        # Should not raise
        manager.prepare_for_attention([seq], is_prefill=True)
        manager.prepare_for_attention([seq], is_prefill=False)

    def test_get_gpu_block_tables(self, manager):
        """Test getting GPU block tables."""
        seq1 = MockSequence([0] * 300)
        seq2 = MockSequence([0] * 600)

        manager.allocate(seq1)
        manager.allocate(seq2)

        tables = manager.get_gpu_block_tables([seq1, seq2])

        assert len(tables) == 2
        assert tables[0] == list(seq1.block_table)
        assert tables[1] == list(seq2.block_table)


class TestGPUOnlyManagerPrefixCaching:
    """Tests for prefix caching in GPU-only manager."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return GPUOnlyManager(num_blocks=32, block_size=256)

    def test_prefix_cache_hit(self, manager):
        """Test that identical prefixes are cached."""
        # Create two sequences with same prefix
        tokens = list(range(512))  # 2 full blocks
        seq1 = MockSequence(tokens)
        seq2 = MockSequence(tokens)

        manager.allocate(seq1)
        initial_free = manager.num_free_blocks

        manager.allocate(seq2)

        # Second sequence should reuse cached blocks
        assert seq2.num_cached_tokens >= 256  # At least first block cached
        # Should use fewer new blocks
        assert manager.num_free_blocks >= initial_free - 2

    def test_prefix_cache_different_suffix(self, manager):
        """Test cache with same prefix but different suffix."""
        prefix = list(range(256))  # 1 full block

        seq1 = MockSequence(prefix + [1000, 1001])
        seq2 = MockSequence(prefix + [2000, 2001])

        manager.allocate(seq1)
        manager.allocate(seq2)

        # First block should be shared
        assert seq1.block_table[0] == seq2.block_table[0]
        # Second block should be different
        assert seq1.block_table[1] != seq2.block_table[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
