"""Tests for CPU-GPU offload engine."""

import pytest
import torch

from nanovllm.kvcache.offload_engine import OffloadEngine


class TestOffloadEngine:
    """Tests for OffloadEngine."""

    @pytest.fixture
    def engine(self):
        """Create a small engine for testing."""
        return OffloadEngine(
            num_layers=2,
            num_gpu_blocks=4,
            num_cpu_blocks=8,
            block_size=256,
            num_kv_heads=4,
            head_dim=64,
            dtype=torch.float16,
            num_streams=2,
        )

    def test_initialization(self, engine):
        """Test engine initialization."""
        # Check GPU cache shape
        assert engine.k_cache_gpu.shape == (2, 4, 256, 4, 64)
        assert engine.v_cache_gpu.shape == (2, 4, 256, 4, 64)

        # Check CPU cache shape
        assert engine.k_cache_cpu.shape == (2, 8, 256, 4, 64)
        assert engine.v_cache_cpu.shape == (2, 8, 256, 4, 64)

        # Check pinned memory
        assert engine.k_cache_cpu.is_pinned()
        assert engine.v_cache_cpu.is_pinned()

        # Check gather indices
        assert engine.gather_indices_cpu.shape == (2, 4)
        assert engine.gather_indices_gpu.shape == (2, 4)

    def test_get_layer_cache(self, engine):
        """Test getting layer cache."""
        k, v = engine.get_layer_cache(0)
        assert k.shape == (4, 256, 4, 64)
        assert v.shape == (4, 256, 4, 64)
        assert k.device.type == "cuda"
        assert v.device.type == "cuda"

    def test_prefetch_and_offload(self, engine):
        """Test async prefetch and offload."""
        # Write some data to CPU block 0
        engine.k_cache_cpu[0, 0].fill_(1.0)
        engine.v_cache_cpu[0, 0].fill_(2.0)

        # Prefetch to GPU block 2
        event = engine.prefetch_block_async(
            layer_id=0,
            cpu_block_id=0,
            gpu_block_id=2,
        )
        event.synchronize()

        # Verify data was copied (move GPU to CPU for comparison)
        assert torch.allclose(engine.k_cache_gpu[0, 2].cpu(), engine.k_cache_cpu[0, 0])
        assert torch.allclose(engine.v_cache_gpu[0, 2].cpu(), engine.v_cache_cpu[0, 0])

        # Modify GPU data
        engine.k_cache_gpu[0, 2].fill_(3.0)
        engine.v_cache_gpu[0, 2].fill_(4.0)

        # Offload to CPU block 5
        event = engine.offload_block_async(
            layer_id=0,
            gpu_block_id=2,
            cpu_block_id=5,
        )
        event.synchronize()

        # Verify data was copied
        assert torch.allclose(engine.k_cache_cpu[0, 5], engine.k_cache_gpu[0, 2].cpu())
        assert torch.allclose(engine.v_cache_cpu[0, 5], engine.v_cache_gpu[0, 2].cpu())

    def test_update_gather_indices(self, engine):
        """Test updating gather indices."""
        # Manually set CPU data
        for i in range(8):
            engine.k_cache_cpu[0, i].fill_(float(i))
            engine.v_cache_cpu[0, i].fill_(float(i + 100))

        # Update indices for layer 0: (cpu_block_id, gpu_slot)
        mappings = [(2, 0), (5, 1), (1, 2), (7, 3)]
        engine.update_gather_indices(layer_id=0, mappings=mappings)
        torch.cuda.synchronize()

        # Verify indices were set
        expected = torch.tensor([2, 5, 1, 7], dtype=torch.int64)
        assert torch.equal(engine.gather_indices_cpu[0], expected)

    def test_gathered_h2d_layer(self, engine):
        """Test gathered H2D copy for a layer."""
        # Set up CPU data with known values
        for i in range(8):
            engine.k_cache_cpu[0, i].fill_(float(i))
            engine.v_cache_cpu[0, i].fill_(float(i + 100))

        # Set gather indices: (cpu_block_id, gpu_slot)
        # GPU slot 0 gets CPU block 3, GPU slot 1 gets CPU block 0, etc.
        mappings = [(3, 0), (0, 1), (7, 2), (2, 3)]
        engine.update_gather_indices(layer_id=0, mappings=mappings)
        torch.cuda.synchronize()

        # Execute gathered H2D
        engine.gathered_h2d_layer(layer_id=0)
        torch.cuda.synchronize()

        # Verify: GPU slot 0 should have CPU block 3's data
        assert torch.allclose(engine.k_cache_gpu[0, 0],
                              torch.full_like(engine.k_cache_gpu[0, 0], 3.0))
        # GPU slot 1 should have CPU block 0's data
        assert torch.allclose(engine.k_cache_gpu[0, 1],
                              torch.full_like(engine.k_cache_gpu[0, 1], 0.0))
        # GPU slot 2 should have CPU block 7's data
        assert torch.allclose(engine.k_cache_gpu[0, 2],
                              torch.full_like(engine.k_cache_gpu[0, 2], 7.0))
        # GPU slot 3 should have CPU block 2's data
        assert torch.allclose(engine.k_cache_gpu[0, 3],
                              torch.full_like(engine.k_cache_gpu[0, 3], 2.0))

    def test_multi_layer_independence(self, engine):
        """Test that layers are independent."""
        # Set different data for each layer
        engine.k_cache_cpu[0, 0].fill_(1.0)
        engine.k_cache_cpu[1, 0].fill_(2.0)

        # Prefetch layer 0
        event = engine.prefetch_block_async(0, 0, 0)
        event.synchronize()

        # Verify only layer 0 was affected
        assert torch.allclose(engine.k_cache_gpu[0, 0],
                              torch.full_like(engine.k_cache_gpu[0, 0], 1.0))
        # Layer 1 should be zeros (initial state)
        assert not torch.allclose(engine.k_cache_gpu[1, 0],
                                  torch.full_like(engine.k_cache_gpu[1, 0], 2.0))


class TestOffloadEngineFixedAddresses:
    """Tests verifying fixed address property for CUDA Graph compatibility."""

    @pytest.fixture
    def engine(self):
        """Create engine for address tests."""
        return OffloadEngine(
            num_layers=2,
            num_gpu_blocks=4,
            num_cpu_blocks=8,
            block_size=256,
            num_kv_heads=4,
            head_dim=64,
            dtype=torch.float16,
            num_streams=2,
        )

    def test_gpu_cache_address_fixed(self, engine):
        """Verify GPU cache addresses don't change."""
        k_ptr_before = engine.k_cache_gpu.data_ptr()
        v_ptr_before = engine.v_cache_gpu.data_ptr()

        # Perform some operations - mappings is List[(cpu_block_id, gpu_slot)]
        mappings = [(0, 0), (1, 1), (2, 2), (3, 3)]
        engine.update_gather_indices(0, mappings)
        engine.gathered_h2d_layer(0)
        torch.cuda.synchronize()

        # Addresses should be the same
        assert engine.k_cache_gpu.data_ptr() == k_ptr_before
        assert engine.v_cache_gpu.data_ptr() == v_ptr_before

    def test_gather_indices_gpu_address_fixed(self, engine):
        """Verify gather indices GPU tensor address doesn't change."""
        ptr_before = engine.gather_indices_gpu.data_ptr()

        # Update indices multiple times - mappings is List[(cpu_block_id, gpu_slot)]
        mappings = [(0, 0), (1, 1), (2, 2), (3, 3)]
        for _ in range(10):
            engine.update_gather_indices(0, mappings)
        torch.cuda.synchronize()

        assert engine.gather_indices_gpu.data_ptr() == ptr_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
