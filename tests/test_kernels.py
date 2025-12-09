"""Tests for Triton gathered copy kernels."""

import pytest
import torch

from nanovllm.kvcache.kernels import gathered_copy, gathered_copy_kv


class TestGatheredCopy:
    """Tests for gathered copy kernel."""

    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors."""
        torch.cuda.manual_seed(42)
        num_src_blocks = 16
        num_dst_blocks = 8
        block_size = 256
        kv_dim = 64

        src = torch.randn(num_src_blocks, block_size, kv_dim,
                          dtype=torch.float16, device="cuda")
        dst = torch.zeros(num_dst_blocks, block_size, kv_dim,
                          dtype=torch.float16, device="cuda")

        # Indices: dst[i] = src[indices[i]]
        indices = torch.randint(0, num_src_blocks, (num_dst_blocks,),
                                dtype=torch.int64, device="cuda")

        return src, dst, indices

    def test_basic_copy(self, setup_tensors):
        """Test basic gathered copy."""
        src, dst, indices = setup_tensors

        gathered_copy(src, dst, indices)

        # Verify copy
        for i in range(len(indices)):
            src_idx = indices[i].item()
            assert torch.allclose(dst[i], src[src_idx]), f"Mismatch at index {i}"

    def test_skip_negative_indices(self, setup_tensors):
        """Test that negative indices are skipped."""
        src, dst, indices = setup_tensors

        # Set some indices to -1
        indices[2] = -1
        indices[5] = -1

        # Fill dst with a known value
        dst.fill_(999.0)

        gathered_copy(src, dst, indices)

        # Skipped slots should be unchanged
        assert (dst[2] == 999.0).all()
        assert (dst[5] == 999.0).all()

        # Non-skipped slots should be copied
        for i in [0, 1, 3, 4, 6, 7]:
            src_idx = indices[i].item()
            assert torch.allclose(dst[i], src[src_idx])

    def test_single_block(self):
        """Test copying a single block."""
        src = torch.randn(4, 256, 64, dtype=torch.float16, device="cuda")
        dst = torch.zeros(1, 256, 64, dtype=torch.float16, device="cuda")
        indices = torch.tensor([2], dtype=torch.int64, device="cuda")

        gathered_copy(src, dst, indices)

        assert torch.allclose(dst[0], src[2])


class TestGatheredCopyKV:
    """Tests for gathered K/V cache copy kernel."""

    @pytest.fixture
    def setup_kv_tensors(self):
        """Create K/V test tensors."""
        torch.cuda.manual_seed(42)
        num_src_blocks = 16
        num_dst_blocks = 8
        block_size = 256
        num_kv_heads = 4
        head_dim = 64

        k_src = torch.randn(num_src_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.float16, device="cuda")
        v_src = torch.randn(num_src_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.float16, device="cuda")
        k_dst = torch.zeros(num_dst_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.float16, device="cuda")
        v_dst = torch.zeros(num_dst_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.float16, device="cuda")

        indices = torch.randint(0, num_src_blocks, (num_dst_blocks,),
                                dtype=torch.int64, device="cuda")

        return k_src, v_src, k_dst, v_dst, indices

    def test_kv_copy(self, setup_kv_tensors):
        """Test K/V gathered copy."""
        k_src, v_src, k_dst, v_dst, indices = setup_kv_tensors

        gathered_copy_kv(k_src, v_src, k_dst, v_dst, indices)

        # Verify copy
        for i in range(len(indices)):
            src_idx = indices[i].item()
            assert torch.allclose(k_dst[i], k_src[src_idx]), f"K mismatch at {i}"
            assert torch.allclose(v_dst[i], v_src[src_idx]), f"V mismatch at {i}"

    def test_kv_skip_negative(self, setup_kv_tensors):
        """Test that negative indices are skipped for K/V."""
        k_src, v_src, k_dst, v_dst, indices = setup_kv_tensors

        indices[0] = -1
        k_dst.fill_(999.0)
        v_dst.fill_(999.0)

        gathered_copy_kv(k_src, v_src, k_dst, v_dst, indices)

        assert (k_dst[0] == 999.0).all()
        assert (v_dst[0] == 999.0).all()


class TestPerformance:
    """Performance benchmarks for gathered copy."""

    @pytest.mark.parametrize("num_blocks", [8, 32, 128])
    def test_throughput(self, num_blocks):
        """Benchmark copy throughput."""
        block_size = 256
        kv_dim = 64

        src = torch.randn(num_blocks * 2, block_size, kv_dim,
                          dtype=torch.float16, device="cuda")
        dst = torch.zeros(num_blocks, block_size, kv_dim,
                          dtype=torch.float16, device="cuda")
        indices = torch.arange(num_blocks, dtype=torch.int64, device="cuda")

        # Warmup
        for _ in range(10):
            gathered_copy(src, dst, indices)
        torch.cuda.synchronize()

        # Benchmark
        import time
        start = time.perf_counter()
        num_iters = 100
        for _ in range(num_iters):
            gathered_copy(src, dst, indices)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_copied = num_blocks * block_size * kv_dim * 2 * num_iters  # fp16
        bandwidth_gbps = bytes_copied / elapsed / 1e9

        print(f"\n{num_blocks} blocks: {bandwidth_gbps:.2f} GB/s")

        # Should achieve reasonable bandwidth (lower threshold for small blocks due to kernel launch overhead)
        min_bandwidth = 5 if num_blocks <= 16 else 10
        assert bandwidth_gbps > min_bandwidth, f"Bandwidth too low: {bandwidth_gbps} GB/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
