"""
Tests for CUDA sgDMA (cudaMemcpy2D) extension.

Author: Zijie Tian
"""

import torch
import time
from nanovllm.comm import memcpy_2d

# ============================================================
# Configuration
# ============================================================

class Config:
    num_layers = 32
    num_blocks = 10
    block_size = 4096
    num_kv_heads = 8
    head_dim = 128
    dtype = torch.float16

    @property
    def features_per_block(self):
        return self.block_size * self.num_kv_heads * self.head_dim

    @property
    def bytes_per_block(self):
        return self.features_per_block * self.dtype.itemsize

    @property
    def bytes_per_layer(self):
        return self.num_blocks * self.bytes_per_block


# ============================================================
# Performance Benchmark
# ============================================================

def benchmark_sgdma():
    """Benchmark cudaMemcpy2D vs standard PyTorch methods."""
    print("\n=== Performance Benchmark ===")

    cfg = Config()

    print(f"  Configuration:")
    print(f"    num_layers: {cfg.num_layers}")
    print(f"    num_blocks: {cfg.num_blocks}")
    print(f"    block_size: {cfg.block_size}")
    print(f"    dtype: {cfg.dtype}")
    print(f"    bytes_per_block: {cfg.bytes_per_block / 1024:.1f} KB")
    print(f"    total transfer size: {cfg.num_layers * cfg.bytes_per_block / 1024 / 1024:.1f} MB")

    num_iterations = 10
    warmup = 3
    test_block_id = 5

    # Allocate memory
    cpu_strided = torch.randn(
        cfg.num_layers,
        cfg.num_blocks,
        cfg.features_per_block,
        dtype=cfg.dtype,
        pin_memory=True
    )

    # ========================================
    # Method A: cudaMemcpy2D with sgDMA
    # ========================================
    gpu_buffer_a = torch.empty(cfg.num_layers, cfg.features_per_block, dtype=cfg.dtype, device='cuda')

    spitch = cfg.bytes_per_layer
    dpitch = cfg.bytes_per_block
    width = cfg.bytes_per_block
    height = cfg.num_layers
    src_view = cpu_strided[:, test_block_id, :]

    # Warmup
    for _ in range(warmup):
        memcpy_2d(gpu_buffer_a, src_view, dpitch, spitch, width, height, "h2d")
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        memcpy_2d(gpu_buffer_a, src_view, dpitch, spitch, width, height, "h2d")
    torch.cuda.synchronize()
    elapsed_a = time.perf_counter() - start

    avg_time_a = elapsed_a / num_iterations * 1000  # ms
    bandwidth_a = (cfg.num_layers * cfg.bytes_per_block * num_iterations / 1e9) / elapsed_a

    print(f"\n  Method A (cudaMemcpy2D sgDMA):")
    print(f"    Avg time: {avg_time_a:.3f} ms")
    print(f"    Bandwidth: {bandwidth_a:.2f} GB/s")

    # ========================================
    # Method B: PyTorch .cuda() on strided view
    # ========================================
    # Warmup
    for _ in range(warmup):
        _ = cpu_strided[:, test_block_id, :].cuda()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = cpu_strided[:, test_block_id, :].cuda()
    torch.cuda.synchronize()
    elapsed_b = time.perf_counter() - start

    avg_time_b = elapsed_b / num_iterations * 1000  # ms
    bandwidth_b = (cfg.num_layers * cfg.bytes_per_block * num_iterations / 1e9) / elapsed_b

    print(f"\n  Method B (PyTorch .cuda() on strided):")
    print(f"    Avg time: {avg_time_b:.3f} ms")
    print(f"    Bandwidth: {bandwidth_b:.2f} GB/s")

    # ========================================
    # Method C: PyTorch .cuda() on contiguous (pinned)
    # ========================================
    # Create contiguous version with pinned memory
    cpu_contiguous = torch.empty(
        cfg.num_layers,
        cfg.features_per_block,
        dtype=cfg.dtype,
        pin_memory=True
    )
    cpu_contiguous.copy_(cpu_strided[:, test_block_id, :])

    # Warmup
    for _ in range(warmup):
        _ = cpu_contiguous.cuda()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = cpu_contiguous.cuda()
    torch.cuda.synchronize()
    elapsed_c = time.perf_counter() - start

    avg_time_c = elapsed_c / num_iterations * 1000  # ms
    bandwidth_c = (cfg.num_layers * cfg.bytes_per_block * num_iterations / 1e9) / elapsed_c

    print(f"\n  Method C (PyTorch .cuda() on contiguous):")
    print(f"    Avg time: {avg_time_c:.3f} ms")
    print(f"    Bandwidth: {bandwidth_c:.2f} GB/s")

    # Summary
    print(f"\n  ========================================")
    print(f"  Performance Summary:")
    print(f"    Method A vs Method B: {bandwidth_a / bandwidth_b:.2f}x speedup")
    print(f"    Method A vs Method C: {bandwidth_a / bandwidth_c * 100:.2f}%")
    print(f"  ========================================")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== CUDA sgDMA (cudaMemcpy2D) Benchmark ===")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        exit(1)

    # Print GPU info
    print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Run benchmark
    benchmark_sgdma()

    print("\n=== Benchmark Complete ===")
