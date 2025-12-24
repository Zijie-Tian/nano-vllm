"""
Test D2H transfer performance with pinned vs non-contiguous memory.
"""

import torch
import time

print("=" * 60)
print("Test: D2H Transfer Performance (for nsys profiling)")
print("=" * 60)

# Setup
num_layers = 8
num_blocks = 16
block_size = 1024
num_kv_heads = 8
head_dim = 64

# Allocate CPU cache (pinned)
k_cache_cpu = torch.zeros(
    num_layers, num_blocks, block_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cpu", pin_memory=True
)

# Allocate GPU cache
k_cache_gpu = torch.randn(
    num_layers, 4, block_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cuda"
)

# Warmup
print("\nWarmup...")
for _ in range(10):
    k_cache_cpu[:, 0].copy_(k_cache_gpu[:, 0], non_blocking=True)
    torch.cuda.synchronize()

print(f"\nTensor info:")
print(f"  k_cache_cpu.is_pinned(): {k_cache_cpu.is_pinned()}")
print(f"  k_cache_cpu.is_contiguous(): {k_cache_cpu.is_contiguous()}")
print(f"  k_cache_cpu[:, 0].is_pinned(): {k_cache_cpu[:, 0].is_pinned()}")
print(f"  k_cache_cpu[:, 0].is_contiguous(): {k_cache_cpu[:, 0].is_contiguous()}")

# Test 1: Non-contiguous slice (current approach)
print(f"\n" + "=" * 60)
print("Test 1: Non-contiguous slice copy (current approach)")
print("=" * 60)

NUM_ITERS = 50  # Reduced for profiling

torch.cuda.nvtx.range_push("Test1_NonContiguous")
times = []
for i in range(NUM_ITERS):
    torch.cuda.nvtx.range_push(f"D2H_NonContig_{i}")
    start = time.perf_counter()
    k_cache_cpu[:, i % num_blocks].copy_(k_cache_gpu[:, 0], non_blocking=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time * 1000:.3f} ms")
print(f"Bandwidth: {k_cache_gpu[:, 0].numel() * 2 / avg_time / 1e9:.2f} GB/s")

# Test 2: Transpose to make dimension contiguous
print(f"\n" + "=" * 60)
print("Test 2: Transpose to contiguous dimension")
print("=" * 60)

# Reshape to [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
k_cache_cpu_transposed = torch.zeros(
    num_blocks, num_layers, block_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cpu", pin_memory=True
)

print(f"  k_cache_cpu_transposed[0].is_pinned(): {k_cache_cpu_transposed[0].is_pinned()}")
print(f"  k_cache_cpu_transposed[0].is_contiguous(): {k_cache_cpu_transposed[0].is_contiguous()}")

torch.cuda.nvtx.range_push("Test2_Contiguous")
times = []
for i in range(NUM_ITERS):
    torch.cuda.nvtx.range_push(f"D2H_Contig_{i}")
    start = time.perf_counter()
    k_cache_cpu_transposed[i % num_blocks].copy_(k_cache_gpu[:, 0], non_blocking=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time * 1000:.3f} ms")
print(f"Bandwidth: {k_cache_gpu[:, 0].numel() * 2 / avg_time / 1e9:.2f} GB/s")

# Test 3: Fully contiguous buffer
print(f"\n" + "=" * 60)
print("Test 3: Fully contiguous buffer")
print("=" * 60)

k_cache_cpu_flat = torch.zeros(
    num_layers * block_size * num_kv_heads * head_dim,
    dtype=torch.float16, device="cpu", pin_memory=True
)

print(f"  k_cache_cpu_flat.is_pinned(): {k_cache_cpu_flat.is_pinned()}")
print(f"  k_cache_cpu_flat.is_contiguous(): {k_cache_cpu_flat.is_contiguous()}")

torch.cuda.nvtx.range_push("Test3_FlatContiguous")
times = []
for i in range(NUM_ITERS):
    torch.cuda.nvtx.range_push(f"D2H_Flat_{i}")
    start = time.perf_counter()
    k_cache_cpu_flat.copy_(k_cache_gpu[:, 0].flatten(), non_blocking=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time * 1000:.3f} ms")
print(f"Bandwidth: {k_cache_cpu_flat.numel() * 2 / avg_time / 1e9:.2f} GB/s")

print("\n" + "=" * 60)
print("test_pinned_transfer: PASSED")
print("=" * 60)
