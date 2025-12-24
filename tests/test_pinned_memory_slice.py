"""
Test if slicing maintains pinned memory property.
"""

import torch

print("=" * 60)
print("Test: Pinned Memory Property with Slicing")
print("=" * 60)

# Create a pinned tensor with shape similar to k_cache_cpu
# [num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim]
tensor = torch.zeros(8, 16, 1024, 8, 64, dtype=torch.float16, device="cpu", pin_memory=True)

print(f"\n1. Original tensor:")
print(f"   - Shape: {tensor.shape}")
print(f"   - is_pinned(): {tensor.is_pinned()}")
print(f"   - is_contiguous(): {tensor.is_contiguous()}")

# Test slicing operation (what we do in offload_slot_to_cpu)
slice_view = tensor[:, 0]  # Same as k_cache_cpu[:, cpu_block_id]

print(f"\n2. Sliced tensor [:, 0]:")
print(f"   - Shape: {slice_view.shape}")
print(f"   - is_pinned(): {slice_view.is_pinned()}")
print(f"   - is_contiguous(): {slice_view.is_contiguous()}")

# Test if contiguous() helps
contiguous_slice = tensor[:, 0].contiguous()

print(f"\n3. Contiguous slice [:, 0].contiguous():")
print(f"   - Shape: {contiguous_slice.shape}")
print(f"   - is_pinned(): {contiguous_slice.is_pinned()}")
print(f"   - is_contiguous(): {contiguous_slice.is_contiguous()}")

# Test copy behavior
gpu_tensor = torch.zeros(8, 4, 1024, 8, 64, dtype=torch.float16, device="cuda")
gpu_slice = gpu_tensor[:, 0]

print(f"\n4. GPU tensor slice:")
print(f"   - Shape: {gpu_slice.shape}")
print(f"   - is_contiguous(): {gpu_slice.is_contiguous()}")

# Simulate the problematic copy operation
print(f"\n5. Testing copy operations:")

# Method 1: Direct slice copy (current approach - SLOW)
slice_dst = tensor[:, 1]
print(f"   Method 1 (slice view): dst.is_pinned()={slice_dst.is_pinned()}")

# Method 2: Use contiguous destination
contiguous_dst = tensor[:, 2].contiguous()
print(f"   Method 2 (contiguous): dst.is_pinned()={contiguous_dst.is_pinned()}")

print("\n" + "=" * 60)
print("Conclusion:")
print("=" * 60)

if not slice_view.is_pinned():
    print("❌ Slicing LOSES pinned memory property!")
    print("   This causes Device-to-Pageable transfers (SLOW)")
else:
    print("✓ Slicing maintains pinned memory property")

if contiguous_slice.is_pinned():
    print("✓ .contiguous() maintains pinned memory property")
else:
    print("❌ .contiguous() also loses pinned memory property")

print("\n" + "=" * 60)
