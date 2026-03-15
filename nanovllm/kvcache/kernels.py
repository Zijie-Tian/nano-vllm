"""
Triton kernels for CPU-GPU KV cache transfer.

These kernels are designed to be CUDA Graph compatible:
- All tensor addresses are fixed at graph capture time
- Only the content of index tensors changes between replays
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gathered_copy_kernel(
    src_ptr,  # Source tensor base pointer (CPU pinned or GPU)
    dst_ptr,  # Destination tensor base pointer (GPU)
    indices_ptr,  # Gather indices [num_dst_blocks]
    num_dst_blocks,  # Number of destination blocks
    block_numel: tl.constexpr,  # Elements per block (block_size * kv_heads * head_dim)
    BLOCK_SIZE: tl.constexpr = 1024,
):
    """
    Gathered copy kernel: dst[i] = src[indices[i]]

    Each program instance handles one destination block.
    The indices tensor specifies which source block to copy from.

    This kernel is CUDA Graph compatible because:
    - src_ptr, dst_ptr, indices_ptr addresses are fixed
    - Only indices content changes between graph replays

    Args:
        src_ptr: Base pointer to source blocks [num_src_blocks, block_numel]
        dst_ptr: Base pointer to destination blocks [num_dst_blocks, block_numel]
        indices_ptr: Gather indices [num_dst_blocks], each value is a source block index
        num_dst_blocks: Number of destination blocks to copy
        block_numel: Number of elements per block
        BLOCK_SIZE: Triton block size for parallelization
    """
    dst_block_idx = tl.program_id(0)

    # Skip if out of range
    if dst_block_idx >= num_dst_blocks:
        return

    # Load source block index from indices tensor
    src_block_idx = tl.load(indices_ptr + dst_block_idx)

    # Skip if index is -1 (invalid/no-op marker)
    if src_block_idx < 0:
        return

    # Calculate base offsets
    src_base = src_block_idx * block_numel
    dst_base = dst_block_idx * block_numel

    # Copy block data in chunks of BLOCK_SIZE
    for start in range(0, block_numel, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_numel

        # Load from source and store to destination
        data = tl.load(src_ptr + src_base + offsets, mask=mask)
        tl.store(dst_ptr + dst_base + offsets, data, mask=mask)


@triton.jit
def gathered_copy_kv_kernel(
    k_src_ptr,  # K cache source [num_src_blocks, block_size, kv_heads, head_dim]
    v_src_ptr,  # V cache source
    k_dst_ptr,  # K cache destination
    v_dst_ptr,  # V cache destination
    indices_ptr,  # Gather indices [num_dst_blocks]
    num_dst_blocks,  # Number of destination blocks
    block_numel: tl.constexpr,  # Elements per block
    BLOCK_SIZE: tl.constexpr = 1024,
):
    """
    Gathered copy for both K and V caches simultaneously.

    More efficient than calling gathered_copy_kernel twice because:
    - Single kernel launch overhead
    - Better memory access patterns when K and V are accessed together
    """
    dst_block_idx = tl.program_id(0)

    if dst_block_idx >= num_dst_blocks:
        return

    src_block_idx = tl.load(indices_ptr + dst_block_idx)

    if src_block_idx < 0:
        return

    src_base = src_block_idx * block_numel
    dst_base = dst_block_idx * block_numel

    for start in range(0, block_numel, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_numel

        # Copy K cache
        k_data = tl.load(k_src_ptr + src_base + offsets, mask=mask)
        tl.store(k_dst_ptr + dst_base + offsets, k_data, mask=mask)

        # Copy V cache
        v_data = tl.load(v_src_ptr + src_base + offsets, mask=mask)
        tl.store(v_dst_ptr + dst_base + offsets, v_data, mask=mask)

