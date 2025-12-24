"""
Test Attention layer with KV cache offload in isolation.

This test demonstrates how to use Attention + HybridKVCacheManager directly
without requiring full LLMEngine/ModelRunner setup.
"""

import torch
from nanovllm.layers.attention import Attention
from nanovllm.kvcache.hybrid_manager import HybridKVCacheManager
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, reset_context


# ============================================================
# Configuration
# ============================================================

NUM_LAYERS = 8  # Multi-layer for realistic profiling
NUM_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 1024  # tokens per block
CHUNK_SIZE = 1024  # tokens per chunk (same as block for simplicity)

NUM_GPU_SLOTS = 4
NUM_CPU_BLOCKS = 16

DTYPE = torch.float16
DEVICE = "cuda"


# ============================================================
# Setup: Create Manager and Attention Layers
# ============================================================

def create_manager():
    """Create and initialize HybridKVCacheManager with OffloadEngine."""
    manager = HybridKVCacheManager(
        num_gpu_slots=NUM_GPU_SLOTS,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        block_size=BLOCK_SIZE,
    )

    # Initialize offload engine (this creates k_cache_gpu/cpu, v_cache_gpu/cpu)
    manager.allocate_cache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )

    return manager


def create_attention_layers(manager):
    """Create attention layers and bind KV cache."""
    layers = []
    for layer_id in range(NUM_LAYERS):
        attn = Attention(
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            scale=HEAD_DIM ** -0.5,
            num_kv_heads=NUM_KV_HEADS,
        )
        attn.layer_id = layer_id

        # Bind KV cache from manager
        k_cache, v_cache = manager.get_layer_cache(layer_id)
        attn.k_cache = k_cache
        attn.v_cache = v_cache

        layers.append(attn.to(DEVICE))

    return layers


def create_test_sequence(manager, num_chunks=3):
    """Create a test sequence and allocate blocks."""
    total_tokens = num_chunks * CHUNK_SIZE

    # Sequence only takes token_ids
    seq = Sequence(token_ids=list(range(total_tokens)))

    # Set block_size for this test
    seq.block_size = BLOCK_SIZE

    # Allocate blocks (will be on CPU in CPU-primary mode)
    manager.allocate(seq)

    return seq


# ============================================================
# Chunked Prefill Simulation
# ============================================================

def simulate_chunk_forward(
    layers,
    manager,
    seq,
    chunk_idx,
    chunk_size,
):
    """
    Simulate forward pass for one chunk through all layers.

    Returns:
        output: Final layer attention output
    """
    # Generate random Q, K, V for this chunk
    hidden = torch.randn(chunk_size, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # Build slot_mapping: maps token positions to GPU slots
    write_slot = manager.offload_engine.get_write_slot_for_prefill(chunk_idx)
    slot_mapping = torch.full((chunk_size,), write_slot * BLOCK_SIZE, dtype=torch.long, device=DEVICE)
    slot_mapping += torch.arange(chunk_size, device=DEVICE)

    # Build cu_seqlens for flash attention
    cu_seqlens = torch.tensor([0, chunk_size], dtype=torch.int32, device=DEVICE)

    # Set context for this chunk
    set_context(
        is_prefill=True,
        is_chunked_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=chunk_size,
        max_seqlen_k=chunk_size,
        slot_mapping=slot_mapping,
        kvcache_manager=manager,
        chunked_seq=seq,
        current_chunk_idx=chunk_idx,
    )

    # Forward through all layers
    output = hidden
    for layer in layers:
        k = torch.randn(chunk_size, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v = torch.randn(chunk_size, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        output = layer.forward(output, k, v)

    # Offload current chunk to CPU
    logical_id = seq.block_table[chunk_idx]
    cpu_block_id = manager.logical_blocks[logical_id].cpu_block_id
    manager.offload_engine.offload_slot_to_cpu(write_slot, cpu_block_id)
    manager.prefilled_blocks.add(logical_id)

    return output


# ============================================================
# Main Test
# ============================================================

print("=" * 60)
print("Test: Attention Layer with KV Cache Offload")
print("=" * 60)

# 1. Setup
print("\n[1] Creating manager and attention layers...")
manager = create_manager()
layers = create_attention_layers(manager)
print(f"  - Manager: {NUM_GPU_SLOTS} GPU slots, {NUM_CPU_BLOCKS} CPU blocks")
print(f"  - Layers: {NUM_LAYERS} layers, {NUM_HEADS} heads, {HEAD_DIM} head_dim")
print(f"  - OffloadEngine initialized: {manager.offload_engine is not None}")

# 2. Setup
print("\n[2] Test configuration...")
NUM_CHUNKS = NUM_CPU_BLOCKS  # Use all CPU blocks
print(f"  - Total tokens: {NUM_CHUNKS * CHUNK_SIZE}")
print(f"  - Chunks: {NUM_CHUNKS}")

# 3. Warmup runs
print(f"\n[3] Warmup runs (3 iterations)...")
for warmup_iter in range(3):
    manager.prefilled_blocks.clear()
    seq = create_test_sequence(manager, num_chunks=NUM_CHUNKS)

    for chunk_idx in range(NUM_CHUNKS):
        write_slot = manager.offload_engine.get_write_slot_for_prefill(chunk_idx)
        output = simulate_chunk_forward(layers, manager, seq, chunk_idx, CHUNK_SIZE)

    manager.deallocate(seq)
    print(f"  - Warmup {warmup_iter + 1}/3 completed")

# 4. Benchmark runs
print(f"\n[4] Benchmark runs (10 iterations)...")
for bench_iter in range(10):
    manager.prefilled_blocks.clear()
    seq = create_test_sequence(manager, num_chunks=NUM_CHUNKS)

    for chunk_idx in range(NUM_CHUNKS):
        write_slot = manager.offload_engine.get_write_slot_for_prefill(chunk_idx)
        load_slots = manager.offload_engine.get_load_slots_for_prefill(write_slot)
        output = simulate_chunk_forward(layers, manager, seq, chunk_idx, CHUNK_SIZE)

    manager.deallocate(seq)
    print(f"  - Iteration {bench_iter + 1}/10 completed")

# 5. Verify results (using last iteration's seq)
print("\n[5] Verifying ring buffer and offload...")
for chunk_idx in range(NUM_CHUNKS):
    expected_slot = chunk_idx % NUM_GPU_SLOTS
    actual_slot = manager.offload_engine.get_write_slot_for_prefill(chunk_idx)
    assert actual_slot == expected_slot, f"Chunk {chunk_idx}: expected slot {expected_slot}, got {actual_slot}"

cpu_block_table = manager.get_prefilled_cpu_blocks(seq)
assert cpu_block_table == seq.block_table[:NUM_CHUNKS], "CPU block table mismatch"
print("  - Ring buffer cycling verified ✓")
print("  - CPU offload verified ✓")

# Cleanup
manager.deallocate(seq)

# Cleanup
reset_context()

print("\n" + "=" * 60)
print("test_attention_offload: PASSED")
print("=" * 60)
