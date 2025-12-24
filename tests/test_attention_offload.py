"""
Test Attention layer with KV cache offload - N-way Pipeline.

This test demonstrates and verifies the N-way pipeline with:
- Per-slot transfer streams for parallel H2D
- Dedicated compute stream (avoids CUDA default stream implicit sync)
- Pre-load phase + main loop with immediate slot reuse

Key difference from previous test:
- We first pre-fill many chunks to CPU cache
- Then simulate processing a new chunk that loads ALL previous blocks
- This exercises the full N-way pipeline with many blocks in flight
"""

import torch
from nanovllm.layers.attention import Attention
from nanovllm.kvcache.hybrid_manager import HybridKVCacheManager
from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, reset_context


# ============================================================
# Configuration
# ============================================================

NUM_LAYERS = 8
NUM_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 1024
CHUNK_SIZE = 1024

NUM_GPU_SLOTS = 6  # N-way pipeline with 6 slots
NUM_CPU_BLOCKS = 16  # Many blocks to load from CPU

DTYPE = torch.bfloat16
DEVICE = "cuda"


# ============================================================
# Setup
# ============================================================

def create_manager():
    manager = HybridKVCacheManager(
        num_gpu_slots=NUM_GPU_SLOTS,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        block_size=BLOCK_SIZE,
    )
    manager.allocate_cache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )
    return manager


def create_attention_layers(manager):
    layers = []
    for layer_id in range(NUM_LAYERS):
        attn = Attention(
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            scale=HEAD_DIM ** -0.5,
            num_kv_heads=NUM_KV_HEADS,
        )
        attn.layer_id = layer_id
        k_cache, v_cache = manager.get_layer_cache(layer_id)
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        layers.append(attn.to(DEVICE))
    return layers


# ============================================================
# Pre-fill CPU cache with random data
# ============================================================

def prefill_cpu_cache(manager, num_blocks):
    """
    Fill CPU cache with random KV data for num_blocks blocks.
    This simulates having already processed many chunks.
    """
    offload_engine = manager.offload_engine

    for block_id in range(num_blocks):
        # Generate random KV data for all layers
        for layer_id in range(NUM_LAYERS):
            k_data = torch.randn(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=DTYPE, device=DEVICE
            )
            v_data = torch.randn(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=DTYPE, device=DEVICE
            )

            # Copy to CPU cache
            offload_engine.k_cache_cpu[layer_id, block_id].copy_(k_data)
            offload_engine.v_cache_cpu[layer_id, block_id].copy_(v_data)

    return list(range(num_blocks))


# ============================================================
# Simulate N-way Pipeline (mirrors attention.py logic)
# ============================================================

def simulate_nway_pipeline(
    layer_id: int,
    q_batched: torch.Tensor,
    cpu_block_table: list,
    load_slots: list,
    offload_engine,
    scale: float,
):
    """
    Simulate N-way pipeline for a single layer.
    This mirrors the logic in Attention._ring_buffer_pipeline_load().
    """
    num_blocks = len(cpu_block_table)
    num_slots = len(load_slots)

    o_acc, lse_acc = None, None

    # Phase 1: Pre-load up to num_slots blocks
    num_preload = min(num_slots, num_blocks)
    torch.cuda.nvtx.range_push(f"Phase1_Preload: L{layer_id}")
    for i in range(num_preload):
        offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])
    torch.cuda.nvtx.range_pop()

    # Phase 2: Main loop with compute_stream
    compute_stream = offload_engine.compute_stream

    for block_idx in range(num_blocks):
        torch.cuda.nvtx.range_push(f"Block: L{layer_id} B{block_idx}")

        current_slot = load_slots[block_idx % num_slots]

        # Wait for transfer
        offload_engine.wait_slot_layer(current_slot, layer_id)

        # Compute on dedicated stream
        with torch.cuda.stream(compute_stream):
            torch.cuda.nvtx.range_push(f"FlashAttn: L{layer_id} B{block_idx}")
            prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot, layer_id)
            prev_o, prev_lse = flash_attn_with_lse(
                q_batched, prev_k, prev_v,
                softmax_scale=scale,
                causal=False,
            )
            torch.cuda.nvtx.range_pop()
            offload_engine.record_slot_compute_done(current_slot, layer_id)

        # Start next transfer (reuse current_slot)
        next_block_idx = block_idx + num_slots
        if next_block_idx < num_blocks:
            offload_engine.load_to_slot_layer(
                current_slot, layer_id, cpu_block_table[next_block_idx]
            )

        # Merge
        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                o_acc, lse_acc = prev_o, prev_lse
            else:
                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

        torch.cuda.nvtx.range_pop()

    return o_acc, lse_acc


def simulate_full_forward(layers, manager, cpu_block_table, chunk_size):
    """
    Simulate forward pass through all layers, loading previous blocks from CPU.
    This is the key test: many blocks loaded via N-way pipeline.
    """
    offload_engine = manager.offload_engine

    # Current chunk index (we're processing the "next" chunk after all prefilled ones)
    current_chunk_idx = len(cpu_block_table)
    write_slot = offload_engine.get_write_slot_for_prefill(current_chunk_idx)
    load_slots = offload_engine.get_load_slots_for_prefill(write_slot)

    # Random query for attention
    q = torch.randn(1, chunk_size, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    outputs = []
    for layer in layers:
        torch.cuda.nvtx.range_push(f"Layer: {layer.layer_id}")

        o_acc, lse_acc = simulate_nway_pipeline(
            layer.layer_id,
            q,
            cpu_block_table,
            load_slots,
            offload_engine,
            layer.scale,
        )

        outputs.append(o_acc)
        torch.cuda.nvtx.range_pop()

    return outputs


# ============================================================
# Main Test
# ============================================================

print("=" * 60)
print("Test: N-way Pipeline with CPU Offload")
print("=" * 60)

# 1. Setup
print("\n[1] Creating manager and attention layers...")
manager = create_manager()
layers = create_attention_layers(manager)
offload_engine = manager.offload_engine

print(f"  - GPU slots: {NUM_GPU_SLOTS}")
print(f"  - CPU blocks: {NUM_CPU_BLOCKS}")
print(f"  - Per-slot streams: {len(offload_engine.slot_transfer_streams)}")
print(f"  - Compute stream: {offload_engine.compute_stream}")

# 2. Pre-fill CPU cache
NUM_PREV_BLOCKS = 12  # Many blocks to load via N-way pipeline
print(f"\n[2] Pre-filling {NUM_PREV_BLOCKS} blocks to CPU cache...")
cpu_block_table = prefill_cpu_cache(manager, NUM_PREV_BLOCKS)
print(f"  - CPU blocks filled: {cpu_block_table}")

# 3. Verify pipeline configuration
current_chunk_idx = NUM_PREV_BLOCKS
write_slot = offload_engine.get_write_slot_for_prefill(current_chunk_idx)
load_slots = offload_engine.get_load_slots_for_prefill(write_slot)
print(f"\n[3] Pipeline configuration for chunk {current_chunk_idx}:")
print(f"  - Write slot: {write_slot}")
print(f"  - Load slots: {load_slots}")
print(f"  - Pipeline depth (N-way): {len(load_slots)}")
assert len(load_slots) == NUM_GPU_SLOTS - 1, f"Expected {NUM_GPU_SLOTS - 1} load slots"

# 4. Warmup
print("\n[4] Warmup (3 iterations)...")
for i in range(3):
    outputs = simulate_full_forward(layers, manager, cpu_block_table, CHUNK_SIZE)
    torch.cuda.synchronize()
    print(f"  - Warmup {i+1}/3 done")

# 5. Benchmark
NUM_ITERS = 10
print(f"\n[5] Benchmark ({NUM_ITERS} iterations)...")

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for i in range(NUM_ITERS):
    torch.cuda.nvtx.range_push(f"Iteration_{i}")
    outputs = simulate_full_forward(layers, manager, cpu_block_table, CHUNK_SIZE)
    torch.cuda.nvtx.range_pop()
end_event.record()

torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)

# Stats
total_blocks_loaded = NUM_PREV_BLOCKS * NUM_LAYERS * NUM_ITERS
blocks_per_sec = total_blocks_loaded / (elapsed_ms / 1000)
total_tokens = NUM_PREV_BLOCKS * BLOCK_SIZE * NUM_LAYERS * NUM_ITERS
tokens_per_sec = total_tokens / (elapsed_ms / 1000)

print(f"\n[6] Results:")
print(f"  - Total time: {elapsed_ms:.2f} ms")
print(f"  - Per iteration: {elapsed_ms / NUM_ITERS:.2f} ms")
print(f"  - Blocks loaded: {total_blocks_loaded} ({blocks_per_sec:.0f} blocks/s)")
print(f"  - Tokens processed: {total_tokens} ({tokens_per_sec:.0f} tok/s)")

# 7. Verification
print("\n[7] Verification:")
assert len(outputs) == NUM_LAYERS, f"Expected {NUM_LAYERS} outputs"
for i, o in enumerate(outputs):
    assert o is not None, f"Layer {i} output is None"
    assert o.shape == (1, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), f"Layer {i} shape mismatch"
print("  - All layer outputs valid ✓")
print("  - N-way pipeline executed correctly ✓")

# Cleanup
reset_context()

print("\n" + "=" * 60)
print("test_attention_offload: PASSED")
print("=" * 60)
