"""
Test script for OffloadEngine - CPU-GPU KV cache transfer engine.

Demonstrates: ring buffer, H2D/D2H transfers, CUDA events, KV access.
"""

import torch
from nanovllm.kvcache.offload_engine import OffloadEngine

# ============================================================
# Utility Functions
# ============================================================

def verify(tensor: torch.Tensor, expected: float, name: str) -> None:
    """Verify tensor contains expected value."""
    actual = tensor.mean().item()
    assert abs(actual - expected) < 0.01, f"{name}: {actual} != {expected}"

# ============================================================
# Configuration
# ============================================================

NUM_LAYERS = 4
NUM_GPU_BLOCKS = 8
NUM_CPU_BLOCKS = 16
BLOCK_SIZE = 64
NUM_KV_HEADS = 4
HEAD_DIM = 32

# ============================================================
# Main Test Script
# ============================================================

# 1. Initialize
engine = OffloadEngine(
    num_layers=NUM_LAYERS,
    num_gpu_blocks=NUM_GPU_BLOCKS,
    num_cpu_blocks=NUM_CPU_BLOCKS,
    block_size=BLOCK_SIZE,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    dtype=torch.float16,
)

# 2. Ring buffer slot management
for chunk_idx in range(12):
    write_slot = engine.get_write_slot_for_prefill(chunk_idx)
    load_slots = engine.get_load_slots_for_prefill(write_slot)
    
    print("chunk idx", chunk_idx, "write slots:", write_slot, "load slots:", load_slots)
    
    assert write_slot == chunk_idx % engine.num_ring_slots
    assert write_slot not in load_slots

assert engine.decode_slot == 0
assert engine.get_load_slots_for_decode() == list(range(1, NUM_GPU_BLOCKS))

# 3. Per-slot per-layer H2D transfer
engine.k_cache_cpu[0, 0].fill_(42.0)
engine.v_cache_cpu[0, 0].fill_(42.5)

engine.load_to_slot_layer(slot_idx=1, layer_id=0, cpu_block_id=0)
engine.wait_slot_layer(slot_idx=1, layer_id=0)

verify(engine.k_cache_gpu[0, 1], 42.0, "H2D K")
verify(engine.v_cache_gpu[0, 1], 42.5, "H2D V")

# 4. Compute-done event (pipeline safety)
engine.record_slot_compute_done(slot_idx=1, layer_id=0)

engine.k_cache_cpu[0, 1].fill_(100.0)
engine.v_cache_cpu[0, 1].fill_(100.5)
engine.load_to_slot_layer(slot_idx=1, layer_id=0, cpu_block_id=1)
engine.wait_slot_layer(slot_idx=1, layer_id=0)

verify(engine.k_cache_gpu[0, 1], 100.0, "Reuse K")
verify(engine.v_cache_gpu[0, 1], 100.5, "Reuse V")

# 5. D2H offload
engine.k_cache_gpu[1, 2].fill_(77.0)
engine.v_cache_gpu[1, 2].fill_(77.5)

engine.offload_slot_to_cpu(slot_idx=2, cpu_block_id=5)
engine.wait_slot_offload(slot_idx=2)

verify(engine.k_cache_cpu[1, 5], 77.0, "D2H K")
verify(engine.v_cache_cpu[1, 5], 77.5, "D2H V")

# 6. KV access methods
k, v = engine.get_kv_for_slot(slot_idx=1, layer_id=0)
assert k.shape == (1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

k, v = engine.get_kv_for_slots(layer_id=0, slot_indices=[0, 1, 2])
assert k.shape == (1, 3 * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

engine.k_cache_gpu[0, engine.decode_slot].fill_(33.0)
k, v = engine.get_kv_for_decode_slot_accumulated(layer_id=0, num_tokens=10)
assert k.shape == (1, 10, NUM_KV_HEADS, HEAD_DIM)
verify(k, 33.0, "Decode slot K")

# 7. Batch transfer
cpu_blocks = [2, 3, 4]
gpu_slots = [3, 4, 5]
for cpu_id in cpu_blocks:
    engine.k_cache_cpu[0, cpu_id].fill_(50.0 + cpu_id)

engine.load_cpu_blocks_to_gpu_slots(layer_id=0, cpu_block_ids=cpu_blocks, gpu_slot_ids=gpu_slots)

for cpu_id, gpu_slot in zip(cpu_blocks, gpu_slots):
    verify(engine.k_cache_gpu[0, gpu_slot], 50.0 + cpu_id, f"Batch slot {gpu_slot}")

# 8. Gather indices (CUDA graph compatible)
engine.update_gather_indices(layer_id=0, mappings=[(0, 0), (1, 1), (2, 2)])
assert engine.gather_indices_gpu[0, :3].tolist() == [0, 1, 2]

engine.clear_gather_indices(layer_id=0)
assert engine.gather_indices_gpu[0, 0].item() == -1

print("test_offload_engine: PASSED")
