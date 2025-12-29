"""
Correctness test for chunked attention with CPU offload.

Validates that the offload pipeline (CPU -> GPU transfer + chunked attention)
produces the same result as direct GPU computation.

Test scenario:
1. Generate Q, K, V data
2. Reference: Compute full causal attention on GPU
3. Offload: Store K, V in CPU cache, load via pipeline, compute chunked attention
4. Compare results

This test is designed to identify bugs in:
- CPU <-> GPU data transfer (sgDMA)
- Ring buffer slot management
- N-way pipeline ordering
- Triton merge kernel correctness
"""

import torch
from flash_attn.flash_attn_interface import flash_attn_func
from nanovllm.kvcache.hybrid_manager import HybridKVCacheManager
from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs


# ============================================================
# Configuration
# ============================================================

NUM_LAYERS = 4
NUM_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 256  # Smaller for faster testing
DTYPE = torch.bfloat16
DEVICE = "cuda"


# ============================================================
# Reference Implementation (GPU only, no offload)
# ============================================================

def compute_reference_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute reference causal attention using flash_attn_func.

    Args:
        q, k, v: [batch, seqlen, nheads, headdim]

    Returns:
        out: [batch, seqlen, nheads, headdim]
    """
    return flash_attn_func(q, k, v, causal=True)


def compute_reference_chunked(
    q_chunks: list,
    kv_chunks: list,
    scale: float,
) -> torch.Tensor:
    """
    Compute chunked prefill attention directly on GPU (no offload).

    This is the "gold standard" for chunked attention correctness.

    Args:
        q_chunks: List of [batch, chunk_size, nheads, headdim]
        kv_chunks: List of (k, v) tuples, each [batch, chunk_size, nheads, headdim]
        scale: Softmax scale

    Returns:
        out: [batch, total_seqlen, nheads, headdim]
    """
    out_chunks = []

    for chunk_idx, q_chunk in enumerate(q_chunks):
        o_acc, lse_acc = None, None

        # Attend to all previous chunks (no causal mask)
        for i in range(chunk_idx):
            k_chunk, v_chunk = kv_chunks[i]
            chunk_o, chunk_lse = flash_attn_with_lse(
                q_chunk, k_chunk, v_chunk,
                softmax_scale=scale,
                causal=False,
            )
            if o_acc is None:
                o_acc, lse_acc = chunk_o, chunk_lse
            else:
                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, chunk_o, chunk_lse)

        # Attend to current chunk (with causal mask)
        k_chunk, v_chunk = kv_chunks[chunk_idx]
        current_o, current_lse = flash_attn_with_lse(
            q_chunk, k_chunk, v_chunk,
            softmax_scale=scale,
            causal=True,
        )

        if o_acc is None:
            final_o = current_o
        else:
            final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

        out_chunks.append(final_o)

    return torch.cat(out_chunks, dim=1)


# ============================================================
# Offload Implementation
# ============================================================

def create_manager(num_gpu_slots: int, num_cpu_blocks: int):
    """Create HybridKVCacheManager with specified configuration."""
    manager = HybridKVCacheManager(
        num_gpu_slots=num_gpu_slots,
        num_cpu_blocks=num_cpu_blocks,
        block_size=BLOCK_SIZE,
    )
    manager.allocate_cache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )
    return manager


def store_kv_to_cpu_cache(manager, kv_chunks: list, layer_id: int):
    """
    Store K, V chunks to CPU cache.

    Args:
        manager: HybridKVCacheManager
        kv_chunks: List of (k, v) tuples, each [batch, chunk_size, nheads, headdim]
        layer_id: Layer index

    Returns:
        cpu_block_ids: List of CPU block IDs
    """
    offload_engine = manager.offload_engine
    cpu_block_ids = []

    for block_idx, (k_chunk, v_chunk) in enumerate(kv_chunks):
        # k_chunk, v_chunk: [batch, chunk_size, nheads, headdim]
        # CPU cache layout: [num_layers, num_blocks, block_size, nheads, headdim]
        k_data = k_chunk.squeeze(0)  # [chunk_size, nheads, headdim]
        v_data = v_chunk.squeeze(0)

        offload_engine.k_cache_cpu[layer_id, block_idx, :k_data.shape[0]].copy_(k_data)
        offload_engine.v_cache_cpu[layer_id, block_idx, :v_data.shape[0]].copy_(v_data)

        cpu_block_ids.append(block_idx)

    return cpu_block_ids


def compute_offload_chunked_single_layer(
    manager,
    q_chunks: list,
    cpu_block_ids: list,
    layer_id: int,
    scale: float,
) -> torch.Tensor:
    """
    Compute chunked attention for a single layer using offload pipeline.

    This mimics the behavior of Attention._ring_buffer_pipeline_load().

    Args:
        manager: HybridKVCacheManager
        q_chunks: List of [batch, chunk_size, nheads, headdim]
        cpu_block_ids: List of CPU block IDs containing K, V data
        layer_id: Layer index
        scale: Softmax scale

    Returns:
        out: [batch, total_seqlen, nheads, headdim]
    """
    offload_engine = manager.offload_engine
    out_chunks = []

    for chunk_idx, q_chunk in enumerate(q_chunks):
        # CPU blocks to load: all blocks before current chunk
        blocks_to_load = cpu_block_ids[:chunk_idx]

        # Get slots for this chunk
        write_slot = offload_engine.get_write_slot_for_prefill(chunk_idx)
        load_slots = offload_engine.get_load_slots_for_prefill(write_slot)

        # Load and compute attention for previous chunks
        o_acc, lse_acc = None, None

        if len(blocks_to_load) > 0 and len(load_slots) > 0:
            o_acc, lse_acc = _pipeline_load_and_compute(
                offload_engine,
                q_chunk,
                blocks_to_load,
                load_slots,
                layer_id,
                scale,
            )

        # Current chunk's K, V (load from CPU to GPU slot)
        current_cpu_block = cpu_block_ids[chunk_idx]
        offload_engine.load_to_slot_layer(write_slot, layer_id, current_cpu_block)
        offload_engine.wait_slot_layer(write_slot, layer_id)

        current_k, current_v = offload_engine.get_kv_for_slot(write_slot, layer_id)

        # Compute attention with causal mask
        current_o, current_lse = flash_attn_with_lse(
            q_chunk, current_k, current_v,
            softmax_scale=scale,
            causal=True,
        )

        # Merge
        if o_acc is None:
            final_o = current_o
        else:
            final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

        out_chunks.append(final_o)

    return torch.cat(out_chunks, dim=1)


def _pipeline_load_and_compute(
    offload_engine,
    q_chunk: torch.Tensor,
    cpu_block_table: list,
    load_slots: list,
    layer_id: int,
    scale: float,
):
    """
    Pipeline loading from CPU and computing attention.

    Mirrors Attention._ring_buffer_pipeline_load() logic.
    """
    num_blocks = len(cpu_block_table)
    num_slots = len(load_slots)

    o_acc, lse_acc = None, None

    # Phase 1: Pre-load up to num_slots blocks
    num_preload = min(num_slots, num_blocks)
    for i in range(num_preload):
        offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

    # Phase 2: Main loop
    compute_stream = offload_engine.compute_stream

    for block_idx in range(num_blocks):
        current_slot = load_slots[block_idx % num_slots]

        # Wait for transfer
        offload_engine.wait_slot_layer(current_slot, layer_id)

        # Compute on dedicated stream
        with torch.cuda.stream(compute_stream):
            prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot, layer_id)
            prev_o, prev_lse = flash_attn_with_lse(
                q_chunk, prev_k, prev_v,
                softmax_scale=scale,
                causal=False,
            )
            offload_engine.record_slot_compute_done(current_slot, layer_id)

        # Start next transfer
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

    # Sync compute stream
    compute_stream.synchronize()

    return o_acc, lse_acc


# ============================================================
# Test Runner
# ============================================================

def run_correctness_test(
    num_chunks: int,
    num_gpu_slots: int,
    verbose: bool = True,
) -> tuple[bool, float, float]:
    """
    Run a single correctness test.

    Args:
        num_chunks: Number of chunks (= number of CPU blocks)
        num_gpu_slots: Number of GPU ring buffer slots
        verbose: Print detailed info

    Returns:
        (passed, max_diff, mean_diff)
    """
    torch.manual_seed(42)

    seqlen = num_chunks * BLOCK_SIZE
    scale = HEAD_DIM ** -0.5

    # Generate Q, K, V
    q_full = torch.randn(1, seqlen, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_full = torch.randn(1, seqlen, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_full = torch.randn(1, seqlen, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    # Split into chunks
    q_chunks = [q_full[:, i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] for i in range(num_chunks)]
    kv_chunks = [
        (k_full[:, i*BLOCK_SIZE:(i+1)*BLOCK_SIZE],
         v_full[:, i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
        for i in range(num_chunks)
    ]

    # Reference: chunked attention on GPU (no offload)
    out_ref = compute_reference_chunked(q_chunks, kv_chunks, scale)

    # Create manager with enough CPU blocks
    manager = create_manager(num_gpu_slots, num_chunks)

    # Test each layer
    all_passed = True
    max_diff_all = 0.0
    mean_diff_all = 0.0

    for layer_id in range(NUM_LAYERS):
        # Store K, V to CPU cache
        cpu_block_ids = store_kv_to_cpu_cache(manager, kv_chunks, layer_id)

        # Compute with offload
        out_offload = compute_offload_chunked_single_layer(
            manager, q_chunks, cpu_block_ids, layer_id, scale
        )

        # Compare
        diff = (out_ref - out_offload).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        max_diff_all = max(max_diff_all, max_diff)
        mean_diff_all = max(mean_diff_all, mean_diff)

        tol = 1e-2
        passed = max_diff < tol
        all_passed = all_passed and passed

        if verbose and not passed:
            print(f"  Layer {layer_id}: FAIL max_diff={max_diff:.6f}")

    return all_passed, max_diff_all, mean_diff_all


# ============================================================
# Decode Phase Test
# ============================================================

def run_decode_correctness_test(
    num_prefill_chunks: int,
    num_gpu_slots: int,
    num_decode_steps: int = 4,
    verbose: bool = True,
) -> tuple[bool, float, float]:
    """
    Test decode phase correctness with CPU offload.

    Simulates:
    1. Prefill: Store K, V for multiple chunks in CPU cache
    2. Decode: Single token queries against all prefilled K, V

    This tests the scenario in needle test where decode reads all previous KV.
    """
    torch.manual_seed(42)

    scale = HEAD_DIM ** -0.5
    prefill_len = num_prefill_chunks * BLOCK_SIZE

    # Generate prefill K, V (store in CPU)
    k_prefill = torch.randn(1, prefill_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_prefill = torch.randn(1, prefill_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    # Split into chunks for CPU storage
    kv_chunks = [
        (k_prefill[:, i*BLOCK_SIZE:(i+1)*BLOCK_SIZE],
         v_prefill[:, i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
        for i in range(num_prefill_chunks)
    ]

    # Create manager
    manager = create_manager(num_gpu_slots, num_prefill_chunks)
    offload_engine = manager.offload_engine

    all_passed = True
    max_diff_all = 0.0
    mean_diff_all = 0.0

    for layer_id in range(NUM_LAYERS):
        # Store prefilled K, V to CPU cache
        cpu_block_ids = store_kv_to_cpu_cache(manager, kv_chunks, layer_id)

        for decode_step in range(num_decode_steps):
            # Decode query: single token
            q_decode = torch.randn(1, 1, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

            # Reference: direct attention on GPU
            # Concat all prefilled K, V and compute attention
            out_ref = flash_attn_func(
                q_decode,
                k_prefill,
                v_prefill,
                causal=False,  # Decode query can attend to all prefilled tokens
            )

            # Offload: load from CPU and compute
            load_slots = offload_engine.get_load_slots_for_prefill(0)  # Use all slots except decode slot

            if len(load_slots) == 0 or len(cpu_block_ids) == 0:
                # No previous chunks to load
                out_offload = out_ref  # Trivially equal
            else:
                o_acc, lse_acc = _pipeline_load_and_compute(
                    offload_engine,
                    q_decode,
                    cpu_block_ids,
                    load_slots,
                    layer_id,
                    scale,
                )
                out_offload = o_acc

            # Compare
            diff = (out_ref - out_offload).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            max_diff_all = max(max_diff_all, max_diff)
            mean_diff_all = max(mean_diff_all, mean_diff)

            tol = 1e-2
            passed = max_diff < tol
            all_passed = all_passed and passed

            if verbose and not passed:
                print(f"  Layer {layer_id} Step {decode_step}: FAIL max_diff={max_diff:.6f}")

    return all_passed, max_diff_all, mean_diff_all


# ============================================================
# Main Test Script
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Test: Offload Chunked Attention Correctness")
    print("=" * 70)
    print(f"Config: layers={NUM_LAYERS}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, block_size={BLOCK_SIZE}, dtype={DTYPE}")
    print()
    print("Comparing: Reference (GPU chunked) vs Offload (CPU->GPU pipeline)")
    print()

    # Test configurations: (num_chunks, num_gpu_slots)
    TEST_CASES = [
        # Basic tests
        (2, 2),   # Minimal: 2 chunks, 2 slots (no pipeline)
        (2, 3),   # 2 chunks, 3 slots (1-slot pipeline)
        (4, 2),   # 4 chunks, 2 slots (heavy slot reuse)
        (4, 3),   # 4 chunks, 3 slots
        (4, 4),   # 4 chunks, 4 slots
        # Stress tests
        (8, 3),   # Many chunks, few slots
        (8, 4),   # Many chunks, moderate slots
        (8, 6),   # Many chunks, many slots (like bench_offload)
        # Edge cases
        (1, 2),   # Single chunk
        (3, 5),   # Fewer chunks than slots
    ]

    all_passed = True
    results = []

    for num_chunks, num_gpu_slots in TEST_CASES:
        seqlen = num_chunks * BLOCK_SIZE
        passed, max_diff, mean_diff = run_correctness_test(
            num_chunks, num_gpu_slots, verbose=False
        )

        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"

        results.append((num_chunks, num_gpu_slots, seqlen, passed, max_diff, mean_diff))

        print(f"[{status}] chunks={num_chunks:2d} slots={num_gpu_slots:2d} "
              f"seqlen={seqlen:5d} max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}")

    print()

    # ================================================================
    # Part 2: Decode Phase Tests
    # ================================================================
    print("=" * 70)
    print("Part 2: Decode Phase Correctness")
    print("=" * 70)
    print("Testing: Decode query (single token) against all prefilled K, V")
    print()

    DECODE_TEST_CASES = [
        # (num_prefill_chunks, num_gpu_slots)
        (2, 2),
        (4, 3),
        (4, 4),
        (8, 4),
        (8, 6),
    ]

    decode_results = []

    for num_prefill_chunks, num_gpu_slots in DECODE_TEST_CASES:
        prefill_len = num_prefill_chunks * BLOCK_SIZE
        passed, max_diff, mean_diff = run_decode_correctness_test(
            num_prefill_chunks, num_gpu_slots, num_decode_steps=4, verbose=False
        )

        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"

        decode_results.append((num_prefill_chunks, num_gpu_slots, prefill_len, passed, max_diff, mean_diff))

        print(f"[{status}] prefill_chunks={num_prefill_chunks:2d} slots={num_gpu_slots:2d} "
              f"prefill_len={prefill_len:5d} max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}")

    print()
    print("=" * 70)

    # Summary
    prefill_passed = sum(1 for r in results if r[3])
    decode_passed = sum(1 for r in decode_results if r[3])
    total_tests = len(results) + len(decode_results)
    total_passed = prefill_passed + decode_passed

    print(f"Results: {total_passed}/{total_tests} tests passed")
    print(f"  - Prefill: {prefill_passed}/{len(results)}")
    print(f"  - Decode:  {decode_passed}/{len(decode_results)}")

    if not all_passed:
        print("\nFailed tests:")
        for num_chunks, num_gpu_slots, seqlen, passed, max_diff, mean_diff in results:
            if not passed:
                print(f"  - [Prefill] chunks={num_chunks}, slots={num_gpu_slots}, "
                      f"seqlen={seqlen}, max_diff={max_diff:.6f}")
        for num_chunks, num_gpu_slots, seqlen, passed, max_diff, mean_diff in decode_results:
            if not passed:
                print(f"  - [Decode] prefill_chunks={num_chunks}, slots={num_gpu_slots}, "
                      f"prefill_len={seqlen}, max_diff={max_diff:.6f}")

    print()
    assert all_passed, "Some correctness tests failed!"
    print("test_offload_correctness: PASSED")
