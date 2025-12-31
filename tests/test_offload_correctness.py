"""
Test script to verify CPU offload correctness using distinctive KV patterns.

Strategy:
1. Hook into attention forward pass
2. Overwrite K/V with distinctive patterns based on chunk_idx (e.g., K=chunk_idx, V=-chunk_idx)
3. After offload to CPU, verify CPU cache contains correct patterns
4. On subsequent chunks, verify loaded KV from CPU has correct patterns

This catches bugs like:
- Wrong block being offloaded
- Wrong block being loaded
- Data corruption during transfer
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
MAX_MODEL_LEN = 64 * 1024
NUM_GPU_BLOCKS = 4
INPUT_LEN = 32 * 1024  # 32K tokens = 32 chunks (fits in 40 CPU blocks)
BLOCK_SIZE = 1024

# Test state
errors = []
chunk_patterns = {}  # chunk_idx -> (k_pattern, v_pattern)
block_coverage = {}  # chunk_idx -> set of blocks that were actually computed
load_operations = []  # List of (chunk_idx, slot_id, cpu_block_id, k_ok, v_ok) tuples
current_chunk_for_load = [0]  # Mutable container to track current chunk during loads


# ============================================================
# Pattern Helpers
# ============================================================

def get_expected_pattern(chunk_idx: int):
    """Get expected K/V pattern for a chunk."""
    # Use float values that are easy to identify
    k_val = float(chunk_idx + 1)       # 1.0, 2.0, 3.0, ...
    v_val = float(-(chunk_idx + 1))    # -1.0, -2.0, -3.0, ...
    return k_val, v_val


def fill_with_pattern(tensor: torch.Tensor, value: float):
    """Fill tensor with a constant value."""
    tensor.fill_(value)


def check_pattern(tensor: torch.Tensor, expected: float, name: str, tolerance: float = 1e-3):
    """Check if tensor contains expected pattern."""
    actual_mean = tensor.float().mean().item()
    if abs(actual_mean - expected) > tolerance:
        return False, f"{name}: expected mean={expected}, got {actual_mean}"
    return True, None


# ============================================================
# Load Verification Instrumentation
# ============================================================

_original_load_to_slot_layer = None
_offload_engine_ref = None

def make_verified_load_to_slot_layer(original_func, offload_engine):
    """
    Create a wrapper around load_to_slot_layer that verifies each load operation.

    After each H2D transfer, checks that the GPU slot contains the expected
    pattern from the source CPU block.
    """
    def verified_load(slot_idx: int, layer_id: int, cpu_block_id: int):
        # Call original load
        original_func(slot_idx, layer_id, cpu_block_id)

        # Only verify layer 0 to reduce overhead
        if layer_id != 0:
            return

        # IMPORTANT: Synchronize CUDA to ensure async transfer is complete
        # The transfer happens on a per-slot stream, and wait_slot_layer only
        # makes compute_stream wait. We need full sync to read on default stream.
        torch.cuda.synchronize()

        # Get the expected pattern for this CPU block
        # cpu_block_id == chunk_idx in our sequential test
        expected_k, expected_v = get_expected_pattern(cpu_block_id)

        # Read GPU slot data (GPU cache has no layer dimension)
        gpu_k = offload_engine.k_cache_gpu[slot_idx]
        gpu_v = offload_engine.v_cache_gpu[slot_idx]

        actual_k = gpu_k.float().mean().item()
        actual_v = gpu_v.float().mean().item()

        k_ok = abs(actual_k - expected_k) < 1e-3
        v_ok = abs(actual_v - expected_v) < 1e-3

        chunk_idx = current_chunk_for_load[0]
        load_operations.append({
            'chunk_idx': chunk_idx,
            'slot_idx': slot_idx,
            'cpu_block_id': cpu_block_id,
            'expected_k': expected_k,
            'expected_v': expected_v,
            'actual_k': actual_k,
            'actual_v': actual_v,
            'k_ok': k_ok,
            'v_ok': v_ok,
        })

        if not (k_ok and v_ok):
            errors.append(f"Load verification failed: chunk {chunk_idx}, "
                         f"CPU block {cpu_block_id} -> GPU slot {slot_idx}: "
                         f"expected K={expected_k:.1f}/V={expected_v:.1f}, "
                         f"got K={actual_k:.4f}/V={actual_v:.4f}")

    return verified_load


def install_load_verification(llm):
    """Install verification wrapper on load_to_slot_layer."""
    global _original_load_to_slot_layer, _offload_engine_ref

    oe = llm.model_runner.kvcache_manager.offload_engine
    _offload_engine_ref = oe
    _original_load_to_slot_layer = oe.load_to_slot_layer

    oe.load_to_slot_layer = make_verified_load_to_slot_layer(
        _original_load_to_slot_layer, oe
    )
    print("Installed load verification wrapper on load_to_slot_layer")


def uninstall_load_verification():
    """Restore original load_to_slot_layer."""
    global _original_load_to_slot_layer, _offload_engine_ref

    if _offload_engine_ref is not None and _original_load_to_slot_layer is not None:
        _offload_engine_ref.load_to_slot_layer = _original_load_to_slot_layer
        print("Restored original load_to_slot_layer")

    _original_load_to_slot_layer = None
    _offload_engine_ref = None


# ============================================================
# Attention Hook
# ============================================================

def make_kv_pattern_pre_hook(layer_id: int):
    """
    Create a PRE-forward hook that overwrites K/V with distinctive patterns BEFORE
    they are stored to cache. This is called before attention.forward().

    register_forward_pre_hook receives (module, inputs) and can modify inputs in-place.
    """
    def hook(module, inputs):
        ctx = get_context()
        if not ctx.is_prefill:
            return

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        kvcache_manager = ctx.kvcache_manager if hasattr(ctx, 'kvcache_manager') else None

        if kvcache_manager is None:
            return

        # Only process layer 0 for cleaner output
        if layer_id != 0:
            return

        q, k, v = inputs
        k_pattern, v_pattern = get_expected_pattern(chunk_idx)

        # === Overwrite current chunk's K/V with distinctive pattern ===
        # This happens BEFORE forward(), so these values will be stored to cache
        k.fill_(k_pattern)
        v.fill_(v_pattern)

        # Only print for first few and last few chunks to reduce noise
        num_chunks = INPUT_LEN // BLOCK_SIZE
        if chunk_idx < 3 or chunk_idx >= num_chunks - 2:
            print(f"[Chunk {chunk_idx:3d}] Set K={k_pattern:.1f}, V={v_pattern:.1f}")
        elif chunk_idx == 3:
            print(f"... (chunks 3 to {num_chunks - 3} omitted) ...")

    return hook


def make_block_coverage_pre_hook(layer_id: int):
    """
    Create a PRE-forward hook to verify that all previous blocks are included
    in the cpu_block_table for chunked prefill attention.

    This catches bugs where:
    - Some blocks are missing from the computation
    - Sparse policy incorrectly filters out blocks (when not intended)
    - Block table construction has off-by-one errors
    """
    def hook(module, inputs):
        ctx = get_context()
        if not ctx.is_prefill:
            return

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        kvcache_manager = ctx.kvcache_manager if hasattr(ctx, 'kvcache_manager') else None

        if kvcache_manager is None:
            return

        # Only process layer 0 for cleaner output
        if layer_id != 0:
            return

        # Update current chunk for load verification tracking
        current_chunk_for_load[0] = chunk_idx

        # No previous blocks for chunk 0
        if chunk_idx == 0:
            return

        # Get the sequence and its block table (same logic as _chunked_prefill_attention)
        seq = ctx.chunked_seq if hasattr(ctx, 'chunked_seq') else None
        if seq is None:
            return

        # Get the CPU block table that will be used for attention
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

        # Expected blocks: 0 to chunk_idx-1 (all previous chunks)
        expected_blocks = set(range(chunk_idx))
        actual_blocks = set(cpu_block_table) if cpu_block_table else set()

        # Store for later summary
        block_coverage[chunk_idx] = {
            'expected': expected_blocks,
            'actual': actual_blocks,
        }

        # Check for missing blocks
        missing_blocks = expected_blocks - actual_blocks
        extra_blocks = actual_blocks - expected_blocks

        num_chunks = INPUT_LEN // BLOCK_SIZE
        if chunk_idx < 4 or chunk_idx >= num_chunks - 2 or missing_blocks:
            if not missing_blocks and not extra_blocks:
                print(f"  Block coverage chunk {chunk_idx:2d}: {len(actual_blocks)}/{len(expected_blocks)} blocks [OK]")
            else:
                status_parts = []
                if missing_blocks:
                    status_parts.append(f"MISSING {sorted(missing_blocks)}")
                if extra_blocks:
                    status_parts.append(f"EXTRA {sorted(extra_blocks)}")
                print(f"  Block coverage chunk {chunk_idx:2d}: {len(actual_blocks)}/{len(expected_blocks)} blocks [{', '.join(status_parts)}]")
        elif chunk_idx == 4:
            # Indicate that middle chunks are being verified silently
            print(f"  ... (verifying chunks 4-{num_chunks - 3} silently) ...")

        if missing_blocks:
            errors.append(f"Chunk {chunk_idx} missing blocks: {sorted(missing_blocks)}")

    return hook


def make_gpu_write_verification_post_hook(layer_id: int):
    """
    Create a POST-forward hook to verify the current chunk's KV was correctly
    written to the GPU ring buffer write_slot.

    This is a more reliable verification than checking load slots, because:
    1. Post-hook runs AFTER forward() writes to GPU cache
    2. write_slot mapping is deterministic: chunk_idx % num_ring_slots
    3. We injected known patterns in pre-hook, now verify they're in GPU cache
    """
    def hook(module, inputs, output):
        ctx = get_context()
        if not ctx.is_prefill:
            return

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        kvcache_manager = ctx.kvcache_manager if hasattr(ctx, 'kvcache_manager') else None

        if kvcache_manager is None:
            return

        # Only process layer 0 for cleaner output
        if layer_id != 0:
            return

        oe = kvcache_manager.offload_engine
        num_ring_slots = oe.num_ring_slots
        write_slot = chunk_idx % num_ring_slots

        # Get expected pattern for current chunk
        expected_k, expected_v = get_expected_pattern(chunk_idx)

        # Verify write_slot contains current chunk's data (GPU cache has no layer dimension)
        gpu_k = oe.k_cache_gpu[write_slot]
        gpu_v = oe.v_cache_gpu[write_slot]

        actual_k_mean = gpu_k.float().mean().item()
        actual_v_mean = gpu_v.float().mean().item()

        k_ok, _ = check_pattern(gpu_k, expected_k, f"GPU slot {write_slot}")
        v_ok, _ = check_pattern(gpu_v, expected_v, f"GPU slot {write_slot}")

        num_chunks = INPUT_LEN // BLOCK_SIZE
        # Print for first/last chunks, or if there's an error
        if True or chunk_idx >= num_chunks - 2 or not (k_ok and v_ok):
            if k_ok and v_ok:
                print(f"  GPU write_slot[{write_slot}] chunk {chunk_idx:2d}: K={expected_k:.1f}, V={expected_v:.1f} [OK]")
            else:
                print(f"  GPU write_slot[{write_slot}] chunk {chunk_idx:2d}: expected K={expected_k:.1f}/V={expected_v:.1f}, "
                      f"got K={actual_k_mean:.2f}/V={actual_v_mean:.2f} [FAIL]")
        elif chunk_idx == 4:
            print(f"  ... (GPU write verification for chunks 4-{num_chunks - 3} silently) ...")

        if not (k_ok and v_ok):
            errors.append(f"GPU write_slot {write_slot} at chunk {chunk_idx}: "
                         f"expected K={expected_k}, V={expected_v}, got K={actual_k_mean:.4f}, V={actual_v_mean:.4f}")

    return hook


def make_kv_verification_post_hook(layer_id: int):
    """
    Create a POST-forward hook to verify CPU cache contains correct patterns
    from previously offloaded blocks.
    """
    def hook(module, inputs, output):
        ctx = get_context()
        if not ctx.is_prefill:
            return

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        kvcache_manager = ctx.kvcache_manager if hasattr(ctx, 'kvcache_manager') else None

        if kvcache_manager is None:
            return

        # Only process layer 0 for cleaner output
        if layer_id != 0:
            return

        # === Verify previously offloaded blocks in CPU cache ===
        if chunk_idx >= 1:
            oe = kvcache_manager.offload_engine
            num_ok = 0
            num_fail = 0

            # Check all previously offloaded blocks
            for prev_chunk in range(chunk_idx):
                # CPU block ID = prev_chunk (in simple sequential case)
                cpu_block_id = prev_chunk

                # Get expected pattern for this block
                expected_k, expected_v = get_expected_pattern(prev_chunk)

                # Read from CPU cache (layer 0)
                cpu_k = oe.k_cache_cpu[layer_id, cpu_block_id]
                cpu_v = oe.v_cache_cpu[layer_id, cpu_block_id]

                # Verify patterns
                k_ok, k_err = check_pattern(cpu_k, expected_k, f"CPU K block {cpu_block_id}")
                v_ok, v_err = check_pattern(cpu_v, expected_v, f"CPU V block {cpu_block_id}")

                if k_ok and v_ok:
                    num_ok += 1
                else:
                    num_fail += 1
                    if k_err:
                        errors.append(k_err)
                    if v_err:
                        errors.append(v_err)

            # Only print summary for each chunk verification
            num_chunks = INPUT_LEN // BLOCK_SIZE
            if chunk_idx < 4 or chunk_idx >= num_chunks - 2 or num_fail > 0:
                status = "OK" if num_fail == 0 else f"FAIL({num_fail})"
                print(f"  CPU verify chunk {chunk_idx:2d}: {num_ok} blocks OK [{status}]")
            elif chunk_idx == 4:
                print(f"  ... (CPU cache verification for chunks 4-{num_chunks - 3} silently) ...")

    return hook


def make_post_chunk_verification_hook(layer_id: int):
    """
    Post-forward hook to verify GPU ring buffer state after attention.
    """
    def hook(module, inputs, output):
        ctx = get_context()
        if not ctx.is_prefill or layer_id != 0:
            return

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        kvcache_manager = ctx.kvcache_manager if hasattr(ctx, 'kvcache_manager') else None

        if kvcache_manager is None:
            return

        oe = kvcache_manager.offload_engine

        # After attention, the current chunk's KV should be in the GPU ring buffer
        # Ring slot = chunk_idx % num_ring_slots
        ring_slot = chunk_idx % oe.num_ring_slots

        expected_k, expected_v = get_expected_pattern(chunk_idx)

        # Check GPU ring buffer (GPU cache has no layer dimension)
        gpu_k = oe.k_cache_gpu[ring_slot]
        gpu_v = oe.v_cache_gpu[ring_slot]

        k_ok, k_err = check_pattern(gpu_k, expected_k, f"GPU K slot {ring_slot}")
        v_ok, v_err = check_pattern(gpu_v, expected_v, f"GPU V slot {ring_slot}")

        if k_ok and v_ok:
            print(f"  [OK] GPU slot {ring_slot} (chunk {chunk_idx}): K={expected_k}, V={expected_v}")
        else:
            if k_err:
                print(f"  [FAIL] {k_err}")
                errors.append(k_err)
            if v_err:
                print(f"  [FAIL] {v_err}")
                errors.append(v_err)

    return hook


def register_hooks(llm):
    """Register pre and post forward hooks."""
    hooks = []
    model = llm.model_runner.model

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn_module = decoder_layer.self_attn.attn

        # PRE-forward hook 1: Verify all previous blocks are in cpu_block_table
        coverage_hook = attn_module.register_forward_pre_hook(make_block_coverage_pre_hook(layer_idx))
        hooks.append(coverage_hook)

        # PRE-forward hook 2: Inject K/V patterns before they're stored to cache
        pattern_hook = attn_module.register_forward_pre_hook(make_kv_pattern_pre_hook(layer_idx))
        hooks.append(pattern_hook)

        # POST-forward hook 1: Verify GPU write_slot contains current chunk's data
        gpu_verify_hook = attn_module.register_forward_hook(make_gpu_write_verification_post_hook(layer_idx))
        hooks.append(gpu_verify_hook)

        # POST-forward hook 2: Verify CPU cache contains correct patterns after offload
        cpu_verify_hook = attn_module.register_forward_hook(make_kv_verification_post_hook(layer_idx))
        hooks.append(cpu_verify_hook)

    return hooks


# ============================================================
# Final Verification
# ============================================================

def verify_final_cpu_state(llm, num_chunks: int):
    """Verify all CPU blocks have correct patterns after prefill completes."""
    print("\n" + "=" * 60)
    print("Final CPU Cache Verification")
    print("=" * 60)

    kvcache_manager = llm.model_runner.kvcache_manager
    oe = kvcache_manager.offload_engine

    num_ok = 0
    num_fail = 0
    fail_details = []

    # After prefill, all chunks should be in CPU
    for chunk_idx in range(num_chunks):
        cpu_block_id = chunk_idx
        expected_k, expected_v = get_expected_pattern(chunk_idx)

        # Check layer 0
        cpu_k = oe.k_cache_cpu[0, cpu_block_id]
        cpu_v = oe.v_cache_cpu[0, cpu_block_id]

        k_ok, k_err = check_pattern(cpu_k, expected_k, f"Final CPU K block {cpu_block_id}")
        v_ok, v_err = check_pattern(cpu_v, expected_v, f"Final CPU V block {cpu_block_id}")

        if k_ok and v_ok:
            num_ok += 1
            # Only print first few and last few
            if chunk_idx < 3 or chunk_idx >= num_chunks - 2:
                actual_k_mean = cpu_k.float().mean().item()
                actual_v_mean = cpu_v.float().mean().item()
                print(f"  Block {cpu_block_id:3d}: K={expected_k:.1f} ({actual_k_mean:.4f}), "
                      f"V={expected_v:.1f} ({actual_v_mean:.4f}) [OK]")
            elif chunk_idx == 3:
                print(f"  ... (blocks 3 to {num_chunks - 3} verified OK) ...")
        else:
            num_fail += 1
            actual_k_mean = cpu_k.float().mean().item()
            actual_v_mean = cpu_v.float().mean().item()
            print(f"  Block {cpu_block_id:3d}: K={expected_k:.1f} ({actual_k_mean:.4f}), "
                  f"V={expected_v:.1f} ({actual_v_mean:.4f}) [FAIL]")
            if k_err:
                errors.append(k_err)
            if v_err:
                errors.append(v_err)

    print(f"\nTotal: {num_ok} OK, {num_fail} FAIL out of {num_chunks} blocks")


def verify_block_coverage_summary(num_chunks: int):
    """Verify that all chunks had complete block coverage during prefill."""
    print("\n" + "=" * 60)
    print("Block Coverage Verification Summary")
    print("=" * 60)

    num_ok = 0
    num_fail = 0
    total_blocks_expected = 0
    total_blocks_computed = 0

    for chunk_idx in range(1, num_chunks):  # Start from 1 (chunk 0 has no previous)
        if chunk_idx not in block_coverage:
            print(f"  Chunk {chunk_idx}: NO COVERAGE DATA [FAIL]")
            errors.append(f"Chunk {chunk_idx} has no block coverage data")
            num_fail += 1
            continue

        coverage = block_coverage[chunk_idx]
        expected = coverage['expected']
        actual = coverage['actual']
        missing = expected - actual

        total_blocks_expected += len(expected)
        total_blocks_computed += len(actual)

        if not missing:
            num_ok += 1
        else:
            num_fail += 1

    # Print summary
    if num_fail == 0:
        print(f"  All {num_ok} chunks had complete block coverage [OK]")
        print(f"  Total blocks computed: {total_blocks_computed} (expected: {total_blocks_expected})")
    else:
        print(f"  {num_ok} chunks OK, {num_fail} chunks with missing blocks [FAIL]")
        print(f"  Total blocks computed: {total_blocks_computed} (expected: {total_blocks_expected})")

    # Verify the total is correct: sum of 0+1+2+...+(n-1) = n*(n-1)/2
    expected_total = num_chunks * (num_chunks - 1) // 2
    if total_blocks_expected == expected_total:
        print(f"  Expected total blocks matches formula: {expected_total} [OK]")
    else:
        print(f"  Expected total mismatch: got {total_blocks_expected}, formula gives {expected_total} [FAIL]")
        errors.append(f"Block coverage total mismatch")


def verify_load_operations_summary(num_chunks: int):
    """Verify all H2D load operations transferred correct data."""
    print("\n" + "=" * 60)
    print("H2D Load Operations Verification Summary")
    print("=" * 60)

    if not load_operations:
        print("  WARNING: No load operations recorded!")
        print("  (This may indicate load verification was not installed)")
        return

    num_ok = 0
    num_fail = 0
    loads_per_chunk = {}

    for op in load_operations:
        chunk_idx = op['chunk_idx']
        if chunk_idx not in loads_per_chunk:
            loads_per_chunk[chunk_idx] = []
        loads_per_chunk[chunk_idx].append(op)

        if op['k_ok'] and op['v_ok']:
            num_ok += 1
        else:
            num_fail += 1

    # Print per-chunk summary for first/last chunks
    for chunk_idx in sorted(loads_per_chunk.keys()):
        ops = loads_per_chunk[chunk_idx]
        chunk_ok = sum(1 for op in ops if op['k_ok'] and op['v_ok'])
        chunk_fail = len(ops) - chunk_ok

        if chunk_idx < 4 or chunk_idx >= num_chunks - 2 or chunk_fail > 0:
            # Show loaded block IDs in order
            block_ids = [op['cpu_block_id'] for op in ops]
            if chunk_fail == 0:
                print(f"  Chunk {chunk_idx:2d}: loaded {len(ops)} blocks {block_ids} [OK]")
            else:
                print(f"  Chunk {chunk_idx:2d}: loaded {len(ops)} blocks, {chunk_fail} FAILED [FAIL]")
                for op in ops:
                    if not (op['k_ok'] and op['v_ok']):
                        print(f"    CPU block {op['cpu_block_id']} -> slot {op['slot_idx']}: "
                              f"expected K={op['expected_k']:.1f}/V={op['expected_v']:.1f}, "
                              f"got K={op['actual_k']:.4f}/V={op['actual_v']:.4f}")
        elif chunk_idx == 4:
            print(f"  ... (chunks 4-{num_chunks - 3} load verification running silently) ...")

    # Print overall summary
    print(f"\n  Total load operations: {len(load_operations)}")
    print(f"  Successful: {num_ok}, Failed: {num_fail}")

    if num_fail == 0:
        print(f"  All H2D transfers verified correct [OK]")
    else:
        print(f"  {num_fail} H2D transfers had incorrect data [FAIL]")


# ============================================================
# Main Test Script
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Test: CPU Offload Correctness with Distinctive KV Patterns")
    print("=" * 60)
    print(f"Input: {INPUT_LEN} tokens, {INPUT_LEN // BLOCK_SIZE} chunks")
    print(f"GPU blocks: {NUM_GPU_BLOCKS}, Block size: {BLOCK_SIZE}")
    print(f"Pattern: K=chunk_idx+1, V=-(chunk_idx+1)")
    print()

    # 1. Initialize LLM
    print("Initializing LLM...")
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        enable_cpu_offload=True,
        kvcache_block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        dtype="float16",
    )

    # 2. Register hooks
    hooks = register_hooks(llm)
    print(f"Registered {len(hooks)} hooks")

    # 3. Install load verification (instrument load_to_slot_layer)
    install_load_verification(llm)

    # 4. Generate prompt
    seed(42)
    prompt_token_ids = [[randint(0, 10000) for _ in range(INPUT_LEN)]]
    num_chunks = INPUT_LEN // BLOCK_SIZE

    # 5. Run prefill
    print("\n" + "=" * 60)
    print("Running Prefill with KV Pattern Injection...")
    print("=" * 60)

    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

    # 6. Remove hooks and uninstall load verification
    for hook in hooks:
        hook.remove()
    uninstall_load_verification()

    # 7. Final verification
    verify_final_cpu_state(llm, num_chunks)

    # 8. Block coverage summary
    verify_block_coverage_summary(num_chunks)

    # 9. H2D load operations summary
    verify_load_operations_summary(num_chunks)

    # 10. Report results
    print("\n" + "=" * 60)
    if errors:
        print(f"test_offload_correctness: FAILED ({len(errors)} errors)")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        exit(1)
    else:
        print("test_offload_correctness: PASSED")
    print("=" * 60)
