"""
Test for QuestPolicy block selection with GQA (Grouped Query Attention).

Demonstrates the key limitation: scores are AVERAGED across heads,
so blocks strongly needed by one head but not others may be dropped.

This is the expected Quest behavior - not a bug.
"""

import torch
from nanovllm.kvcache.sparse import (
    create_sparse_policy,
    SparsePolicyType,
    PolicyContext,
)

# ============================================================
# Test: Per-Head Score Averaging in GQA
# ============================================================

# Determine device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running test on device: {device}")

# Setup: 2 KV heads, 4 query heads (GQA group_size=2)
# topk=2 to make selection competitive

quest = create_sparse_policy(SparsePolicyType.QUEST, topk_blocks=2, threshold_blocks=0)
quest.initialize(
    num_layers=1,
    num_kv_heads=2,
    head_dim=4,
    num_cpu_blocks=6,
    dtype=torch.float32,
    device=device,  # Metadata stored on GPU
)

metadata = quest.metadata

def set_key(block_id, head_id, values):
    """Set both key_min and key_max to same values for deterministic scoring."""
    # Values need to be on the same device as metadata
    tensor = torch.tensor(values, device=device)
    metadata.key_min[block_id, 0, head_id, :] = tensor
    metadata.key_max[block_id, 0, head_id, :] = tensor

# ============================================================
# Design: Different heads want different blocks
# ============================================================
#
# Query = [1,1,1,1] for all heads, so score = sum(key values)
#
# Block | Head 0 | Head 1 | Average | Result
# ------|--------|--------|---------|--------
#   0   |   +4   |   -4   |    0    | Head0 wants, Head1 doesn't → DROPPED
#   1   |   -4   |   +4   |    0    | Head1 wants, Head0 doesn't → DROPPED
#   2   |   +4   |   +4   |   +4    | Both want → SELECTED (rank 1)
#   3   |   +3   |   +3   |   +3    | Both want → SELECTED (rank 2)
#   4   |   +4   |    0   |   +2    | Head0 strongly wants, Head1 neutral → rank 3
#   5   |    0   |   +4   |   +2    | Head1 strongly wants, Head0 neutral → rank 3

# Block 0: Head 0 strongly wants, Head 1 strongly rejects
set_key(0, 0, [1.0, 1.0, 1.0, 1.0])      # head0: +4
set_key(0, 1, [-1.0, -1.0, -1.0, -1.0])  # head1: -4

# Block 1: Head 1 strongly wants, Head 0 strongly rejects
set_key(1, 0, [-1.0, -1.0, -1.0, -1.0])  # head0: -4
set_key(1, 1, [1.0, 1.0, 1.0, 1.0])      # head1: +4

# Block 2: Both heads want equally (highest average)
set_key(2, 0, [1.0, 1.0, 1.0, 1.0])      # head0: +4
set_key(2, 1, [1.0, 1.0, 1.0, 1.0])      # head1: +4

# Block 3: Both heads want moderately
set_key(3, 0, [0.75, 0.75, 0.75, 0.75])  # head0: +3
set_key(3, 1, [0.75, 0.75, 0.75, 0.75])  # head1: +3

# Block 4: Head 0 strongly wants, Head 1 neutral
set_key(4, 0, [1.0, 1.0, 1.0, 1.0])      # head0: +4
set_key(4, 1, [0.0, 0.0, 0.0, 0.0])      # head1: 0

# Block 5: Head 1 strongly wants, Head 0 neutral
set_key(5, 0, [0.0, 0.0, 0.0, 0.0])      # head0: 0
set_key(5, 1, [1.0, 1.0, 1.0, 1.0])      # head1: +4

# ============================================================
# Run selection
# ============================================================

# Query on same device as metadata
query = torch.ones(1, 4, 4, device=device)  # GQA: 4 query heads → 2 KV heads

ctx = PolicyContext(
    query_chunk_idx=0,
    num_query_chunks=1,
    layer_id=0,
    query=query,
    is_prefill=False,
    block_size=1024,
    total_kv_len=6144,
)

available = list(range(6))
selected = quest.select_blocks(available, ctx)

# ============================================================
# Verify: Averaging behavior
# ============================================================

# topk=2, so only blocks 2 (+4 avg) and 3 (+3 avg) should be selected
assert len(selected) == 2, f"Expected 2 blocks, got {len(selected)}"
assert selected == [2, 3], f"Expected [2, 3], got {selected}"

# Key insight: blocks 0 and 1 have score +4 for ONE head,
# but they cancel out due to averaging with the other head's -4
assert 0 not in selected, "Block 0 should NOT be selected (head scores cancel out)"
assert 1 not in selected, "Block 1 should NOT be selected (head scores cancel out)"

# Blocks 4 and 5 have +4 for one head, 0 for other → avg=+2
# But +2 < +3 (block 3), so they don't make the top-2
assert 4 not in selected, "Block 4 avg=+2 < block 3 avg=+3"
assert 5 not in selected, "Block 5 avg=+2 < block 3 avg=+3"

print("✓ Block 2 selected: both heads want it (+4, +4) → avg=+4")
print("✓ Block 3 selected: both heads want it (+3, +3) → avg=+3")
print("✓ Block 0 NOT selected: head0=+4, head1=-4 → avg=0 (cancel out)")
print("✓ Block 1 NOT selected: head0=-4, head1=+4 → avg=0 (cancel out)")
print("✓ Block 4 NOT selected: head0=+4, head1=0 → avg=+2 (lower rank)")
print("✓ Block 5 NOT selected: head0=0, head1=+4 → avg=+2 (lower rank)")

# Verify metadata is on correct device
assert metadata.key_min.device.type == device.type, f"key_min on wrong device: {metadata.key_min.device}"
assert metadata.key_max.device.type == device.type, f"key_max on wrong device: {metadata.key_max.device}"
print(f"✓ Metadata stored on {device.type.upper()}")

print("\ntest_quest_policy: PASSED")
