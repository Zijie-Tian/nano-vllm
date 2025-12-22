"""
Test sparse attention policies.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python tests/test_sparse_policy.py [policy_name]

Policy names: full, vertical_slash, streaming_llm, quest
"""

import sys
import os

os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

import torch
from typing import List

# Test the sparse policy implementations
from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.kvcache.sparse.full_policy import FullAttentionPolicy
from nanovllm.kvcache.sparse.vertical_slash import VerticalSlashPolicy, VerticalSlashConfig
from nanovllm.kvcache.sparse.streaming_llm import StreamingLLMPolicy, StreamingLLMConfig
from nanovllm.kvcache.sparse.quest import QuestPolicy, QuestConfig, BlockMetadataManager


def test_full_attention_policy():
    """Test FullAttentionPolicy returns all blocks."""
    print("\n=== Testing FullAttentionPolicy ===")
    policy = FullAttentionPolicy()

    available_blocks = list(range(10))
    ctx = PolicyContext(
        query_chunk_idx=5,
        num_query_chunks=10,
        layer_id=0,
        query=None,
        is_prefill=True,
    )

    selected = policy.select_blocks(available_blocks, ctx)
    assert selected == available_blocks, f"Expected all blocks, got {selected}"
    print(f"  Prefill: input={available_blocks}, selected={selected} [PASS]")

    # Test decode
    ctx.is_prefill = False
    selected = policy.select_blocks(available_blocks, ctx)
    assert selected == available_blocks, f"Expected all blocks, got {selected}"
    print(f"  Decode:  input={available_blocks}, selected={selected} [PASS]")


def test_vertical_slash_policy():
    """Test VerticalSlashPolicy selects sink + local window."""
    print("\n=== Testing VerticalSlashPolicy ===")
    config = VerticalSlashConfig(
        num_sink_blocks=2,
        local_window_blocks=3,
        threshold_blocks=4,
    )
    policy = VerticalSlashPolicy(config)

    # Test with 10 blocks, chunk 7 (should select sink[0,1] + local[4,5,6])
    available_blocks = list(range(10))
    ctx = PolicyContext(
        query_chunk_idx=7,
        num_query_chunks=10,
        layer_id=0,
        query=None,
        is_prefill=True,
    )

    selected = policy.select_blocks(available_blocks, ctx)
    expected = [0, 1, 4, 5, 6]  # sink + local window before chunk 7
    assert selected == expected, f"Expected {expected}, got {selected}"
    print(f"  Prefill chunk 7: input={available_blocks}, selected={selected} [PASS]")

    # Test with small number of blocks (below threshold)
    available_blocks = [0, 1, 2]
    selected = policy.select_blocks(available_blocks, ctx)
    assert selected == [0, 1, 2], f"Expected all blocks for small input, got {selected}"
    print(f"  Below threshold: input={[0,1,2]}, selected={selected} [PASS]")

    # Test decode (local window is last M blocks)
    available_blocks = list(range(10))
    ctx.is_prefill = False
    selected = policy.select_blocks(available_blocks, ctx)
    expected = [0, 1, 7, 8, 9]  # sink + last 3 blocks
    assert selected == expected, f"Expected {expected}, got {selected}"
    print(f"  Decode: input={available_blocks}, selected={selected} [PASS]")


def test_streaming_llm_policy():
    """Test StreamingLLMPolicy selects sink + recent only."""
    print("\n=== Testing StreamingLLMPolicy ===")
    config = StreamingLLMConfig(
        num_sink_blocks=1,
        num_recent_blocks=2,
    )
    policy = StreamingLLMPolicy(config)

    available_blocks = list(range(10))
    ctx = PolicyContext(
        query_chunk_idx=0,
        num_query_chunks=1,
        layer_id=0,
        query=None,
        is_prefill=False,
    )

    selected = policy.select_blocks(available_blocks, ctx)
    expected = [0, 8, 9]  # sink[0] + recent[8,9]
    assert selected == expected, f"Expected {expected}, got {selected}"
    print(f"  10 blocks: selected={selected} [PASS]")

    # Test with 3 blocks (all fit in sink+recent)
    available_blocks = [0, 1, 2]
    selected = policy.select_blocks(available_blocks, ctx)
    assert selected == [0, 1, 2], f"Expected all blocks, got {selected}"
    print(f"  3 blocks:  selected={selected} [PASS]")


def test_quest_policy():
    """Test QuestPolicy with mock metadata."""
    print("\n=== Testing QuestPolicy ===")

    # Create metadata manager
    num_blocks = 10
    num_layers = 2
    num_kv_heads = 4
    head_dim = 64
    dtype = torch.float32

    metadata = BlockMetadataManager(
        num_blocks=num_blocks,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # Simulate offloading blocks with different key patterns
    # Blocks 0, 5, 9 will have high scores (keys aligned with query)
    for block_id in range(num_blocks):
        for layer_id in range(num_layers):
            k_cache = torch.randn(100, num_kv_heads, head_dim)  # 100 tokens per block
            if block_id in [0, 5, 9]:
                # Make these blocks have keys that score high
                k_cache = k_cache.abs()  # All positive
            else:
                k_cache = -k_cache.abs()  # All negative
            metadata.update_metadata(block_id, layer_id, k_cache, 100)

    config = QuestConfig(
        topk_blocks=4,
        threshold_blocks=3,
    )
    policy = QuestPolicy(config, metadata)

    available_blocks = list(range(10))

    # Create query that scores high with positive keys
    query = torch.ones(1, num_kv_heads, head_dim, device='cuda')

    ctx = PolicyContext(
        query_chunk_idx=0,
        num_query_chunks=1,
        layer_id=0,
        query=query,
        is_prefill=False,
    )

    selected = policy.select_blocks(available_blocks, ctx)
    print(f"  Top-4 selection: input={available_blocks}, selected={selected}")

    # High-scoring blocks [0, 5, 9] should be in selection
    for expected_block in [0, 5, 9]:
        assert expected_block in selected, f"Expected block {expected_block} in selection"
    print(f"  High-score blocks [0, 5, 9] in selection [PASS]")

    # Test below threshold (should return all)
    available_blocks = [0, 1, 2]
    selected = policy.select_blocks(available_blocks, ctx)
    assert selected == [0, 1, 2], f"Expected all blocks below threshold, got {selected}"
    print(f"  Below threshold: selected={selected} [PASS]")

    # Test without query (should return all)
    ctx.query = None
    available_blocks = list(range(10))
    selected = policy.select_blocks(available_blocks, ctx)
    assert selected == available_blocks, f"Expected all blocks without query, got {selected}"
    print(f"  No query: selected all [PASS]")


def test_custom_policy():
    """Test creating a custom policy."""
    print("\n=== Testing Custom Policy ===")

    class EveryOtherPolicy(SparsePolicy):
        """Select every other block."""

        def select_blocks(self, available_blocks: List[int], ctx: PolicyContext) -> List[int]:
            return [available_blocks[i] for i in range(0, len(available_blocks), 2)]

    policy = EveryOtherPolicy()
    available_blocks = list(range(10))
    ctx = PolicyContext(
        query_chunk_idx=0,
        num_query_chunks=1,
        layer_id=0,
        query=None,
        is_prefill=True,
    )

    selected = policy.select_blocks(available_blocks, ctx)
    expected = [0, 2, 4, 6, 8]
    assert selected == expected, f"Expected {expected}, got {selected}"
    print(f"  Every other: input={available_blocks}, selected={selected} [PASS]")


def run_all_tests():
    """Run all policy tests."""
    print("Running Sparse Policy Tests...")

    test_full_attention_policy()
    test_vertical_slash_policy()
    test_streaming_llm_policy()
    test_quest_policy()
    test_custom_policy()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        policy_name = sys.argv[1].lower()
        if policy_name == "full":
            test_full_attention_policy()
        elif policy_name == "vertical_slash":
            test_vertical_slash_policy()
        elif policy_name == "streaming_llm":
            test_streaming_llm_policy()
        elif policy_name == "quest":
            test_quest_policy()
        elif policy_name == "custom":
            test_custom_policy()
        else:
            print(f"Unknown policy: {policy_name}")
            print("Available: full, vertical_slash, streaming_llm, quest, custom")
            sys.exit(1)
    else:
        run_all_tests()
