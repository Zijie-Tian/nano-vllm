"""
Sparse Attention Policy module.

Provides pluggable policies for selecting which KV blocks to load
during chunked attention with CPU offload.

Usage:
    from nanovllm.kvcache.sparse import create_sparse_policy, SparsePolicyType

    # Create policy using factory function
    policy = create_sparse_policy(SparsePolicyType.QUEST, topk_blocks=8)

    # Or create custom policy
    class MyPolicy(SparsePolicy):
        supports_prefill = True
        supports_decode = True

        def select_blocks(self, available_blocks, ctx):
            return available_blocks[:5]  # Just first 5 blocks
"""

from nanovllm.config import SparsePolicyType
from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.kvcache.sparse.full_policy import FullAttentionPolicy
from nanovllm.kvcache.sparse.triattention import TriAttentionPolicy
from nanovllm.kvcache.sparse.quest import QuestPolicy, QuestConfig, BlockMetadataManager
from nanovllm.kvcache.sparse.xattn_bsa import XAttentionBSAPolicy
from nanovllm.kvcache.sparse.compass import COMPASSPolicy
from nanovllm.kvcache.sparse.blasst import BLASSTPolicy


def create_sparse_policy(policy_type: SparsePolicyType, **kwargs) -> SparsePolicy:
    """
    Create a sparse policy instance from an enum type.

    The returned policy is not yet initialized. Call policy.initialize()
    or let the framework call it during KV cache allocation.

    Args:
        policy_type: SparsePolicyType enum value
        **kwargs: Policy-specific configuration options

    Returns:
        SparsePolicy instance (not initialized)

    Example:
        policy = create_sparse_policy(SparsePolicyType.QUEST, topk_blocks=4)
        policy.initialize(num_layers=28, num_kv_heads=8, ...)
    """
    if policy_type == SparsePolicyType.FULL:
        return FullAttentionPolicy()

    elif policy_type == SparsePolicyType.TRIATTENTION:
        return TriAttentionPolicy(**kwargs)

    elif policy_type == SparsePolicyType.QUEST:
        config = QuestConfig(
            topk_blocks=kwargs.get("topk_blocks", 8),
            threshold_blocks=kwargs.get("threshold_blocks", 4),
            include_sink_blocks=kwargs.get("include_sink_blocks", 0),
            include_recent_blocks=kwargs.get("include_recent_blocks", 0),
        )
        return QuestPolicy(config)

    elif policy_type == SparsePolicyType.XATTN_BSA:
        return XAttentionBSAPolicy(
            block_size=kwargs.get("block_size", 128),
            samples_per_chunk=kwargs.get("samples_per_chunk", 128),
            threshold=kwargs.get("threshold", 0.9),
            stride=kwargs.get("stride", 8),
            chunk_size=kwargs.get("chunk_size", 16384),
            use_triton=kwargs.get("use_triton", True),
        )

    elif policy_type == SparsePolicyType.COMPASS:
        return COMPASSPolicy(
            lambda_threshold=kwargs.get("lambda_threshold", 0.001),
            top_p=kwargs.get("top_p", 0.9),
            theta=kwargs.get("theta", 0.6),
        )

    elif policy_type == SparsePolicyType.BLASST:
        return BLASSTPolicy(
            a=kwargs.get("a", 16384),
            fixed_lambda=kwargs.get("fixed_lambda", 0.5),
            granularity=kwargs.get("granularity", 128),
        )

    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


__all__ = [
    "SparsePolicy",
    "PolicyContext",
    "SparsePolicyType",
    "FullAttentionPolicy",
    "TriAttentionPolicy",
    "QuestPolicy",
    "QuestConfig",
    "BlockMetadataManager",
    "XAttentionBSAPolicy",
    "COMPASSPolicy",
    "BLASSTPolicy",
    "create_sparse_policy",
]
