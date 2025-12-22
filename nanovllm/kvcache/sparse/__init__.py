"""
Sparse Attention Policy module.

Provides pluggable policies for selecting which KV blocks to load
during chunked attention with CPU offload.

Usage:
    from nanovllm.kvcache.sparse import SparsePolicy, PolicyContext
    from nanovllm.kvcache.sparse import VerticalSlashPolicy, QuestPolicy

    # Use built-in policy
    policy = VerticalSlashPolicy(VerticalSlashConfig())

    # Or create custom policy
    class MyPolicy(SparsePolicy):
        def select_blocks(self, available_blocks, ctx):
            return available_blocks[:5]  # Just first 5 blocks
"""

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.kvcache.sparse.full_policy import FullAttentionPolicy
from nanovllm.kvcache.sparse.vertical_slash import VerticalSlashPolicy, VerticalSlashConfig
from nanovllm.kvcache.sparse.quest import QuestPolicy, QuestConfig, BlockMetadataManager
from nanovllm.kvcache.sparse.streaming_llm import StreamingLLMPolicy, StreamingLLMConfig
from nanovllm.kvcache.sparse.hybrid import HybridPolicy

# Built-in policy registry
BUILTIN_SPARSE_POLICIES = {
    "full": FullAttentionPolicy,
    "vertical_slash": VerticalSlashPolicy,
    "streaming_llm": StreamingLLMPolicy,
}


def get_sparse_policy(policy_name: str, **kwargs) -> SparsePolicy:
    """
    Get a sparse attention policy instance by name.

    Args:
        policy_name: Policy name ("full", "vertical_slash", "streaming_llm", "quest")
        **kwargs: Policy-specific configuration

    Returns:
        SparsePolicy instance
    """
    policy_name = policy_name.lower()

    if policy_name == "full":
        return FullAttentionPolicy()
    elif policy_name == "vertical_slash":
        config = VerticalSlashConfig(
            num_sink_blocks=kwargs.get("num_sink_blocks", 1),
            local_window_blocks=kwargs.get("local_window_blocks", 2),
            threshold_blocks=kwargs.get("threshold_blocks", 4),
        )
        return VerticalSlashPolicy(config)
    elif policy_name == "streaming_llm":
        config = StreamingLLMConfig(
            num_sink_blocks=kwargs.get("num_sink_blocks", 1),
            num_recent_blocks=kwargs.get("num_recent_blocks", 3),
        )
        return StreamingLLMPolicy(config)
    elif policy_name == "quest":
        # Quest requires metadata_manager to be passed separately
        raise ValueError(
            "Quest policy requires BlockMetadataManager. "
            "Use QuestPolicy(config, metadata_manager) directly."
        )
    else:
        raise ValueError(
            f"Unknown sparse policy '{policy_name}'. "
            f"Available policies: {list(BUILTIN_SPARSE_POLICIES.keys())}"
        )


__all__ = [
    "SparsePolicy",
    "PolicyContext",
    "FullAttentionPolicy",
    "VerticalSlashPolicy",
    "VerticalSlashConfig",
    "QuestPolicy",
    "QuestConfig",
    "BlockMetadataManager",
    "StreamingLLMPolicy",
    "StreamingLLMConfig",
    "HybridPolicy",
    "get_sparse_policy",
    "BUILTIN_SPARSE_POLICIES",
]
