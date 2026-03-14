"""
KV Cache management module.

This module provides pluggable KV cache management strategies:
- GPUOnlyManager: Pure GPU (default, current nano-vllm behavior)
- HybridKVCacheManager: CPU-primary storage with GPU ring buffer for computation

Usage:
    from nanovllm.kvcache import create_kvcache_manager

    manager = create_kvcache_manager(config)
"""

from typing import TYPE_CHECKING

from nanovllm.kvcache.base_manager import KVCacheManager
from nanovllm.kvcache.gpu_manager import GPUOnlyManager

if TYPE_CHECKING:
    from nanovllm.config import Config


def create_kvcache_manager(config: "Config") -> KVCacheManager:
    """
    Factory function to create the appropriate KV cache manager.

    Decision logic:
    1. If enable_cpu_offload=False: use GPUOnlyManager (optionally with sparse policy)
    2. If enable_cpu_offload=True but all blocks fit in GPU: use GPUOnlyManager
    3. If enable_cpu_offload=True and need CPU blocks: use HybridKVCacheManager

    Args:
        config: Model configuration with offload settings

    Returns:
        KVCacheManager instance
    """
    if not getattr(config, "enable_cpu_offload", False):
        # Default: pure GPU mode
        # Check if sparse policy is requested for GPU-only mode
        from nanovllm.config import SparsePolicyType

        sparse_policy_type = getattr(config, "sparse_policy", None)
        # Handle None case - use FULL as default
        if sparse_policy_type is None:
            sparse_policy_type = SparsePolicyType.FULL

        sparse_policy = None
        if sparse_policy_type != SparsePolicyType.FULL:
            # Create sparse policy for GPU-only mode
            from nanovllm.kvcache.sparse import create_sparse_policy

            policy_kwargs = {}
            if sparse_policy_type == SparsePolicyType.QUEST:
                policy_kwargs = {
                    "topk_blocks": getattr(config, "sparse_topk_blocks", 8),
                    "threshold_blocks": getattr(config, "sparse_threshold_blocks", 4),
                }
            elif sparse_policy_type == SparsePolicyType.XATTN_BSA:
                policy_kwargs = {
                    "block_size": getattr(config, "sparse_block_size", 128),
                    "samples_per_chunk": getattr(
                        config, "sparse_samples_per_chunk", 128
                    ),
                    "threshold": getattr(config, "sparse_threshold", 0.9),
                    "use_triton": getattr(config, "sparse_use_triton", True),
                    "stride": getattr(config, "sparse_stride", 8),
                    "chunk_size": getattr(config, "sparse_chunk_size", 16384),
                }
            elif sparse_policy_type == SparsePolicyType.COMPASS:
                policy_kwargs = {}  # COMPASS has no extra parameters
            elif sparse_policy_type == SparsePolicyType.BLASST:
                policy_kwargs = {
                    "a": getattr(config, "blasst_a", 16384),
                    "fixed_lambda": getattr(config, "blasst_fixed_lambda", None),
                    "granularity": getattr(config, "blasst_granularity", 128),
                }

            sparse_policy = create_sparse_policy(sparse_policy_type, **policy_kwargs)
        else:
            # FULL policy for GPU-only mode - always create for consistent API
            from nanovllm.kvcache.sparse import FullAttentionPolicy

            sparse_policy = FullAttentionPolicy()

        return GPUOnlyManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            sparse_policy=sparse_policy,
        )

    # CPU offload is enabled
    num_gpu_blocks = config.num_gpu_kvcache_blocks
    num_cpu_blocks = config.num_cpu_kvcache_blocks

    # Create sparse policy first (needed for both GPU-only and hybrid paths)
    from nanovllm.kvcache.sparse import create_sparse_policy
    from nanovllm.config import SparsePolicyType

    # Create sparse policy from config enum
    # Quest is decode-only: prefill returns all blocks (query=None), decode does Top-K
    sparse_policy_type = getattr(config, "sparse_policy", SparsePolicyType.FULL)

    # Build policy kwargs based on policy type
    policy_kwargs = {}
    if sparse_policy_type == SparsePolicyType.QUEST:
        policy_kwargs = {
            "topk_blocks": getattr(config, "sparse_topk_blocks", 8),
            "threshold_blocks": getattr(config, "sparse_threshold_blocks", 4),
        }
    elif sparse_policy_type == SparsePolicyType.XATTN_BSA:
        policy_kwargs = {
            "block_size": getattr(config, "sparse_block_size", 128),
            "samples_per_chunk": getattr(config, "sparse_samples_per_chunk", 128),
            "threshold": getattr(config, "sparse_threshold", 0.9),
            "use_triton": getattr(config, "sparse_use_triton", True),
            "stride": getattr(config, "sparse_stride", 8),
            "chunk_size": getattr(config, "sparse_chunk_size", 16384),
        }
    elif sparse_policy_type == SparsePolicyType.COMPASS:
        policy_kwargs = {
            "lambda_threshold": getattr(config, "lambda_threshold", 0.001),
        }
    elif sparse_policy_type == SparsePolicyType.BLASST:
        policy_kwargs = {
            "a": getattr(config, "blasst_a", 16384),
            "fixed_lambda": getattr(config, "blasst_fixed_lambda", None),
            "granularity": getattr(config, "blasst_granularity", 128),
        }

    sparse_policy = create_sparse_policy(sparse_policy_type, **policy_kwargs)

    if num_cpu_blocks <= 0:
        # All blocks fit in GPU, use pure GPU mode
        # But still need sparse_policy for consistent API
        return GPUOnlyManager(
            num_blocks=num_gpu_blocks,
            block_size=config.kvcache_block_size,
            sparse_policy=sparse_policy,
        )

    # Need CPU offload: use hybrid manager
    from nanovllm.kvcache.hybrid_manager import HybridKVCacheManager
    from nanovllm.kvcache.policies import get_policy

    eviction_policy = get_policy(getattr(config, "offload_policy", "lru"))

    return HybridKVCacheManager(
        num_gpu_slots=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=config.kvcache_block_size,
        policy=eviction_policy,
        sparse_policy=sparse_policy,
    )


__all__ = [
    "KVCacheManager",
    "GPUOnlyManager",
    "create_kvcache_manager",
]
