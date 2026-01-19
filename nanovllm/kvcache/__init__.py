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
    1. If enable_cpu_offload=False: use GPUOnlyManager
    2. If enable_cpu_offload=True but all blocks fit in GPU: use GPUOnlyManager
    3. If enable_cpu_offload=True and need CPU blocks: use HybridKVCacheManager

    Args:
        config: Model configuration with offload settings

    Returns:
        KVCacheManager instance
    """
    if not getattr(config, 'enable_cpu_offload', False):
        # Default: pure GPU mode
        return GPUOnlyManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
        )

    # CPU offload is enabled
    num_gpu_blocks = config.num_gpu_kvcache_blocks
    num_cpu_blocks = config.num_cpu_kvcache_blocks

    if num_cpu_blocks <= 0:
        # All blocks fit in GPU, use pure GPU mode
        return GPUOnlyManager(
            num_blocks=num_gpu_blocks,
            block_size=config.kvcache_block_size,
        )

    # Need CPU offload: use hybrid manager
    from nanovllm.kvcache.hybrid_manager import HybridKVCacheManager
    from nanovllm.kvcache.policies import get_policy
    from nanovllm.kvcache.sparse import create_sparse_policy
    from nanovllm.config import SparsePolicyType

    eviction_policy = get_policy(getattr(config, 'offload_policy', 'lru'))

    # Create sparse policy from config enum
    # Quest is decode-only: prefill returns all blocks (query=None), decode does Top-K
    sparse_policy_type = getattr(config, 'sparse_policy', SparsePolicyType.FULL)

    # Build policy kwargs based on policy type
    policy_kwargs = {}
    if sparse_policy_type == SparsePolicyType.QUEST:
        policy_kwargs = {
            'topk_blocks': getattr(config, 'sparse_topk_blocks', 8),
            'threshold_blocks': getattr(config, 'sparse_threshold_blocks', 4),
        }
    elif sparse_policy_type == SparsePolicyType.XATTN_BSA:
        policy_kwargs = {
            'block_size': getattr(config, 'sparse_block_size', 128),
            'samples_per_chunk': getattr(config, 'sparse_samples_per_chunk', 128),
            'threshold': getattr(config, 'sparse_threshold', 0.9),
            'use_triton': getattr(config, 'sparse_use_triton', True),
            'stride': getattr(config, 'sparse_stride', 8),
        }

    sparse_policy = create_sparse_policy(sparse_policy_type, **policy_kwargs)

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
