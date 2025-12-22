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

    policy = get_policy(getattr(config, 'offload_policy', 'lru'))
    num_prefetch_blocks = getattr(config, 'num_prefetch_blocks', 2)

    return HybridKVCacheManager(
        num_gpu_slots=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=config.kvcache_block_size,
        policy=policy,
        num_prefetch_blocks=num_prefetch_blocks,
    )


__all__ = [
    "KVCacheManager",
    "GPUOnlyManager",
    "create_kvcache_manager",
]
