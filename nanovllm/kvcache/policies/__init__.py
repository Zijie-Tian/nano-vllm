"""
Eviction policy plugins for KV cache offloading.

Users can create custom policies by subclassing EvictionPolicy
and specifying the full class path in config.offload_policy.
"""

from nanovllm.kvcache.policies.base_policy import EvictionPolicy
from nanovllm.kvcache.policies.lru_policy import LRUPolicy
from nanovllm.kvcache.policies.fifo_policy import FIFOPolicy

# Built-in policy registry
BUILTIN_POLICIES = {
    "lru": LRUPolicy,
    "fifo": FIFOPolicy,
}


def get_policy(policy_name: str) -> EvictionPolicy:
    """
    Get an eviction policy instance by name or class path.

    Args:
        policy_name: Either a built-in name ("lru", "fifo") or
                    a full class path ("mymodule.MyPolicy")

    Returns:
        EvictionPolicy instance
    """
    # Check built-in policies first
    if policy_name.lower() in BUILTIN_POLICIES:
        return BUILTIN_POLICIES[policy_name.lower()]()

    # Try to import custom policy
    try:
        module_path, class_name = policy_name.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        policy_class = getattr(module, class_name)
        if not issubclass(policy_class, EvictionPolicy):
            raise TypeError(f"{policy_name} is not a subclass of EvictionPolicy")
        return policy_class()
    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(
            f"Unknown policy '{policy_name}'. "
            f"Available built-in policies: {list(BUILTIN_POLICIES.keys())}. "
            f"For custom policies, use full class path: 'mymodule.MyPolicy'"
        ) from e


__all__ = ["EvictionPolicy", "LRUPolicy", "FIFOPolicy", "get_policy", "BUILTIN_POLICIES"]