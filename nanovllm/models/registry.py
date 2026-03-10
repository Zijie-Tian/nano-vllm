"""Model registry for dynamic model loading."""

from typing import Type
from torch import nn

# Global registry mapping architecture names to model classes
MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(*architectures: str):
    """
    Decorator to register a model class for given architecture names.

    Usage:
        @register_model("LlamaForCausalLM")
        class LlamaForCausalLM(nn.Module):
            ...
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        for arch in architectures:
            MODEL_REGISTRY[arch] = cls
        return cls

    return decorator


def get_model_class(hf_config) -> Type[nn.Module]:
    """
    Get model class based on HuggingFace config.

    Args:
        hf_config: HuggingFace model config with 'architectures' field

    Returns:
        Model class for the given architecture

    Raises:
        ValueError: If architecture is not supported
    """
    architectures = getattr(hf_config, "architectures", [])
    for arch in architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(
        f"Unsupported architecture: {architectures}. "
        f"Supported: {list(MODEL_REGISTRY.keys())}"
    )
