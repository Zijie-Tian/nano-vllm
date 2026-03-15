"""Model registry and model implementations."""

from nanovllm.models.registry import register_model, get_model_class, MODEL_REGISTRY

# Import models to trigger registration
from nanovllm.models import glm4, llama, qwen2  # noqa: F401
try:
    from nanovllm.models import qwen3  # noqa: F401
except ImportError:
    pass

__all__ = ["register_model", "get_model_class", "MODEL_REGISTRY"]
