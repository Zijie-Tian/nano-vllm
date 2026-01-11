"""Model registry and model implementations."""

from nanovllm.models.registry import register_model, get_model_class, MODEL_REGISTRY

# Import models to trigger registration
# Llama is always available
from nanovllm.models import llama

# Qwen3 requires transformers >= 4.51.0
try:
    from nanovllm.models import qwen3
except ImportError:
    import warnings
    warnings.warn(
        "Qwen3 models require transformers >= 4.51.0. "
        "Install with: pip install 'transformers>=4.51.0'"
    )

__all__ = ["register_model", "get_model_class", "MODEL_REGISTRY"]
