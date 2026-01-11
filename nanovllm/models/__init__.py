"""Model registry and model implementations."""

from nanovllm.models.registry import register_model, get_model_class, MODEL_REGISTRY

# Import models to trigger registration
# Qwen3 requires transformers>=4.51.0 for Qwen3Config
try:
    from nanovllm.models import qwen3
except ImportError as e:
    import warnings
    warnings.warn(f"Qwen3 model not available (requires transformers>=4.51.0): {e}")

from nanovllm.models import llama

__all__ = ["register_model", "get_model_class", "MODEL_REGISTRY"]
