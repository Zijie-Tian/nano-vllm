"""Communication utilities for nano-vLLM, including sgDMA support."""

try:
    from .sgdma import memcpy_2d, memcpy_2d_async
    __all__ = ['memcpy_2d', 'memcpy_2d_async']
except ImportError:
    # Extension not compiled yet
    __all__ = []
