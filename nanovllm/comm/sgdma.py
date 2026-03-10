"""
Scatter-Gather DMA utilities using cudaMemcpy2D for efficient strided memory transfers.

Author: Zijie Tian
"""

import torch
from typing import Literal, Optional

try:
    from nanovllm.comm._sgdma_cuda import memcpy_2d as _memcpy_2d_cuda
    from nanovllm.comm._sgdma_cuda import memcpy_2d_async as _memcpy_2d_async_cuda

    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    _import_error = e


def memcpy_2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    dpitch: int,
    spitch: int,
    width: int,
    height: int,
    kind: Literal["h2d", "d2h", "d2d", "h2h"] = "h2d",
) -> None:
    """
    Perform 2D memory copy using cudaMemcpy2D for efficient strided transfers.

    This function enables efficient copying of strided (non-contiguous) memory layouts
    without requiring data reorganization. It's particularly useful for transferring
    blocks from multi-dimensional tensors where dimensions are not in the desired order.

    Args:
        dst: Destination tensor
        src: Source tensor
        dpitch: Destination pitch in bytes (stride between rows in destination)
        spitch: Source pitch in bytes (stride between rows in source)
        width: Width of data to copy per row in bytes
        height: Number of rows to copy
        kind: Transfer direction
            - "h2d": Host to Device (CPU to GPU)
            - "d2h": Device to Host (GPU to CPU)
            - "d2d": Device to Device (GPU to GPU)
            - "h2h": Host to Host (CPU to CPU)

    Raises:
        RuntimeError: If CUDA extension is not compiled
        ValueError: If pitch/width parameters are invalid
        ValueError: If tensor devices don't match the transfer kind

    Example:
        >>> # Scenario: Copy a single block from all layers in strided CPU layout
        >>> # CPU layout: [num_layers=32, num_blocks=100, block_features=8192]
        >>> cpu_cache = torch.randn(32, 100, 8192, dtype=torch.float16, pin_memory=True)
        >>> gpu_buffer = torch.empty(32, 8192, dtype=torch.float16, device='cuda')
        >>>
        >>> # Copy block_id=50 from all layers
        >>> block_id = 50
        >>> dtype_size = 2  # float16
        >>> spitch = 100 * 8192 * dtype_size  # num_blocks * features * dtype_size
        >>> dpitch = 8192 * dtype_size        # features * dtype_size (contiguous)
        >>> width = 8192 * dtype_size         # bytes per row
        >>> height = 32                       # num_layers
        >>>
        >>> # Source pointer: first element of block_id in layer 0
        >>> # In strided layout, we need to point to cpu_cache[0, block_id, 0]
        >>> src_view = cpu_cache[:, block_id, :]  # This creates a strided view
        >>> memcpy_2d(gpu_buffer, src_view, dpitch, spitch, width, height, "h2d")

    Technical Notes:
        - Both dpitch and spitch must be >= width
        - For contiguous transfers, set dpitch = spitch = width
        - The function handles non-contiguous source tensors efficiently using
          cudaMemcpy2D's pitch parameters, avoiding the need for temporary buffers
        - Pinned memory (pin_memory=True) is recommended for CPU tensors to
          achieve optimal transfer bandwidth

    Performance:
        - Strided transfers achieve ~25 GB/s on PCIe Gen3 x16 (same as contiguous)
        - Much faster than layer-by-layer cudaMemcpy calls (~1.02x speedup)
        - Avoids the 16x slowdown of PyTorch's non-contiguous tensor transfers
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA extension not compiled. Please run: python setup.py build_ext --inplace\n"
            f"Original import error: {_import_error}"
        )

    # Validate pitch parameters
    if dpitch < width:
        raise ValueError(f"dpitch ({dpitch}) must be >= width ({width})")
    if spitch < width:
        raise ValueError(f"spitch ({spitch}) must be >= width ({width})")

    # The C++ extension will validate tensor devices
    _memcpy_2d_cuda(dst, src, dpitch, spitch, width, height, kind)


def memcpy_2d_async(
    dst: torch.Tensor,
    src: torch.Tensor,
    dpitch: int,
    spitch: int,
    width: int,
    height: int,
    kind: Literal["h2d", "d2h", "d2d", "h2h"] = "h2d",
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Asynchronous version of memcpy_2d using cudaMemcpy2DAsync.

    All parameters are the same as memcpy_2d, with an additional stream parameter.

    Args:
        dst: Destination tensor
        src: Source tensor
        dpitch: Destination pitch in bytes
        spitch: Source pitch in bytes
        width: Width to copy per row in bytes
        height: Number of rows
        kind: Transfer direction ("h2d", "d2h", "d2d", "h2h")
        stream: CUDA stream for async execution (default: current stream)

    Example:
        >>> stream = torch.cuda.Stream()
        >>> with torch.cuda.stream(stream):
        ...     memcpy_2d_async(dst, src, dpitch, spitch, width, height, "h2d", stream)
        ...     # Other operations can overlap with transfer
        >>> stream.synchronize()  # Wait for transfer to complete

    Note:
        - For async H2D/D2H transfers, source memory must be pinned (pin_memory=True)
        - The stream will be synchronized before the transfer completes
        - Use stream.synchronize() or torch.cuda.synchronize() to wait
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA extension not compiled. Please run: python setup.py build_ext --inplace\n"
            f"Original import error: {_import_error}"
        )

    # Validate pitch parameters
    if dpitch < width:
        raise ValueError(f"dpitch ({dpitch}) must be >= width ({width})")
    if spitch < width:
        raise ValueError(f"spitch ({spitch}) must be >= width ({width})")

    # Get stream pointer
    if stream is None:
        stream = torch.cuda.current_stream()

    stream_ptr = stream.cuda_stream

    # The C++ extension will validate tensor devices
    _memcpy_2d_async_cuda(dst, src, dpitch, spitch, width, height, kind, stream_ptr)
