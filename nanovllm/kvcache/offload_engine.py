"""
High-performance CPU-GPU KV cache transfer engine.

Key design principles for CUDA Graph compatibility:
1. All tensor addresses are fixed at initialization
2. Only index tensor contents change between graph replays
3. Supports both async transfer (for prefill) and graph-based transfer (for decode)
"""

import torch
import torch.cuda.nvtx
import nvtx
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from nanovllm.kvcache.kernels import gathered_copy_kv
from nanovllm.comm import memcpy_2d_async
from nanovllm.utils.logger import get_logger
from nanovllm.utils.memory_observer import MemoryObserver

# Import for type hints only (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanovllm.kvcache.sparse import SparsePolicy

logger = get_logger("offload_engine")


@dataclass
class TransferEvent:
    """Tracks a pending async transfer."""
    event: torch.cuda.Event
    layer_id: int
    src_block_id: int
    dst_block_id: int
    direction: str  # "h2d" or "d2h"


class OffloadEngine:
    """
    High-performance CPU-GPU async transfer engine for KV cache offloading.

    Memory layout:
    - GPU cache: [num_gpu_blocks, block_size, kv_heads, head_dim] (no layer dimension)
    - CPU cache: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim] (pinned)

    Features:
    - Unified ring buffer for chunked prefill/decode
    - Per-layer prefill buffer for async offload
    - Cross-layer pipeline for decode with double-buffering
    """

    def __init__(
        self,
        num_layers: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        num_streams: int = 4,
        sparse_policy: "SparsePolicy" = None,
    ):
        self.num_layers = num_layers
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.kv_dim = num_kv_heads * head_dim
        self.block_numel = block_size * self.kv_dim

        # ========== sgDMA pitch parameters for strided transfers ==========
        # CPU cache: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
        # GPU cache: [num_gpu_blocks, block_size, kv_heads, head_dim] (no layer dim)
        # For CPU-to-GPU transfer (H2D): copy single layer, single block at a time
        # For all-layer CPU operations (D2H offload to all layers): use sgDMA
        self.dtype_size = dtype.itemsize
        # CPU pitch: stride between layers in CPU cache (for all-layer operations)
        self.cpu_pitch = num_cpu_blocks * self.block_numel * self.dtype_size
        # GPU has no layer dimension, so single block transfer is contiguous
        self.gpu_block_bytes = self.block_numel * self.dtype_size
        self.height = num_layers  # For CPU all-layer operations

        logger.info(f"sgDMA parameters: cpu_pitch={self.cpu_pitch}, "
                    f"gpu_block_bytes={self.gpu_block_bytes}, height={self.height}")

        # ========== Unified Ring Buffer configuration ==========
        # Constraint checks
        assert num_gpu_blocks >= 2, \
            f"Need at least 2 GPU blocks for ring buffer, got {num_gpu_blocks}"

        # Unified Ring Buffer: all slots cycle for prefill
        # Prefill: use ALL slots as ring buffer (slot[chunk_idx % N])
        # Decode: slot[0] as decode_slot, slots[1:] for loading previous chunks
        self.num_ring_slots = num_gpu_blocks
        self.ring_slots = list(range(num_gpu_blocks))

        # Decode phase uses slot[0] for writing new token's KV
        self.decode_slot = 0
        # Decode phase uses slots[1:] for loading previous chunks from CPU
        self.decode_load_slots = list(range(1, num_gpu_blocks))
        self.num_decode_load_slots = len(self.decode_load_slots)

        self.num_gpu_slots = num_gpu_blocks  # alias

        logger.info(f"Unified Ring Buffer: {self.num_ring_slots} slots total")
        logger.info(f"  Prefill: all slots as ring buffer [0..{num_gpu_blocks-1}]")
        logger.info(f"  Decode: slot[0] as decode_slot, slots[1..{num_gpu_blocks-1}] for loading")

        # ========== Fixed-address GPU KV cache ==========
        # Shape: [num_gpu_blocks, block_size, kv_heads, head_dim]
        # NOTE: No num_layers dimension! GPU slots are shared across layers.
        # Each layer reuses the same slots (layers execute sequentially).
        # This saves 28x GPU memory compared to per-layer allocation.
        self.k_cache_gpu = torch.zeros(
            num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        self.v_cache_gpu = torch.zeros(
            num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )

        # ========== Per-layer decode buffer ==========
        # During decode, all layers share decode_slot (no layer dimension in GPU cache).
        # This causes accumulated tokens to be overwritten by each layer.
        # Solution: Maintain separate per-layer buffers for decode tokens.
        # Shape: [num_layers, block_size, kv_heads, head_dim]
        # Memory: num_layers * block_size * kv_heads * head_dim * dtype_size
        #         e.g., 28 * 1024 * 8 * 128 * 2 = 58.7 MB (acceptable)
        self.decode_k_buffer = torch.zeros(
            num_layers, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        self.decode_v_buffer = torch.zeros(
            num_layers, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        decode_buf_mb = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        logger.info(f"  Per-layer decode buffer: {decode_buf_mb:.1f} MB")

        # ========== Per-layer prefill buffer for async offload ==========
        # During chunked prefill, all layers share the same GPU slot. This means
        # each layer must wait for offload to complete before the next layer can
        # write to the same slot. This serializes offloads and hurts performance.
        # Solution: Maintain separate per-layer buffers for prefill.
        # Each layer writes to its own buffer, enabling fully async offloads.
        # Shape: [num_layers, block_size, kv_heads, head_dim]
        self.prefill_k_buffer = torch.zeros(
            num_layers, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        self.prefill_v_buffer = torch.zeros(
            num_layers, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        prefill_buf_mb = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        logger.info(f"  Per-layer prefill buffer: {prefill_buf_mb:.1f} MB")

        # Per-layer offload events for async prefill offload
        # Each layer has its own event to track offload completion
        self.prefill_offload_events = [torch.cuda.Event() for _ in range(num_layers)]
        # Per-layer transfer streams for parallel offloads
        self.prefill_offload_streams = [torch.cuda.Stream() for _ in range(num_layers)]

        # ========== Fixed-address CPU KV cache (pinned memory) ==========
        self.k_cache_cpu = torch.zeros(
            num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cpu", pin_memory=True
        )
        self.v_cache_cpu = torch.zeros(
            num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cpu", pin_memory=True
        )

        # Log memory allocation
        gpu_mem_mb = self.gpu_memory_bytes() / (1024 * 1024)
        cpu_mem_mb = self.cpu_memory_bytes() / (1024 * 1024)
        logger.info(f"  GPU memory: {gpu_mem_mb:.1f} MB, CPU memory: {cpu_mem_mb:.1f} MB")

        # ========== Transfer streams for async operations ==========
        self.transfer_streams = [torch.cuda.Stream() for _ in range(num_streams)]
        # IMPORTANT: Create a dedicated compute stream (not default stream!)
        # Default stream has implicit synchronization with other streams,
        # which prevents overlap between transfer and compute.
        self.compute_stream = torch.cuda.Stream()
        self._stream_idx = 0

        # ========== Per-slot transfer streams for parallel H2D ==========
        # Each slot has its own stream to enable parallel transfers
        # This allows multiple slots to load simultaneously
        self.slot_transfer_streams = [torch.cuda.Stream() for _ in range(self.num_ring_slots)]
        logger.info(f"  Created {self.num_ring_slots} per-slot transfer streams")

        # ========== Ring Buffer dedicated stream and events ==========
        self.transfer_stream_main = torch.cuda.Stream()  # Main transfer stream (for legacy/batch ops)

        # Decode offload event
        self.decode_offload_done = torch.cuda.Event()

        # ========== Per-slot events for ring buffer ==========
        # Since GPU cache has no layer dimension and layers execute sequentially,
        # we only need per-slot events (not per-slot per-layer).
        # ring_slot_ready[slot_idx] = CUDA Event for H2D completion
        # ring_slot_offload_done[slot_idx] = CUDA Event for D2H completion
        self.ring_slot_ready = [torch.cuda.Event() for _ in range(self.num_ring_slots)]
        self.ring_slot_offload_done = [torch.cuda.Event() for _ in range(self.num_ring_slots)]

        # ========== Per-slot compute_done events for async pipeline ==========
        # ring_slot_compute_done[slot_idx] = CUDA Event for compute completion
        # This ensures we don't overwrite data before it's been read by attention
        self.ring_slot_compute_done = [torch.cuda.Event() for _ in range(self.num_ring_slots)]

        # Initialize all compute_done events (record them once)
        # This prevents undefined behavior on first load_to_slot_layer call
        for slot_idx in range(self.num_ring_slots):
            self.ring_slot_compute_done[slot_idx].record()
        # torch.cuda.synchronize()  # Ensure all events are recorded

        # ========== Event tracking for async transfers ==========
        self.pending_events: Dict[Tuple[int, int], torch.cuda.Event] = {}

        # ========== Debug hook mode ==========
        self._debug_mode = False
        self._debug_hooks: List = []  # External hooks for debug events

        # ========== Sparse attention policy (set at construction time) ==========
        self.sparse_policy = sparse_policy

    # ========== Cache access methods ==========

    def get_layer_cache(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        """
        Get GPU K/V cache tensors for attention layer.

        NOTE: GPU cache has no layer dimension - all layers share the same slots.
        The layer_id parameter is kept for API compatibility but not used.

        Returns:
            (k_cache, v_cache) tensors
            Shape: [num_gpu_blocks, block_size, kv_heads, head_dim]
        """
        return self.k_cache_gpu, self.v_cache_gpu

    def reset(self) -> None:
        """
        Reset all KV cache buffers to zero.

        This clears all GPU and CPU-side KV cache storage, preventing
        request-to-request contamination. Must be called between generate()
        calls when reusing the same OffloadEngine instance.

        Clears:
        - GPU ring buffer slots (k_cache_gpu, v_cache_gpu)
        - Per-layer decode buffers (decode_k_buffer, decode_v_buffer)
        - Per-layer prefill buffers (prefill_k/v_buffer)
        - CPU KV cache (k_cache_cpu, v_cache_cpu)
        - All pending async transfer events
        """
        # Clear GPU ring buffer slots
        self.k_cache_gpu.zero_()
        self.v_cache_gpu.zero_()

        # Clear per-layer decode buffers
        self.decode_k_buffer.zero_()
        self.decode_v_buffer.zero_()

        # Clear per-layer prefill buffers
        self.prefill_k_buffer.zero_()
        self.prefill_v_buffer.zero_()

        # Clear CPU cache (critical: prevents cross-request state leakage)
        # This ensures KV cache from previous requests doesn't contaminate new requests
        self.k_cache_cpu.zero_()
        self.v_cache_cpu.zero_()

        # Clear all pending async transfer events
        self.pending_events.clear()

    # ========== Memory info ==========

    def gpu_memory_bytes(self) -> int:
        """Total GPU memory used by KV caches."""
        return (
            self.k_cache_gpu.numel() * self.k_cache_gpu.element_size() +
            self.v_cache_gpu.numel() * self.v_cache_gpu.element_size()
        )

    def cpu_memory_bytes(self) -> int:
        """Total CPU memory used by KV caches."""
        return (
            self.k_cache_cpu.numel() * self.k_cache_cpu.element_size() +
            self.v_cache_cpu.numel() * self.v_cache_cpu.element_size()
        )

    def __repr__(self) -> str:
        return (
            f"OffloadEngine(\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_gpu_blocks={self.num_gpu_blocks},\n"
            f"  num_cpu_blocks={self.num_cpu_blocks},\n"
            f"  block_size={self.block_size},\n"
            f"  kv_heads={self.num_kv_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  dtype={self.dtype},\n"
            f"  ring_buffer: {self.num_ring_slots} slots, decode_slot={self.decode_slot}, decode_load_slots={self.decode_load_slots},\n"
            f"  gpu_memory={self.gpu_memory_bytes() / 1024**2:.1f}MB,\n"
            f"  cpu_memory={self.cpu_memory_bytes() / 1024**2:.1f}MB\n"
            f")"
        )

    def wait_all_offload_done(self) -> None:
        """Wait for all offload operations to complete."""
        self.transfer_stream_main.synchronize()

    # ========== Unified Ring Buffer methods ==========

    # ----- Prefill: Ring Buffer slot management -----

    def get_write_slot_for_prefill(self, chunk_idx: int) -> int:
        """
        Get ring buffer slot for writing prefill chunk.

        For prefill, ALL slots are used as ring buffer, cycling through.

        Args:
            chunk_idx: Current chunk index (0, 1, 2, ...)

        Returns:
            GPU slot index for writing
        """
        return chunk_idx % self.num_ring_slots

    def get_load_slots_for_prefill(self, write_slot_idx: int) -> List[int]:
        """
        Get available slots for loading previous chunks during prefill.

        Excludes the current write slot to avoid conflict.

        Args:
            write_slot_idx: Current write slot index

        Returns:
            List of slot indices available for loading (N-1 slots)
        """
        return [i for i in range(self.num_ring_slots) if i != write_slot_idx]

    # ----- Decode: slot management -----

    def get_load_slots_for_decode(self) -> List[int]:
        """
        Get slots available for loading during decode.

        Excludes decode_slot (slot[0]) since it's used for writing new token's KV.

        Returns:
            List of slot indices for loading (slots[1:])
        """
        return self.decode_load_slots

    # ----- Per-slot Per-layer loading methods -----

    def record_slot_compute_done(self, slot_idx: int) -> None:
        """
        Record that computation using this slot's data is done.

        This event is used by load_to_slot_layer to ensure we don't overwrite
        data before it's been read by attention computation.

        Args:
            slot_idx: GPU slot index that was just used for computation
        """
        self.ring_slot_compute_done[slot_idx].record()

    def load_to_slot_layer(
        self, slot_idx: int, layer_id: int, cpu_block_id: int, chunk_idx: int = -1,
        is_prefill: bool = True,
    ) -> None:
        """
        Async load a single CPU block to a ring buffer slot for one layer.

        This is the core building block for ring buffer pipelining.
        GPU cache has no layer dimension - slots are shared across all layers.
        CPU cache still has layer dimension for persistent storage.

        Before starting the transfer, waits for:
        1. Any previous compute on this slot to complete

        Args:
            slot_idx: Target GPU slot index
            layer_id: Layer index to load (for CPU cache indexing)
            cpu_block_id: Source CPU block ID
            chunk_idx: Optional chunk index for NVTX labeling (-1 means not specified)
            is_prefill: True if in prefill phase, False if in decode phase (for MemoryObserver)
        """
        logger.debug(f"Ring load: layer={layer_id}, CPU[{cpu_block_id}] -> GPU slot[{slot_idx}]")

        # Use per-slot stream for parallel transfers across different slots
        stream = self.slot_transfer_streams[slot_idx]

        # Build NVTX label with optional chunk info
        if chunk_idx >= 0:
            nvtx_label = f"H2D: L{layer_id} Chunk{chunk_idx} CPU[{cpu_block_id}]->Slot[{slot_idx}]"
        else:
            nvtx_label = f"H2D: L{layer_id} CPU[{cpu_block_id}]->Slot[{slot_idx}]"

        nvtx.push_range(message=nvtx_label, color="blue")
        with torch.cuda.stream(stream):
            # Wait for previous compute on this slot to complete before overwriting
            # This prevents data race: transfer must not start until attention finishes reading
            stream.wait_event(self.ring_slot_compute_done[slot_idx])

            # Also wait for any pending offload of this slot to complete
            # This prevents race: load must not write GPU slot while offload is reading from it
            stream.wait_event(self.ring_slot_offload_done[slot_idx])

            # GPU: no layer dimension, CPU: has layer dimension
            self.k_cache_gpu[slot_idx].copy_(
                self.k_cache_cpu[layer_id, cpu_block_id], non_blocking=True
            )
            self.v_cache_gpu[slot_idx].copy_(
                self.v_cache_cpu[layer_id, cpu_block_id], non_blocking=True
            )
            self.ring_slot_ready[slot_idx].record(stream)
        nvtx.pop_range()

        # Record H2D transfer: K + V = 2 * block_bytes
        MemoryObserver.record_h2d(2 * self.gpu_block_bytes, is_prefill=is_prefill)

    def load_k_only_to_slot_layer(
        self, slot_idx: int, layer_id: int, cpu_block_id: int, chunk_idx: int = -1,
        is_prefill: bool = True,
    ) -> None:
        """
        Async load only K (not V) from CPU block to GPU slot.

        Used by XAttention estimate phase which only needs K for attention score
        computation. Saves 50% communication compared to loading K+V.

        Args:
            slot_idx: Target GPU slot index
            layer_id: Layer index to load (for CPU cache indexing)
            cpu_block_id: Source CPU block ID
            chunk_idx: Optional chunk index for NVTX labeling (-1 means not specified)
            is_prefill: True if in prefill phase, False if in decode phase
        """
        logger.debug(f"Ring load K-only: layer={layer_id}, CPU[{cpu_block_id}] -> GPU slot[{slot_idx}]")

        stream = self.slot_transfer_streams[slot_idx]

        if chunk_idx >= 0:
            nvtx_label = f"H2D K-only: L{layer_id} Chunk{chunk_idx} CPU[{cpu_block_id}]->Slot[{slot_idx}]"
        else:
            nvtx_label = f"H2D K-only: L{layer_id} CPU[{cpu_block_id}]->Slot[{slot_idx}]"

        nvtx.push_range(message=nvtx_label, color="cyan")
        with torch.cuda.stream(stream):
            stream.wait_event(self.ring_slot_compute_done[slot_idx])
            stream.wait_event(self.ring_slot_offload_done[slot_idx])

            # Only copy K, not V
            self.k_cache_gpu[slot_idx].copy_(
                self.k_cache_cpu[layer_id, cpu_block_id], non_blocking=True
            )
            self.ring_slot_ready[slot_idx].record(stream)
        nvtx.pop_range()

        # Record H2D transfer: K only = 1 * block_bytes
        MemoryObserver.record_h2d(self.gpu_block_bytes, is_prefill=is_prefill)

    def get_k_for_slot(self, slot_idx: int) -> Tensor:
        """
        Get only K for a ring buffer slot (no V).

        Used by XAttention estimate phase which only needs K for attention
        score computation.

        Args:
            slot_idx: GPU slot index

        Returns:
            k_cache, shape: [1, block_size, kv_heads, head_dim]
        """
        return self.k_cache_gpu[slot_idx].unsqueeze(0)

    def wait_slot_layer(self, slot_idx: int) -> None:
        """
        Wait for a slot's loading to complete.

        Args:
            slot_idx: GPU slot index to wait for
        """
        self.compute_stream.wait_event(self.ring_slot_ready[slot_idx])

    # NOTE: load_to_slot_all_layers removed - GPU cache no longer has layer dimension.
    # Each GPU slot holds data for ONE layer at a time. Layers execute sequentially,
    # reusing the same GPU slots.

    # ----- Slot offload methods -----

    # NOTE: offload_slot_to_cpu (all-layers) removed - GPU cache no longer has layer dimension.
    # Use offload_slot_layer_to_cpu for per-layer offloading.

    def wait_slot_offload(self, slot_idx: int) -> None:
        """Wait for slot offload to complete."""
        self.compute_stream.wait_event(self.ring_slot_offload_done[slot_idx])

    def offload_slot_layer_to_cpu(
        self,
        slot_idx: int,
        layer_id: int,
        cpu_block_id: int,
        num_valid_tokens: int = -1,
        is_prefill: bool = True,
    ) -> None:
        """
        Async offload a ring buffer slot to CPU for one layer.

        GPU cache has no layer dimension, so we copy from GPU slot to the
        specific layer in CPU cache.

        Args:
            slot_idx: Source GPU slot index
            layer_id: Target layer in CPU cache
            cpu_block_id: Target CPU block ID
            num_valid_tokens: Number of valid tokens in this block (-1 = use block_size)
            is_prefill: True if in prefill phase, False if in decode phase
        """
        logger.debug(f"Ring offload: GPU slot[{slot_idx}] -> CPU[layer={layer_id}, block={cpu_block_id}]")

        # Collect metadata BEFORE offload (while k_cache is still on GPU)
        valid_tokens = num_valid_tokens if num_valid_tokens > 0 else self.block_size
        k_cache = self.k_cache_gpu[slot_idx]

        if self.sparse_policy is not None:
            if is_prefill:
                self.sparse_policy.on_prefill_offload(cpu_block_id, layer_id, k_cache, valid_tokens)
            else:
                self.sparse_policy.on_decode_offload(cpu_block_id, layer_id, k_cache, valid_tokens)

        nvtx_label = f"D2H: Slot[{slot_idx}]->CPU[L{layer_id},B{cpu_block_id}]"
        nvtx.push_range(message=nvtx_label, color="green")
        with torch.cuda.stream(self.transfer_stream_main):
            # Wait for both compute_stream and default stream
            # - compute_stream: for flash attention operations
            # - default_stream: for store_kvcache which runs on default stream
            self.transfer_stream_main.wait_stream(self.compute_stream)
            self.transfer_stream_main.wait_stream(torch.cuda.default_stream())

            # GPU: no layer dimension, CPU: has layer dimension
            self.k_cache_cpu[layer_id, cpu_block_id].copy_(
                self.k_cache_gpu[slot_idx], non_blocking=True
            )
            self.v_cache_cpu[layer_id, cpu_block_id].copy_(
                self.v_cache_gpu[slot_idx], non_blocking=True
            )
            self.ring_slot_offload_done[slot_idx].record(self.transfer_stream_main)
        nvtx.pop_range()

        # Record D2H transfer: K + V = 2 * block_bytes
        MemoryObserver.record_d2h(2 * self.gpu_block_bytes, is_prefill=is_prefill)

    # ----- KV access methods for ring buffer -----

    def get_kv_for_slot(self, slot_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get KV for a single ring buffer slot.

        GPU cache has no layer dimension - slots contain data for whatever
        layer was most recently loaded.

        Args:
            slot_idx: GPU slot index

        Returns:
            (k_cache, v_cache), shape: [1, block_size, kv_heads, head_dim]
        """
        k = self.k_cache_gpu[slot_idx].unsqueeze(0)  # [1, block_size, heads, dim]
        v = self.v_cache_gpu[slot_idx].unsqueeze(0)
        return k, v

    def get_kv_for_slots(
        self,
        slot_indices: List[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for multiple ring buffer slots.

        GPU cache has no layer dimension - returns data from specified slots.

        Args:
            slot_indices: List of GPU slot indices

        Returns:
            (k_cache, v_cache), shape: [1, len(slots) * block_size, kv_heads, head_dim]
        """
        if not slot_indices:
            return None, None
        k = self.k_cache_gpu[slot_indices]
        v = self.v_cache_gpu[slot_indices]
        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        return k, v

    # ----- Decode slot methods (kept for decode phase) -----
    # NOTE: For decode with CPU offload, the flow is per-layer:
    # 1. Each layer stores to decode_slot (same GPU memory, reused)
    # 2. Each layer offloads its data to CPU[layer_id, block_id]
    # 3. Each layer loads prev blocks from CPU[layer_id] when needed

    def offload_decode_slot_layer(self, layer_id: int, cpu_block_id: int) -> None:
        """
        Offload KV from decode slot (slot[0]) to CPU for one layer.

        Args:
            layer_id: Layer ID
            cpu_block_id: Target CPU block ID
        """
        # Reuse the existing per-layer offload method
        self.offload_slot_layer_to_cpu(self.decode_slot, layer_id, cpu_block_id)

    def wait_decode_offload(self) -> None:
        """Wait for decode slot offload to complete."""
        self.wait_slot_offload(self.decode_slot)

    def get_kv_for_decode_slot(
        self,
        pos_in_block: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV at specified position in decode slot.

        GPU cache has no layer dimension - decode slot contains data for
        whatever layer was most recently stored.

        Args:
            pos_in_block: Token position within block (0 to block_size-1)

        Returns:
            (k_cache, v_cache), shape: [1, 1, kv_heads, head_dim]
        """
        k = self.k_cache_gpu[self.decode_slot, pos_in_block:pos_in_block+1]
        v = self.v_cache_gpu[self.decode_slot, pos_in_block:pos_in_block+1]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        return k, v

    def get_kv_for_decode_slot_accumulated(
        self,
        num_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get accumulated KV in decode slot (positions 0 to num_tokens-1).

        GPU cache has no layer dimension - decode slot contains data for
        whatever layer was most recently stored.

        Args:
            num_tokens: Number of accumulated tokens (1 to block_size)

        Returns:
            (k_cache, v_cache), shape: [1, num_tokens, kv_heads, head_dim]
        """
        k = self.k_cache_gpu[self.decode_slot, :num_tokens]
        v = self.v_cache_gpu[self.decode_slot, :num_tokens]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        return k, v

    # ========== Debug Hook Interface ==========
    #
    # Minimal generic hook system for debugging.
    # Framework only provides hook registration and tensor access.
    # All verification logic is external.

    def enable_debug_mode(self) -> None:
        """Enable debug mode."""
        self._debug_mode = True
        logger.info("OffloadEngine debug mode ENABLED")

    def disable_debug_mode(self) -> None:
        """Disable debug mode and clear all hooks."""
        self._debug_mode = False
        self._debug_hooks.clear()
        logger.info("OffloadEngine debug mode DISABLED")

    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self._debug_mode

    def register_debug_hook(self, hook_fn) -> None:
        """
        Register a debug hook.

        The hook is called after H2D load completes (after wait_slot_layer),
        receiving the loaded tensor for inspection.

        Args:
            hook_fn: Callable with signature:
                (slot_idx: int, layer_id: int, cpu_block_id: int, k: Tensor, v: Tensor) -> None
                - k, v: GPU tensor views for the loaded slot

        Example:
            def my_hook(slot_idx, layer_id, cpu_block_id, k, v):
                if layer_id == 0:
                    k_val = k.float().mean().item()
                    print(f"Loaded block {cpu_block_id}, K mean = {k_val}")

            offload_engine.register_debug_hook(my_hook)
        """
        self._debug_hooks.append(hook_fn)

    def remove_debug_hook(self, hook_fn) -> None:
        """Remove a registered debug hook."""
        if hook_fn in self._debug_hooks:
            self._debug_hooks.remove(hook_fn)

    def _call_debug_hooks(self, slot_idx: int, layer_id: int, cpu_block_id: int) -> None:
        """
        Call all registered debug hooks with loaded tensor (internal use).

        Called by attention.py after wait_slot_layer completes.
        GPU cache has no layer dimension - slot contains data for the layer
        that was just loaded.
        """
        if not self._debug_mode or not self._debug_hooks:
            return

        # Use get_kv_for_slot for consistency with attention.py
        k, v = self.get_kv_for_slot(slot_idx)

        for hook in self._debug_hooks:
            try:
                hook(slot_idx, layer_id, cpu_block_id, k, v)
            except Exception as e:
                # Allow pdb quit to propagate
                if e.__class__.__name__ == 'BdbQuit':
                    raise
                logger.warning(f"Debug hook error: {e}")

    # ========== Per-layer Prefill Buffer Methods ==========
    # These methods enable async offload during chunked prefill by using
    # per-layer buffers instead of shared GPU slots.

    def get_prefill_buffer(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        """
        Get prefill buffer for a layer.

        Args:
            layer_id: Layer index

        Returns:
            (k_buffer, v_buffer), shape: [block_size, kv_heads, head_dim]
        """
        return self.prefill_k_buffer[layer_id], self.prefill_v_buffer[layer_id]

    def get_prefill_buffer_slice(
        self,
        layer_id: int,
        num_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get a slice of prefill buffer for attention computation.

        Args:
            layer_id: Layer index
            num_tokens: Number of valid tokens in current chunk

        Returns:
            (k, v) with shape [1, num_tokens, kv_heads, head_dim]
        """
        k = self.prefill_k_buffer[layer_id, :num_tokens].unsqueeze(0)
        v = self.prefill_v_buffer[layer_id, :num_tokens].unsqueeze(0)
        return k, v

    def write_to_prefill_buffer(
        self,
        layer_id: int,
        k: Tensor,
        v: Tensor,
        chunk_idx: int = -1,
    ) -> None:
        """
        Write KV tensors to prefill buffer (D2D copy within GPU).

        This is called during chunked prefill to store current chunk's KV
        before computing attention.

        Args:
            layer_id: Layer index
            k: Key tensor [num_tokens, kv_heads, head_dim]
            v: Value tensor [num_tokens, kv_heads, head_dim]
            chunk_idx: Current chunk index for NVTX labeling (-1 = not specified)
        """
        num_tokens = k.shape[0]

        # Build NVTX label
        if chunk_idx >= 0:
            nvtx_label = f"D2D: L{layer_id} Chunk{chunk_idx} WritePrefillBuffer"
        else:
            nvtx_label = f"D2D: L{layer_id} WritePrefillBuffer"

        torch.cuda.nvtx.range_push(nvtx_label)
        self.prefill_k_buffer[layer_id, :num_tokens].copy_(k)
        self.prefill_v_buffer[layer_id, :num_tokens].copy_(v)
        torch.cuda.nvtx.range_pop()

        # Record D2D transfer: K + V
        transfer_bytes = 2 * k.numel() * k.element_size()
        MemoryObserver.record_d2d(transfer_bytes)

    def write_to_decode_buffer(
        self,
        layer_id: int,
        pos_in_block: int,
        k: Tensor,
        v: Tensor,
    ) -> None:
        """
        Write KV tensors to decode buffer (D2D copy within GPU).

        This is called during chunked decode to store current decode token's KV.

        Args:
            layer_id: Layer index
            pos_in_block: Position within the current block
            k: Key tensor [kv_heads, head_dim] (single token, squeezed)
            v: Value tensor [kv_heads, head_dim] (single token, squeezed)
        """
        torch.cuda.nvtx.range_push(f"D2D: L{layer_id} Pos{pos_in_block} WriteDecodeBuffer")
        self.decode_k_buffer[layer_id, pos_in_block].copy_(k)
        self.decode_v_buffer[layer_id, pos_in_block].copy_(v)
        torch.cuda.nvtx.range_pop()

        # Record D2D transfer: K + V (single token)
        transfer_bytes = 2 * k.numel() * k.element_size()
        MemoryObserver.record_d2d(transfer_bytes)

    def offload_prefill_buffer_async(
        self,
        layer_id: int,
        cpu_block_id: int,
        num_valid_tokens: int = -1,
    ) -> None:
        """
        Async offload prefill buffer to CPU (no waiting required).

        This uses per-layer streams and events to enable fully async offloads.
        Each layer can offload independently without blocking other layers.

        Args:
            layer_id: Layer index
            cpu_block_id: Target CPU block ID
            num_valid_tokens: Number of valid tokens (-1 = use block_size)
        """
        valid_tokens = num_valid_tokens if num_valid_tokens > 0 else self.block_size

        # Collect sparse policy metadata before offload
        if self.sparse_policy is not None:
            k_cache = self.prefill_k_buffer[layer_id]
            self.sparse_policy.on_prefill_offload(cpu_block_id, layer_id, k_cache, valid_tokens)

        # Use per-layer stream for parallel offloads
        stream = self.prefill_offload_streams[layer_id]

        nvtx_label = f"D2H: PrefillBuffer L{layer_id}->CPU[{cpu_block_id}]"
        nvtx.push_range(message=nvtx_label, color="orange")
        with torch.cuda.stream(stream):
            # Wait for compute to finish writing to prefill buffer
            stream.wait_stream(self.compute_stream)

            # Copy from prefill buffer to CPU
            self.k_cache_cpu[layer_id, cpu_block_id].copy_(
                self.prefill_k_buffer[layer_id], non_blocking=True
            )
            self.v_cache_cpu[layer_id, cpu_block_id].copy_(
                self.prefill_v_buffer[layer_id], non_blocking=True
            )

            # Record completion event
            self.prefill_offload_events[layer_id].record(stream)
        nvtx.pop_range()

        # Record D2H transfer: K + V = 2 * block_bytes
        MemoryObserver.record_d2h(2 * self.gpu_block_bytes, is_prefill=True)

    def wait_all_prefill_offloads(self) -> None:
        """Wait for all prefill buffer offloads to complete."""
        for stream in self.prefill_offload_streams:
            stream.synchronize()

    def wait_prefill_offload(self, layer_id: int) -> None:
        """Wait for a specific layer's prefill offload to complete."""
        self.prefill_offload_events[layer_id].synchronize()

    # ========== XAttention BSA Helper Methods ==========

    def load_block_sample_from_cpu(
        self,
        cpu_block_id: int,
        layer_id: int,
        num_samples: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Load sample tokens from a CPU block for XAttention BSA estimation.

        This is used in the estimate phase of XAttention BSA to load a small
        sample of tokens from each historical chunk for importance estimation.

        Args:
            cpu_block_id: Source CPU block ID
            layer_id: Layer index
            num_samples: Number of tokens to sample

        Returns:
            (k_sample, v_sample) tensors, shape: [num_samples, kv_heads, head_dim]
        """
        # Sample from the beginning of the block
        k_sample = self.k_cache_cpu[
            layer_id, cpu_block_id, :num_samples
        ].clone().cuda()
        v_sample = self.v_cache_cpu[
            layer_id, cpu_block_id, :num_samples
        ].clone().cuda()

        # Record H2D transfer: K + V samples
        transfer_bytes = 2 * k_sample.numel() * k_sample.element_size()
        MemoryObserver.record_h2d(transfer_bytes, is_prefill=True)

        return k_sample, v_sample

    def load_block_full_from_cpu(
        self,
        cpu_block_id: int,
        layer_id: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Load full tokens from a CPU block for XAttention BSA computation.

        This is used in the compute phase of XAttention BSA to load the full
        data for selected important chunks.

        Args:
            cpu_block_id: Source CPU block ID
            layer_id: Layer index

        Returns:
            (k_full, v_full) tensors, shape: [block_size, kv_heads, head_dim]
        """
        k_full = self.k_cache_cpu[
            layer_id, cpu_block_id
        ].clone().cuda()
        v_full = self.v_cache_cpu[
            layer_id, cpu_block_id
        ].clone().cuda()

        # Record H2D transfer: K + V full block
        MemoryObserver.record_h2d(2 * self.gpu_block_bytes, is_prefill=True)

        return k_full, v_full
