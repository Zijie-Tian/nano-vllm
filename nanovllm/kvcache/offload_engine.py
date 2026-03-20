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
from typing import Dict, List, Tuple


from nanovllm.utils.logger import get_logger
from nanovllm.utils.memory_observer import MemoryObserver

# Import for type hints only (avoid circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.kvcache.sparse import SparsePolicy

logger = get_logger("offload_engine")



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

        logger.info(
            f"sgDMA parameters: cpu_pitch={self.cpu_pitch}, "
            f"gpu_block_bytes={self.gpu_block_bytes}, height={self.height}"
        )

        # ========== Unified Ring Buffer configuration ==========
        # Constraint checks
        assert num_gpu_blocks >= 2, (
            f"Need at least 2 GPU blocks for ring buffer, got {num_gpu_blocks}"
        )

        # Unified Ring Buffer: all slots cycle for prefill
        # Prefill: use ALL slots as ring buffer (slot[chunk_idx % N])
        # Decode: slot[0] as decode_slot, slots[1:] for loading previous chunks
        self.num_ring_slots = num_gpu_blocks


        # Decode phase uses slot[0] for writing new token's KV
        self.decode_slot = 0
        # Decode phase uses slots[1:] for loading previous chunks from CPU
        self.decode_load_slots = list(range(1, num_gpu_blocks))


        self.num_gpu_slots = num_gpu_blocks  # alias

        logger.info(f"Unified Ring Buffer: {self.num_ring_slots} slots total")
        logger.info(f"  Prefill: all slots as ring buffer [0..{num_gpu_blocks - 1}]")
        logger.info(
            f"  Decode: slot[0] as decode_slot, slots[1..{num_gpu_blocks - 1}] for loading"
        )

        # ========== Fixed-address GPU KV cache ==========
        # ========== Layout configuration ==========
        self.sparse_policy = sparse_policy
        self.is_head_first = (
            sparse_policy is not None 
            and getattr(sparse_policy.__class__, '__name__', '') == "COMPASSPolicy"
        )
        if self.is_head_first:
            logger.info("Using Head-First layout: [..., Heads, Tokens, Head_Dim] for COMPASS")
            self._gpu_shape = (num_gpu_blocks, num_kv_heads, block_size, head_dim)
            self._layer_shape = (num_layers, num_kv_heads, block_size, head_dim)
            self._cpu_shape = (num_layers, num_cpu_blocks, num_kv_heads, block_size, head_dim)
            self._staging_shape = (num_kv_heads, block_size, head_dim)
        else:
            self._gpu_shape = (num_gpu_blocks, block_size, num_kv_heads, head_dim)
            self._layer_shape = (num_layers, block_size, num_kv_heads, head_dim)
            self._cpu_shape = (num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim)
            self._staging_shape = (block_size, num_kv_heads, head_dim)

        # ========== Fixed-address GPU KV cache ==========
        self.k_cache_gpu = torch.zeros(*self._gpu_shape, dtype=dtype, device="cuda")
        self.v_cache_gpu = torch.zeros(*self._gpu_shape, dtype=dtype, device="cuda")

        # ========== Per-layer decode buffer ==========
        self.decode_k_buffer = torch.zeros(*self._layer_shape, dtype=dtype, device="cuda")
        self.decode_v_buffer = torch.zeros(*self._layer_shape, dtype=dtype, device="cuda")
        decode_buf_mb = (
            2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        )
        logger.info(f"  Per-layer decode buffer: {decode_buf_mb:.1f} MB")

        # ========== Per-layer prefill buffer for async offload ==========
        self.prefill_k_buffer = torch.zeros(*self._layer_shape, dtype=dtype, device="cuda")
        self.prefill_v_buffer = torch.zeros(*self._layer_shape, dtype=dtype, device="cuda")
        prefill_buf_mb = (
            2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        )
        logger.info(f"  Per-layer prefill buffer: {prefill_buf_mb:.1f} MB")

        # Per-layer offload events for async prefill offload
        self.prefill_offload_events = [torch.cuda.Event() for _ in range(num_layers)]
        # Per-layer transfer streams for parallel offloads
        self.prefill_offload_streams = [torch.cuda.Stream() for _ in range(num_layers)]

        # ========== Fixed-address CPU KV cache (pinned memory) ==========
        self.k_cache_cpu = torch.zeros(*self._cpu_shape, dtype=dtype, device="cpu", pin_memory=True)
        self.v_cache_cpu = torch.zeros(*self._cpu_shape, dtype=dtype, device="cpu", pin_memory=True)

        # Log memory allocation
        gpu_mem_mb = self.gpu_memory_bytes() / (1024 * 1024)
        cpu_mem_mb = self.cpu_memory_bytes() / (1024 * 1024)
        logger.info(
            f"  GPU memory: {gpu_mem_mb:.1f} MB, CPU memory: {cpu_mem_mb:.1f} MB"
        )

        # ========== Staging buffers for sub-block gather ==========
        self.num_staging_buffers = 2 * num_gpu_blocks
        self.staging_k_cpu = [
            torch.zeros(*self._staging_shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(self.num_staging_buffers)
        ]
        self.staging_v_cpu = [
            torch.zeros(*self._staging_shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(self.num_staging_buffers)
        ]
        per_buf_mb = (
            2 * block_size * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        )
        logger.info(
            f"  Sub-block staging buffers: {self.num_staging_buffers} x {per_buf_mb:.1f} MB "
            f"= {self.num_staging_buffers * per_buf_mb:.1f} MB total (Pinned CPU)"
        )

        # ========== V2 Jagged Buffers ==========
        # For V2 jagged attention, gather dynamically sized subblocks across ALL heads.
        self.num_jagged_slots = self.num_ring_slots
        max_jagged_tokens = num_cpu_blocks * block_size * num_kv_heads
        
        self.jagged_staging_k = [torch.zeros((max_jagged_tokens, head_dim), dtype=dtype, device="cpu", pin_memory=True) for _ in range(self.num_jagged_slots)]
        self.jagged_staging_v = [torch.zeros((max_jagged_tokens, head_dim), dtype=dtype, device="cpu", pin_memory=True) for _ in range(self.num_jagged_slots)]
        self.jagged_k_gpu = [torch.zeros((max_jagged_tokens, head_dim), dtype=dtype, device="cuda") for _ in range(self.num_jagged_slots)]
        self.jagged_v_gpu = [torch.zeros((max_jagged_tokens, head_dim), dtype=dtype, device="cuda") for _ in range(self.num_jagged_slots)]
        
        # Events to track when the GPU is done computing with a specific jagged slot
        self.jagged_compute_done = [torch.cuda.Event() for _ in range(self.num_jagged_slots)]
        for slot_idx in range(self.num_jagged_slots):
            self.jagged_compute_done[slot_idx].record()
        
        logger.info(f"  V2 Jagged buffers allocated ({self.num_jagged_slots} slots, {max_jagged_tokens} max tokens)")

        # ========== Compute stream for async operations ==========
        # IMPORTANT: Create a dedicated compute stream (not default stream!)
        # Default stream has implicit synchronization with other streams,
        # which prevents overlap between transfer and compute.
        self.compute_stream = torch.cuda.Stream()

        # ========== Per-slot transfer streams for parallel H2D ==========
        # Each slot has its own stream to enable parallel transfers
        # This allows multiple slots to load simultaneously
        self.slot_transfer_streams = [
            torch.cuda.Stream() for _ in range(self.num_ring_slots)
        ]
        logger.info(f"  Created {self.num_ring_slots} per-slot transfer streams")

        # ========== Ring Buffer dedicated stream and events ==========
        self.transfer_stream_main = (
            torch.cuda.Stream()
        )  # Main transfer stream (for legacy/batch ops)


        # ========== Per-slot events for ring buffer ==========
        # Since GPU cache has no layer dimension and layers execute sequentially,
        # we only need per-slot events (not per-slot per-layer).
        # ring_slot_ready[slot_idx] = CUDA Event for H2D completion
        # ring_slot_offload_done[slot_idx] = CUDA Event for D2H completion
        self.ring_slot_ready = [torch.cuda.Event() for _ in range(self.num_ring_slots)]
        self.ring_slot_offload_done = [
            torch.cuda.Event() for _ in range(self.num_ring_slots)
        ]

        # ========== Per-slot compute_done events for async pipeline ==========
        # ring_slot_compute_done[slot_idx] = CUDA Event for compute completion
        # This ensures we don't overwrite data before it's been read by attention
        self.ring_slot_compute_done = [
            torch.cuda.Event() for _ in range(self.num_ring_slots)
        ]

        # ========== Per-staging-buffer read_done events for async pipeline ==========
        # staging_read_done[staging_idx] = CUDA Event for when GPU finishes reading from staging buffer
        # This prevents CPU head-of-line blocking when reusing staging buffers
        self.staging_read_done = [
            torch.cuda.Event() for _ in range(self.num_staging_buffers)
        ]

        # Initialize all compute_done events (record them once)
        # This prevents undefined behavior on first load_to_slot_layer call
        for slot_idx in range(self.num_ring_slots):
            self.ring_slot_compute_done[slot_idx].record()
        for staging_idx in range(self.num_staging_buffers):
            self.staging_read_done[staging_idx].record()
        # torch.cuda.synchronize()  # Ensure all events are recorded

        # ========== Event tracking for async transfers ==========
        self.pending_events: Dict[Tuple[int, int], torch.cuda.Event] = {}


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
        k, v = self.k_cache_gpu, self.v_cache_gpu
        if self.is_head_first:
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
        return k, v

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
            self.k_cache_gpu.numel() * self.k_cache_gpu.element_size()
            + self.v_cache_gpu.numel() * self.v_cache_gpu.element_size()
        )

    def cpu_memory_bytes(self) -> int:
        """Total CPU memory used by KV caches."""
        return (
            self.k_cache_cpu.numel() * self.k_cache_cpu.element_size()
            + self.v_cache_cpu.numel() * self.v_cache_cpu.element_size()
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


    # ----- Decode: slot management -----


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
        self,
        slot_idx: int,
        layer_id: int,
        cpu_block_id: int,
        chunk_idx: int = -1,
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
        logger.debug(
            f"Ring load: layer={layer_id}, CPU[{cpu_block_id}] -> GPU slot[{slot_idx}]"
        )

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
        self,
        slot_idx: int,
        layer_id: int,
        cpu_block_id: int,
        chunk_idx: int = -1,
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
        logger.debug(
            f"Ring load K-only: layer={layer_id}, CPU[{cpu_block_id}] -> GPU slot[{slot_idx}]"
        )

        stream = self.slot_transfer_streams[slot_idx]

        if chunk_idx >= 0:
            nvtx_label = f"H2D K-only: L{layer_id} Chunk{chunk_idx} CPU[{cpu_block_id}]->Slot[{slot_idx}]"
        else:
            nvtx_label = (
                f"H2D K-only: L{layer_id} CPU[{cpu_block_id}]->Slot[{slot_idx}]"
            )

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
        logger.debug(
            f"Ring offload: GPU slot[{slot_idx}] -> CPU[layer={layer_id}, block={cpu_block_id}]"
        )

        # Collect metadata BEFORE offload (while k_cache is still on GPU)
        valid_tokens = num_valid_tokens if num_valid_tokens > 0 else self.block_size
        k_cache = self.k_cache_gpu[slot_idx]

        if self.sparse_policy is not None:
            if is_prefill:
                self.sparse_policy.on_prefill_offload(
                    cpu_block_id, layer_id, k_cache, valid_tokens
                )
            else:
                self.sparse_policy.on_decode_offload(
                    cpu_block_id, layer_id, k_cache, valid_tokens
                )

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
        k = self.k_cache_gpu[slot_idx]
        v = self.v_cache_gpu[slot_idx]
        if self.is_head_first:
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()
        return k.unsqueeze(0), v.unsqueeze(0)


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


    # ========== Per-layer Prefill Buffer Methods ==========
    # These methods enable async offload during chunked prefill by using
    # per-layer buffers instead of shared GPU slots.


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
        if self.is_head_first:
            k = self.prefill_k_buffer[layer_id, :, :num_tokens].transpose(0, 1).contiguous()
            v = self.prefill_v_buffer[layer_id, :, :num_tokens].transpose(0, 1).contiguous()
        else:
            k = self.prefill_k_buffer[layer_id, :num_tokens]
            v = self.prefill_v_buffer[layer_id, :num_tokens]
        return k.unsqueeze(0), v.unsqueeze(0)

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
        if self.is_head_first:
            self.prefill_k_buffer[layer_id, :, :num_tokens].copy_(k.transpose(0, 1))
            self.prefill_v_buffer[layer_id, :, :num_tokens].copy_(v.transpose(0, 1))
        else:
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
        torch.cuda.nvtx.range_push(
            f"D2D: L{layer_id} Pos{pos_in_block} WriteDecodeBuffer"
        )
        if getattr(self, 'is_head_first', False):
            self.decode_k_buffer[layer_id, :, pos_in_block].copy_(k)
            self.decode_v_buffer[layer_id, :, pos_in_block].copy_(v)
        else:
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
            self.sparse_policy.on_prefill_offload(
                cpu_block_id, layer_id, k_cache, valid_tokens
            )

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


    # ========== Sub-Block Gather & Load Methods ==========


    def load_staging_to_slot(
        self,
        slot_idx: int,
        num_tokens: int,
        layer_id: int = -1,
        is_prefill: bool = True,
        staging_idx: int = 0,
        head_idx: int = -1,
    ) -> None:
        """
        Async H2D transfer from staging buffer to a GPU slot.

        Uses the per-slot transfer stream and records the ring_slot_ready event,
        so callers can use wait_slot_layer() just like regular load_to_slot_layer.

        Args:
            slot_idx: Target GPU slot index.
            num_tokens: Number of valid tokens in staging buffer.
            layer_id: Layer index for NVTX labeling (-1 = not specified).
            is_prefill: True if in prefill phase.
            staging_idx: Index of the staging buffer to read from.
            head_idx: Specific head to transfer (only used if is_head_first).
        """
        stream = self.slot_transfer_streams[slot_idx]
        staging_k = self.staging_k_cpu[staging_idx]
        staging_v = self.staging_v_cpu[staging_idx]

        nvtx_label = f"H2D Gather: L{layer_id} stg{staging_idx} {num_tokens}tok->Slot[{slot_idx}]"
        nvtx.push_range(message=nvtx_label, color="magenta")
        with torch.cuda.stream(stream):
            # Wait for previous compute on this slot to complete
            stream.wait_event(self.ring_slot_compute_done[slot_idx])
            stream.wait_event(self.ring_slot_offload_done[slot_idx])

            # H2D: only transfer num_tokens worth of data (not full block)
            if self.is_head_first and head_idx != -1:
                self.k_cache_gpu[slot_idx, head_idx, :num_tokens].copy_(
                    staging_k[head_idx, :num_tokens], non_blocking=True
                )
                self.v_cache_gpu[slot_idx, head_idx, :num_tokens].copy_(
                    staging_v[head_idx, :num_tokens], non_blocking=True
                )
            else:
                self.k_cache_gpu[slot_idx, :num_tokens].copy_(
                    staging_k[:num_tokens], non_blocking=True
                )
                self.v_cache_gpu[slot_idx, :num_tokens].copy_(
                    staging_v[:num_tokens], non_blocking=True
                )
                
            # Record that the staging buffer has been read by the GPU
            self.staging_read_done[staging_idx].record(stream)
            
            self.ring_slot_ready[slot_idx].record(stream)
        nvtx.pop_range()

        # Record H2D transfer: only the compacted amount
        if self.is_head_first and head_idx != -1:
            transfer_bytes = 2 * num_tokens * self.head_dim * self.dtype_size
        else:
            transfer_bytes = 2 * num_tokens * self.num_kv_heads * self.head_dim * self.dtype_size
        MemoryObserver.record_h2d(transfer_bytes, is_prefill=is_prefill)

    def gather_subblocks_per_head(
        self,
        layer_id: int,
        per_head_selections: "List[List]",  # [H][(bid, [si...])]
        sub_block_size: int = 128,
        head_offset: int = 0,
        staging_idx: int = 0,
    ) -> "Tuple[int, List[int]]":
        """
        Per-head gather into structured staging buffer [block_size, kv_heads, head_dim].

        Each head's selected sub-blocks are written into the staging buffer at
        the correct head dimension, packed contiguously along the token axis.
        This preserves the [tokens, heads, head_dim] interleaved layout that matches
        both the staging buffer and GPU slot shapes.

        E.g., for head h with 3 sub-blocks of 128 tokens each:
            staging_k[0:128, h, :] = sub-block 0
            staging_k[128:256, h, :] = sub-block 1
            staging_k[256:384, h, :] = sub-block 2

        Args:
            layer_id: Layer index for CPU cache indexing.
            per_head_selections: [H] lists of (cpu_block_id, sub_block_indices).
            sub_block_size: Tokens per sub-block (default 128).
            head_offset: Starting KV head index (for single-head calls, pass the
                actual head index so the correct head's data is read from CPU cache).
            staging_idx: Index of the staging buffer to write to.

        Returns:
            max_tokens: Maximum tokens across all heads (for H2D transfer sizing).
            per_head_tokens: [H] list of token counts per head.
        """
        staging_k = self.staging_k_cpu[staging_idx]
        staging_v = self.staging_v_cpu[staging_idx]
        per_head_tokens = []

        for h_local, head_sels in enumerate(per_head_selections):
            h_actual = h_local + head_offset
            token_offset = 0
            for cpu_block_id, sub_indices in head_sels:
                # Coalesce contiguous sub-indices into runs to minimize copy_() calls.
                # E.g., [0,1,2,5,6] → [(0,3), (5,2)] → 2 copies instead of 5.
                i = 0
                while i < len(sub_indices):
                    run_start = sub_indices[i]
                    run_len = 1
                    while (i + run_len < len(sub_indices)
                           and sub_indices[i + run_len] == run_start + run_len):
                        run_len += 1

                    n_tok = run_len * sub_block_size
                    src_start = run_start * sub_block_size
                    src_end = src_start + n_tok
                    dst_start = token_offset
                    dst_end = token_offset + n_tok
                    if self.is_head_first:
                        staging_k[h_actual, dst_start:dst_end, :].copy_(
                            self.k_cache_cpu[layer_id, cpu_block_id, h_actual, src_start:src_end, :]
                        )
                        staging_v[h_actual, dst_start:dst_end, :].copy_(
                            self.v_cache_cpu[layer_id, cpu_block_id, h_actual, src_start:src_end, :]
                        )
                    else:
                        staging_k[dst_start:dst_end, h_actual, :].copy_(
                            self.k_cache_cpu[layer_id, cpu_block_id, src_start:src_end, h_actual, :]
                        )
                        staging_v[dst_start:dst_end, h_actual, :].copy_(
                            self.v_cache_cpu[layer_id, cpu_block_id, src_start:src_end, h_actual, :]
                        )
                    token_offset += n_tok
                    i += run_len
            per_head_tokens.append(token_offset)

        max_tokens = max(per_head_tokens) if per_head_tokens else 0
        return max_tokens, per_head_tokens

    def gather_packed_subblocks_all_heads(
        self,
        layer_id: int,
        per_head_selections: "List[List]",  # [H_KV][(bid, [si...])]
        sub_block_size: int = 128,
        slot_idx: int = 0,
    ) -> "Tuple[int, torch.Tensor]":
        """
        (V2) Gathers all subblocks for all heads into a pure 1D flattened array
        directly onto the globally allocated jagged_staging_cpu.
        Returns:
            total_tokens: int (total valid tokens scattered across all heads)
            kv_indptr_tensor: torch.Tensor [H_KV + 1] on CPU representing head boundaries
        """
        k_staging = self.jagged_staging_k[slot_idx]
        v_staging = self.jagged_staging_v[slot_idx]
        
        kv_indptr = [0]
        token_offset = 0

        for h_actual, head_sels in enumerate(per_head_selections):
            for cpu_block_id, sub_indices in head_sels:
                i = 0
                while i < len(sub_indices):
                    run_start = sub_indices[i]
                    run_len = 1
                    while (i + run_len < len(sub_indices)
                           and sub_indices[i + run_len] == run_start + run_len):
                        run_len += 1

                    n_tok = run_len * sub_block_size
                    src_start = run_start * sub_block_size
                    src_end = src_start + n_tok
                    dst_start = token_offset
                    dst_end = token_offset + n_tok
                    
                    if self.is_head_first:
                        k_staging[dst_start:dst_end, :].copy_(
                            self.k_cache_cpu[layer_id, cpu_block_id, h_actual, src_start:src_end, :]
                        )
                        v_staging[dst_start:dst_end, :].copy_(
                            self.v_cache_cpu[layer_id, cpu_block_id, h_actual, src_start:src_end, :]
                        )
                    else:
                        k_staging[dst_start:dst_end, :].copy_(
                            self.k_cache_cpu[layer_id, cpu_block_id, src_start:src_end, h_actual, :]
                        )
                        v_staging[dst_start:dst_end, :].copy_(
                            self.v_cache_cpu[layer_id, cpu_block_id, src_start:src_end, h_actual, :]
                        )
                    
                    token_offset += n_tok
                    i += run_len
            kv_indptr.append(token_offset)
            
        kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device="cpu")
        return token_offset, kv_indptr_tensor

    def load_packed_staging_to_jagged_gpu(
        self,
        total_tokens: int,
        stream: torch.cuda.Stream,
        is_prefill: bool = True,
        slot_idx: int = 0,
    ):
        """
        (V2) Loads the 1D flattened packed array from the globally sizing CPU staging 
        to the globally sized GPU staging buffer.
        """
        gpu_k = self.jagged_k_gpu[slot_idx]
        gpu_v = self.jagged_v_gpu[slot_idx]
        staging_k = self.jagged_staging_k[slot_idx]
        staging_v = self.jagged_staging_v[slot_idx]

        with torch.cuda.stream(stream):
            # 1D exact flat copy
            gpu_k[:total_tokens, :].copy_(staging_k[:total_tokens, :], non_blocking=True)
            gpu_v[:total_tokens, :].copy_(staging_v[:total_tokens, :], non_blocking=True)

        from nanovllm.utils.memory_observer import MemoryObserver
        transfer_bytes = 2 * total_tokens * self.head_dim * gpu_k.element_size()
        MemoryObserver.record_h2d(transfer_bytes, is_prefill=is_prefill)

