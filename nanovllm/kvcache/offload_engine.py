"""
High-performance CPU-GPU KV cache transfer engine.

Key design principles for CUDA Graph compatibility:
1. All tensor addresses are fixed at initialization
2. Only index tensor contents change between graph replays
3. Supports both async transfer (for prefill) and graph-based transfer (for decode)
"""

import torch
import torch.cuda.nvtx
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from nanovllm.kvcache.kernels import gathered_copy_kv
from nanovllm.comm import memcpy_2d_async
from nanovllm.utils.logger import get_logger

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
    - GPU cache: [num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]
    - CPU cache: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim] (pinned)
    - Gather indices: [num_layers, num_gpu_blocks] (fixed address, variable content)

    CUDA Graph compatibility:
    - gathered_h2d_layer() can be captured into CUDA graphs
    - update_gather_indices() is called outside graphs to prepare indices
    - All tensor addresses remain fixed across graph replays
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

        # ========== Fixed-address CPU KV cache (pinned memory) ==========
        self.k_cache_cpu = torch.zeros(
            num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cpu", pin_memory=True
        )
        self.v_cache_cpu = torch.zeros(
            num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cpu", pin_memory=True
        )

        # ========== Fixed-address gather indices (content is variable) ==========
        # gather_indices[layer][i] = CPU block id to copy to GPU slot i
        # -1 means no-op (skip this slot)
        self.gather_indices_cpu = torch.empty(
            num_layers, num_gpu_blocks,
            dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.gather_indices_cpu.fill_(-1)
        self.gather_indices_gpu = torch.full(
            (num_layers, num_gpu_blocks), -1,
            dtype=torch.int64, device="cuda"
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
        torch.cuda.synchronize()  # Ensure all events are recorded

        # ========== Event tracking for async transfers ==========
        self.pending_events: Dict[Tuple[int, int], torch.cuda.Event] = {}

        # ========== Debug hook mode ==========
        self._debug_mode = False
        self._debug_hooks: List = []  # External hooks for debug events

    def _get_next_stream(self) -> torch.cuda.Stream:
        """Round-robin stream selection for parallel transfers."""
        stream = self.transfer_streams[self._stream_idx]
        self._stream_idx = (self._stream_idx + 1) % len(self.transfer_streams)
        return stream

    # ========== CUDA Graph compatible methods ==========
    # NOTE: These methods need to be updated for the new GPU cache architecture.
    # GPU cache no longer has layer dimension, so gathered copy semantics change.
    # For now, these are kept for reference but should not be used without updating.

    def gathered_h2d_layer(self, layer_id: int) -> None:
        """
        Execute gathered H2D copy for a single layer.

        WARNING: This method needs updating for new GPU cache architecture.
        GPU cache no longer has layer dimension.
        """
        # GPU cache has no layer dimension - use flat indexing
        # Source is CPU[layer_id], dest is GPU (shared across layers)
        gathered_copy_kv(
            k_src=self.k_cache_cpu[layer_id],
            v_src=self.v_cache_cpu[layer_id],
            k_dst=self.k_cache_gpu,  # No layer indexing
            v_dst=self.v_cache_gpu,  # No layer indexing
            indices=self.gather_indices_gpu[layer_id],
        )

    def gathered_h2d_all_layers(self) -> None:
        """
        Execute gathered H2D copy for all layers.

        WARNING: In new architecture, GPU slots are shared across layers.
        This method would overwrite slots multiple times. Not recommended.
        """
        for layer_id in range(self.num_layers):
            self.gathered_h2d_layer(layer_id)

    def update_gather_indices(
        self,
        layer_id: int,
        mappings: List[Tuple[int, int]],
    ) -> None:
        """
        Update gather indices for a layer (call OUTSIDE CUDA graph).

        Args:
            layer_id: Layer index
            mappings: List of (cpu_block_id, gpu_slot) tuples
                     Only these slots will be updated; others keep their values
        """
        for cpu_block_id, gpu_slot in mappings:
            self.gather_indices_cpu[layer_id, gpu_slot] = cpu_block_id

        # Async copy to GPU
        self.gather_indices_gpu[layer_id].copy_(
            self.gather_indices_cpu[layer_id],
            non_blocking=True
        )

    def update_gather_indices_all_layers(
        self,
        mappings_per_layer: List[List[Tuple[int, int]]],
    ) -> None:
        """
        Update gather indices for all layers.

        Args:
            mappings_per_layer: mappings_per_layer[layer_id] = [(cpu_block_id, gpu_slot), ...]
        """
        for layer_id, mappings in enumerate(mappings_per_layer):
            for cpu_block_id, gpu_slot in mappings:
                self.gather_indices_cpu[layer_id, gpu_slot] = cpu_block_id

        # Batch copy all layers
        self.gather_indices_gpu.copy_(self.gather_indices_cpu, non_blocking=True)

    def clear_gather_indices(self, layer_id: Optional[int] = None) -> None:
        """
        Clear gather indices (set all to -1, meaning no-op).

        Args:
            layer_id: If provided, clear only this layer; otherwise clear all
        """
        if layer_id is not None:
            self.gather_indices_cpu[layer_id].fill_(-1)
            self.gather_indices_gpu[layer_id].fill_(-1)
        else:
            self.gather_indices_cpu.fill_(-1)
            self.gather_indices_gpu.fill_(-1)

    # ========== Async transfer methods (for prefill, outside CUDA graph) ==========

    def prefetch_block_async(
        self,
        layer_id: int,
        cpu_block_id: int,
        gpu_block_id: int,
    ) -> torch.cuda.Event:
        """
        Async prefetch a single block from CPU to GPU.

        GPU cache has no layer dimension - layer_id is for CPU cache indexing.

        Args:
            layer_id: Layer index (for CPU cache)
            cpu_block_id: Source block in CPU cache
            gpu_block_id: Destination slot in GPU cache

        Returns:
            CUDA event that signals completion
        """
        stream = self._get_next_stream()
        event = torch.cuda.Event()

        logger.debug(f"H2D prefetch: layer={layer_id}, CPU[{cpu_block_id}] -> GPU[{gpu_block_id}]")

        with torch.cuda.stream(stream):
            # GPU: no layer dimension, CPU: has layer dimension
            self.k_cache_gpu[gpu_block_id].copy_(
                self.k_cache_cpu[layer_id, cpu_block_id],
                non_blocking=True
            )
            self.v_cache_gpu[gpu_block_id].copy_(
                self.v_cache_cpu[layer_id, cpu_block_id],
                non_blocking=True
            )
            event.record()

        self.pending_events[(layer_id, gpu_block_id)] = event
        return event

    def prefetch_blocks_batch_async(
        self,
        transfers: List[Tuple[int, int, int]],  # [(layer_id, cpu_block_id, gpu_block_id), ...]
    ) -> List[torch.cuda.Event]:
        """
        Batch async prefetch multiple blocks.

        Args:
            transfers: List of (layer_id, cpu_block_id, gpu_block_id) tuples

        Returns:
            List of CUDA events for each transfer
        """
        events = []
        for layer_id, cpu_block_id, gpu_block_id in transfers:
            event = self.prefetch_block_async(layer_id, cpu_block_id, gpu_block_id)
            events.append(event)
        return events

    def offload_block_async(
        self,
        layer_id: int,
        gpu_block_id: int,
        cpu_block_id: int,
    ) -> torch.cuda.Event:
        """
        Async offload a block from GPU to CPU.

        GPU cache has no layer dimension - layer_id is for CPU cache indexing.

        Args:
            layer_id: Layer index (for CPU cache)
            gpu_block_id: Source slot in GPU cache
            cpu_block_id: Destination block in CPU cache

        Returns:
            CUDA event that signals completion
        """
        stream = self._get_next_stream()
        event = torch.cuda.Event()

        logger.debug(f"D2H offload: layer={layer_id}, GPU[{gpu_block_id}] -> CPU[{cpu_block_id}]")

        with torch.cuda.stream(stream):
            # Wait for any compute using this block
            stream.wait_stream(self.compute_stream)

            # GPU: no layer dimension, CPU: has layer dimension
            self.k_cache_cpu[layer_id, cpu_block_id].copy_(
                self.k_cache_gpu[gpu_block_id],
                non_blocking=True
            )
            self.v_cache_cpu[layer_id, cpu_block_id].copy_(
                self.v_cache_gpu[gpu_block_id],
                non_blocking=True
            )
            event.record()

        return event

    def offload_blocks_batch_async(
        self,
        transfers: List[Tuple[int, int, int]],  # [(layer_id, gpu_block_id, cpu_block_id), ...]
    ) -> List[torch.cuda.Event]:
        """
        Batch async offload multiple blocks.

        Args:
            transfers: List of (layer_id, gpu_block_id, cpu_block_id) tuples

        Returns:
            List of CUDA events
        """
        events = []
        for layer_id, gpu_block_id, cpu_block_id in transfers:
            event = self.offload_block_async(layer_id, gpu_block_id, cpu_block_id)
            events.append(event)
        return events

    # ========== Chunked Decode: Load CPU blocks to GPU slots ==========

    def load_cpu_blocks_to_gpu_slots(
        self,
        layer_id: int,
        cpu_block_ids: List[int],
        gpu_slot_ids: List[int],
    ) -> None:
        """
        Load CPU blocks to specific GPU slots for chunked decode.

        GPU cache has no layer dimension - layer_id is for CPU cache indexing.

        Args:
            layer_id: Layer index (for CPU cache)
            cpu_block_ids: List of CPU block IDs to load
            gpu_slot_ids: List of GPU slot IDs to load into (must be same length)
        """
        assert len(cpu_block_ids) == len(gpu_slot_ids)

        if cpu_block_ids:
            logger.debug(f"H2D chunked load: layer={layer_id}, CPU{cpu_block_ids} -> GPU{gpu_slot_ids}")

        stream = self._get_next_stream()

        with torch.cuda.stream(stream):
            for cpu_block_id, gpu_slot in zip(cpu_block_ids, gpu_slot_ids):
                # GPU: no layer dimension, CPU: has layer dimension
                self.k_cache_gpu[gpu_slot].copy_(
                    self.k_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )
                self.v_cache_gpu[gpu_slot].copy_(
                    self.v_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )

        # Wait for transfer to complete
        stream.synchronize()

    def load_cpu_blocks_to_gpu_slots_async(
        self,
        layer_id: int,
        cpu_block_ids: List[int],
        gpu_slot_ids: List[int],
    ) -> torch.cuda.Event:
        """
        Async version: Load CPU blocks to GPU slots.

        GPU cache has no layer dimension - layer_id is for CPU cache indexing.

        Args:
            layer_id: Layer index (for CPU cache)
            cpu_block_ids: List of CPU block IDs to load
            gpu_slot_ids: List of GPU slot IDs to load into

        Returns:
            CUDA event to wait on
        """
        assert len(cpu_block_ids) == len(gpu_slot_ids)

        if cpu_block_ids:
            logger.debug(f"H2D chunked load async: layer={layer_id}, CPU{cpu_block_ids} -> GPU{gpu_slot_ids}")

        stream = self._get_next_stream()
        event = torch.cuda.Event()

        with torch.cuda.stream(stream):
            for cpu_block_id, gpu_slot in zip(cpu_block_ids, gpu_slot_ids):
                # GPU: no layer dimension, CPU: has layer dimension
                self.k_cache_gpu[gpu_slot].copy_(
                    self.k_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )
                self.v_cache_gpu[gpu_slot].copy_(
                    self.v_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )
            event.record()

        return event

    # NOTE: load_cpu_blocks_to_gpu_slots_all_layers removed - GPU cache no longer has
    # layer dimension. Each GPU slot holds data for ONE layer at a time.

    # ========== Synchronization methods ==========

    def wait_for_block(self, layer_id: int, gpu_block_id: int) -> None:
        """Wait for a specific block's transfer to complete."""
        key = (layer_id, gpu_block_id)
        if key in self.pending_events:
            self.pending_events[key].synchronize()
            del self.pending_events[key]

    def wait_all_transfers(self) -> None:
        """Wait for all pending transfers to complete."""
        for stream in self.transfer_streams:
            stream.synchronize()
        self.pending_events.clear()

    def sync_indices(self) -> None:
        """Synchronize to ensure all index updates are complete."""
        torch.cuda.default_stream().synchronize()

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
        # GPU cache is shared across all layers (no layer dimension)
        return self.k_cache_gpu, self.v_cache_gpu

    def get_all_gpu_cache(self) -> Tuple[Tensor, Tensor]:
        """
        Get full GPU K/V cache tensors.

        NOTE: GPU cache has no layer dimension in the new architecture.

        Returns:
            (k_cache, v_cache) tensors
            Shape: [num_gpu_blocks, block_size, kv_heads, head_dim]
        """
        return self.k_cache_gpu, self.v_cache_gpu

    def get_cpu_block(
        self,
        layer_id: int,
        cpu_block_id: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get a specific CPU block's K/V cache.

        Returns:
            (k_cache, v_cache) for the block
            Shape: [block_size, kv_heads, head_dim]
        """
        return (
            self.k_cache_cpu[layer_id, cpu_block_id],
            self.v_cache_cpu[layer_id, cpu_block_id],
        )

    # ========== Memory info ==========

    def gpu_memory_bytes(self) -> int:
        """Total GPU memory used by KV caches."""
        return (
            self.k_cache_gpu.numel() * self.k_cache_gpu.element_size() +
            self.v_cache_gpu.numel() * self.v_cache_gpu.element_size() +
            self.gather_indices_gpu.numel() * self.gather_indices_gpu.element_size()
        )

    def cpu_memory_bytes(self) -> int:
        """Total CPU memory used by KV caches."""
        return (
            self.k_cache_cpu.numel() * self.k_cache_cpu.element_size() +
            self.v_cache_cpu.numel() * self.v_cache_cpu.element_size() +
            self.gather_indices_cpu.numel() * self.gather_indices_cpu.element_size()
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

    def load_to_slot_layer(self, slot_idx: int, layer_id: int, cpu_block_id: int) -> None:
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
        """
        logger.debug(f"Ring load: layer={layer_id}, CPU[{cpu_block_id}] -> GPU slot[{slot_idx}]")

        # Use per-slot stream for parallel transfers across different slots
        stream = self.slot_transfer_streams[slot_idx]

        torch.cuda.nvtx.range_push(f"H2D: L{layer_id} CPU[{cpu_block_id}]->Slot[{slot_idx}]")
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
        torch.cuda.nvtx.range_pop()

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

    def offload_slot_layer_to_cpu(self, slot_idx: int, layer_id: int, cpu_block_id: int) -> None:
        """
        Async offload a ring buffer slot to CPU for one layer.

        GPU cache has no layer dimension, so we copy from GPU slot to the
        specific layer in CPU cache.

        Args:
            slot_idx: Source GPU slot index
            layer_id: Target layer in CPU cache
            cpu_block_id: Target CPU block ID
        """
        logger.debug(f"Ring offload: GPU slot[{slot_idx}] -> CPU[layer={layer_id}, block={cpu_block_id}]")

        torch.cuda.nvtx.range_push(f"D2H: Slot[{slot_idx}]->CPU[L{layer_id},B{cpu_block_id}]")
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
        torch.cuda.nvtx.range_pop()

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

    # ----- Legacy compatibility methods (for decode double-buffering) -----
    # NOTE: GPU cache has no layer dimension. Layer ID is used for CPU cache indexing only.

    def load_to_compute_layer(self, layer_id: int, cpu_block_ids: List[int]) -> None:
        """
        Legacy: Load CPU blocks to decode_load_slots for decode double-buffering.

        Uses first half of decode_load_slots as 'compute' region.
        GPU cache has no layer dimension - layer_id is for CPU cache indexing.
        """
        if not cpu_block_ids:
            return

        half = max(1, len(self.decode_load_slots) // 2)
        slots = self.decode_load_slots[:half]
        num_to_load = min(len(cpu_block_ids), len(slots))

        with torch.cuda.stream(self.transfer_stream_main):
            for i in range(num_to_load):
                cpu_id = cpu_block_ids[i]
                gpu_slot = slots[i]
                # GPU: no layer dimension, CPU: has layer dimension
                self.k_cache_gpu[gpu_slot].copy_(
                    self.k_cache_cpu[layer_id, cpu_id], non_blocking=True
                )
                self.v_cache_gpu[gpu_slot].copy_(
                    self.v_cache_cpu[layer_id, cpu_id], non_blocking=True
                )
            if num_to_load > 0:
                self.ring_slot_ready[slots[0]].record(self.transfer_stream_main)

    def wait_compute_layer(self) -> None:
        """Legacy: Wait for 'compute' region loading."""
        if self.decode_load_slots:
            self.wait_slot_layer(self.decode_load_slots[0])

    def load_to_prefetch_layer(self, layer_id: int, cpu_block_ids: List[int]) -> None:
        """
        Legacy: Load CPU blocks to decode_load_slots for decode double-buffering.

        Uses second half of decode_load_slots as 'prefetch' region.
        GPU cache has no layer dimension - layer_id is for CPU cache indexing.
        """
        if not cpu_block_ids:
            return

        half = max(1, len(self.decode_load_slots) // 2)
        slots = self.decode_load_slots[half:]
        if not slots:
            slots = self.decode_load_slots  # Fallback if only 1-2 slots
        num_to_load = min(len(cpu_block_ids), len(slots))

        with torch.cuda.stream(self.transfer_stream_main):
            for i in range(num_to_load):
                cpu_id = cpu_block_ids[i]
                gpu_slot = slots[i]
                # GPU: no layer dimension, CPU: has layer dimension
                self.k_cache_gpu[gpu_slot].copy_(
                    self.k_cache_cpu[layer_id, cpu_id], non_blocking=True
                )
                self.v_cache_gpu[gpu_slot].copy_(
                    self.v_cache_cpu[layer_id, cpu_id], non_blocking=True
                )
            if num_to_load > 0:
                self.ring_slot_ready[slots[0]].record(self.transfer_stream_main)

    def wait_prefetch_layer(self) -> None:
        """Legacy: Wait for 'prefetch' region loading."""
        half = max(1, len(self.decode_load_slots) // 2)
        slots = self.decode_load_slots[half:]
        if slots:
            self.wait_slot_layer(slots[0])
        elif self.decode_load_slots:
            self.wait_slot_layer(self.decode_load_slots[0])

    def get_kv_for_compute(
        self,
        num_blocks: int,
    ) -> Tuple[Tensor, Tensor]:
        """Legacy: Get KV from 'compute' region (first half of decode_load_slots)."""
        half = max(1, len(self.decode_load_slots) // 2)
        slots = self.decode_load_slots[:half][:num_blocks]
        return self.get_kv_for_slots(slots)

    def get_kv_for_prefetch(
        self,
        num_blocks: int,
    ) -> Tuple[Tensor, Tensor]:
        """Legacy: Get KV from 'prefetch' region (second half of decode_load_slots)."""
        half = max(1, len(self.decode_load_slots) // 2)
        slots = self.decode_load_slots[half:]
        if not slots:
            slots = self.decode_load_slots
        slots = slots[:num_blocks]
        return self.get_kv_for_slots(slots)

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