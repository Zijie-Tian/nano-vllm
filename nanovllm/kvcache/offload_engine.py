"""
High-performance CPU-GPU KV cache transfer engine.

Key design principles for CUDA Graph compatibility:
1. All tensor addresses are fixed at initialization
2. Only index tensor contents change between graph replays
3. Supports both async transfer (for prefill) and graph-based transfer (for decode)
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from nanovllm.kvcache.kernels import gathered_copy_kv
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

        # ========== Fixed-address GPU KV cache ==========
        # Shape: [num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]
        self.k_cache_gpu = torch.empty(
            num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        self.v_cache_gpu = torch.empty(
            num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )

        # ========== Fixed-address CPU KV cache (pinned memory) ==========
        self.k_cache_cpu = torch.empty(
            num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cpu", pin_memory=True
        )
        self.v_cache_cpu = torch.empty(
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

        # ========== Transfer streams for async operations ==========
        self.transfer_streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.compute_stream = torch.cuda.current_stream()
        self._stream_idx = 0

        # ========== Event tracking for async transfers ==========
        self.pending_events: Dict[Tuple[int, int], torch.cuda.Event] = {}

    def _get_next_stream(self) -> torch.cuda.Stream:
        """Round-robin stream selection for parallel transfers."""
        stream = self.transfer_streams[self._stream_idx]
        self._stream_idx = (self._stream_idx + 1) % len(self.transfer_streams)
        return stream

    # ========== CUDA Graph compatible methods ==========

    def gathered_h2d_layer(self, layer_id: int) -> None:
        """
        Execute gathered H2D copy for a single layer.

        This method is CUDA Graph compatible - can be captured into a graph.
        Before calling, update_gather_indices() must be called to set up
        which CPU blocks to copy to which GPU slots.

        Args:
            layer_id: Layer index to transfer
        """
        gathered_copy_kv(
            k_src=self.k_cache_cpu[layer_id],
            v_src=self.v_cache_cpu[layer_id],
            k_dst=self.k_cache_gpu[layer_id],
            v_dst=self.v_cache_gpu[layer_id],
            indices=self.gather_indices_gpu[layer_id],
        )

    def gathered_h2d_all_layers(self) -> None:
        """
        Execute gathered H2D copy for all layers.

        CUDA Graph compatible - can be captured into a single graph.
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

        For use in prefill phase where CUDA graphs are not used.

        Args:
            layer_id: Layer index
            cpu_block_id: Source block in CPU cache
            gpu_block_id: Destination slot in GPU cache

        Returns:
            CUDA event that signals completion
        """
        stream = self._get_next_stream()
        event = torch.cuda.Event()

        logger.debug(f"H2D prefetch: layer={layer_id}, CPU[{cpu_block_id}] -> GPU[{gpu_block_id}]")

        with torch.cuda.stream(stream):
            # K cache
            self.k_cache_gpu[layer_id, gpu_block_id].copy_(
                self.k_cache_cpu[layer_id, cpu_block_id],
                non_blocking=True
            )
            # V cache
            self.v_cache_gpu[layer_id, gpu_block_id].copy_(
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

        Args:
            layer_id: Layer index
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

            # K cache
            self.k_cache_cpu[layer_id, cpu_block_id].copy_(
                self.k_cache_gpu[layer_id, gpu_block_id],
                non_blocking=True
            )
            # V cache
            self.v_cache_cpu[layer_id, cpu_block_id].copy_(
                self.v_cache_gpu[layer_id, gpu_block_id],
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

        Uses the main GPU KV cache slots, not a separate temp buffer.
        This is the same mechanism as chunked prefill uses.

        Args:
            layer_id: Layer index
            cpu_block_ids: List of CPU block IDs to load
            gpu_slot_ids: List of GPU slot IDs to load into (must be same length)
        """
        assert len(cpu_block_ids) == len(gpu_slot_ids)

        if cpu_block_ids:
            logger.debug(f"H2D chunked load: layer={layer_id}, CPU{cpu_block_ids} -> GPU{gpu_slot_ids}")

        stream = self._get_next_stream()

        with torch.cuda.stream(stream):
            for cpu_block_id, gpu_slot in zip(cpu_block_ids, gpu_slot_ids):
                # Copy from pinned CPU memory to GPU KV cache slot
                self.k_cache_gpu[layer_id, gpu_slot].copy_(
                    self.k_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )
                self.v_cache_gpu[layer_id, gpu_slot].copy_(
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

        Args:
            layer_id: Layer index
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
                self.k_cache_gpu[layer_id, gpu_slot].copy_(
                    self.k_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )
                self.v_cache_gpu[layer_id, gpu_slot].copy_(
                    self.v_cache_cpu[layer_id, cpu_block_id],
                    non_blocking=True
                )
            event.record()

        return event

    def load_cpu_blocks_to_gpu_slots_all_layers(
        self,
        cpu_block_ids: List[int],
        gpu_slot_ids: List[int],
    ) -> None:
        """
        Load CPU blocks to GPU slots for ALL layers at once.

        More efficient than per-layer loading when we know the mapping upfront.

        Args:
            cpu_block_ids: List of CPU block IDs to load
            gpu_slot_ids: List of GPU slot IDs to load into
        """
        assert len(cpu_block_ids) == len(gpu_slot_ids)

        if cpu_block_ids:
            logger.debug(f"H2D all layers: CPU{cpu_block_ids} -> GPU{gpu_slot_ids}")

        stream = self._get_next_stream()

        with torch.cuda.stream(stream):
            for cpu_block_id, gpu_slot in zip(cpu_block_ids, gpu_slot_ids):
                # Copy all layers at once
                self.k_cache_gpu[:, gpu_slot].copy_(
                    self.k_cache_cpu[:, cpu_block_id],
                    non_blocking=True
                )
                self.v_cache_gpu[:, gpu_slot].copy_(
                    self.v_cache_cpu[:, cpu_block_id],
                    non_blocking=True
                )

        stream.synchronize()

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
        torch.cuda.current_stream().synchronize()

    # ========== Cache access methods ==========

    def get_layer_cache(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        """
        Get GPU K/V cache tensors for a specific layer.

        Returns:
            (k_cache, v_cache) tensors for the layer
            Shape: [num_gpu_blocks, block_size, kv_heads, head_dim]
        """
        return self.k_cache_gpu[layer_id], self.v_cache_gpu[layer_id]

    def get_all_gpu_cache(self) -> Tuple[Tensor, Tensor]:
        """
        Get full GPU K/V cache tensors.

        Returns:
            (k_cache, v_cache) tensors
            Shape: [num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]
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
            f"  gpu_memory={self.gpu_memory_bytes() / 1024**2:.1f}MB,\n"
            f"  cpu_memory={self.cpu_memory_bytes() / 1024**2:.1f}MB\n"
            f")"
        )