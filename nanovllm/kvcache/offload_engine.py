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
        num_prefetch_blocks: int = 2,
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

        # ========== Three-region GPU Buffer configuration ==========
        # Constraint checks
        assert num_gpu_blocks >= 3, \
            f"Need at least 3 GPU blocks: 1 decode + 1 compute + 1 prefetch, got {num_gpu_blocks}"
        assert num_prefetch_blocks >= 1, \
            f"Need at least 1 prefetch block, got {num_prefetch_blocks}"
        assert num_gpu_blocks >= 1 + 1 + num_prefetch_blocks, \
            f"Insufficient GPU blocks: need 1(decode) + 1(compute) + {num_prefetch_blocks}(prefetch), got {num_gpu_blocks}"

        # Three-region configuration
        # Decode region: [0] - Fixed 1 block for writing new KV
        self.decode_slot = 0

        # Compute region: [1, ..., num_gpu_blocks - num_prefetch_blocks - 1]
        compute_start = 1
        compute_end = num_gpu_blocks - num_prefetch_blocks
        self.compute_slots = list(range(compute_start, compute_end))
        self.num_compute_blocks = len(self.compute_slots)

        # Prefetch region: [num_gpu_blocks - num_prefetch_blocks, ..., num_gpu_blocks - 1]
        prefetch_start = compute_end
        self.prefetch_slots = list(range(prefetch_start, num_gpu_blocks))
        self.num_prefetch_blocks = num_prefetch_blocks

        self.num_gpu_slots = num_gpu_blocks  # alias

        # Keep old ping/pong attributes for compatibility (will be removed later)
        self.ping_size = self.num_compute_blocks
        self.pong_size = self.num_prefetch_blocks
        self.ping_slots = self.compute_slots.copy()
        self.pong_slots = self.prefetch_slots.copy()

        logger.info(f"Three-region GPU Buffer: decode_slot={self.decode_slot}, "
                    f"compute_slots={self.compute_slots}, prefetch_slots={self.prefetch_slots}")

        # ========== Fixed-address GPU KV cache ==========
        # Shape: [num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]
        # Use zeros initialization to avoid uninitialized memory issues
        self.k_cache_gpu = torch.zeros(
            num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        self.v_cache_gpu = torch.zeros(
            num_layers, num_gpu_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )

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

        # ========== Transfer streams for async operations ==========
        self.transfer_streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.compute_stream = torch.cuda.current_stream()
        self._stream_idx = 0

        # ========== Three-region dedicated stream and events ==========
        self.transfer_stream_main = torch.cuda.Stream()  # Main transfer stream

        # Sync events - three-region loading completion
        self.compute_ready = torch.cuda.Event()
        self.prefetch_ready = torch.cuda.Event()
        self.decode_offload_done = torch.cuda.Event()

        # Keep old ping/pong events for compatibility (will be removed later)
        self.pingpong_stream = self.transfer_stream_main
        self.ping_ready = self.compute_ready
        self.pong_ready = self.prefetch_ready
        self.ping_offload_done = torch.cuda.Event()
        self.pong_offload_done = torch.cuda.Event()

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
            f"  three_regions: decode_slot={self.decode_slot}, compute={self.compute_slots}, prefetch={self.prefetch_slots},\n"
            f"  gpu_memory={self.gpu_memory_bytes() / 1024**2:.1f}MB,\n"
            f"  cpu_memory={self.cpu_memory_bytes() / 1024**2:.1f}MB\n"
            f")"
        )

    # ========== Ping-Pong double buffering methods ==========

    def load_to_ping(self, cpu_block_ids: List[int]) -> None:
        """
        Async load CPU blocks to Ping buffer.

        Args:
            cpu_block_ids: List of CPU block IDs to load
        """
        if not cpu_block_ids:
            self.ping_ready.record(self.pingpong_stream)
            return

        num_to_load = min(len(cpu_block_ids), self.ping_size)
        logger.debug(f"Ping load: CPU{cpu_block_ids[:num_to_load]} -> GPU ping slots {self.ping_slots[:num_to_load]}")

        with torch.cuda.stream(self.pingpong_stream):
            for i in range(num_to_load):
                cpu_id = cpu_block_ids[i]
                gpu_slot = self.ping_slots[i]
                # Copy all layers together
                self.k_cache_gpu[:, gpu_slot].copy_(
                    self.k_cache_cpu[:, cpu_id], non_blocking=True
                )
                self.v_cache_gpu[:, gpu_slot].copy_(
                    self.v_cache_cpu[:, cpu_id], non_blocking=True
                )
            self.ping_ready.record(self.pingpong_stream)

    def load_to_pong(self, cpu_block_ids: List[int]) -> None:
        """
        Async load CPU blocks to Pong buffer.

        Args:
            cpu_block_ids: List of CPU block IDs to load
        """
        if not cpu_block_ids:
            self.pong_ready.record(self.pingpong_stream)
            return

        num_to_load = min(len(cpu_block_ids), self.pong_size)
        logger.debug(f"Pong load: CPU{cpu_block_ids[:num_to_load]} -> GPU pong slots {self.pong_slots[:num_to_load]}")

        with torch.cuda.stream(self.pingpong_stream):
            for i in range(num_to_load):
                cpu_id = cpu_block_ids[i]
                gpu_slot = self.pong_slots[i]
                self.k_cache_gpu[:, gpu_slot].copy_(
                    self.k_cache_cpu[:, cpu_id], non_blocking=True
                )
                self.v_cache_gpu[:, gpu_slot].copy_(
                    self.v_cache_cpu[:, cpu_id], non_blocking=True
                )
            self.pong_ready.record(self.pingpong_stream)

    def wait_ping(self) -> None:
        """Wait for Ping buffer loading to complete."""
        self.compute_stream.wait_event(self.ping_ready)

    def wait_pong(self) -> None:
        """Wait for Pong buffer loading to complete."""
        self.compute_stream.wait_event(self.pong_ready)

    def offload_buffer_to_cpu(
        self,
        buffer: str,
        cpu_block_ids: List[int],
    ) -> None:
        """
        Async offload KV from buffer to CPU.

        Args:
            buffer: "ping" or "pong"
            cpu_block_ids: Target CPU block IDs list
        """
        slots = self.ping_slots if buffer == "ping" else self.pong_slots
        event = self.ping_offload_done if buffer == "ping" else self.pong_offload_done

        if not cpu_block_ids:
            event.record(self.pingpong_stream)
            return

        num_to_offload = min(len(cpu_block_ids), len(slots))
        logger.debug(f"{buffer.capitalize()} offload: GPU {slots[:num_to_offload]} -> CPU{cpu_block_ids[:num_to_offload]}")

        with torch.cuda.stream(self.pingpong_stream):
            # Wait for compute to complete
            self.pingpong_stream.wait_stream(self.compute_stream)

            for i in range(num_to_offload):
                gpu_slot = slots[i]
                cpu_id = cpu_block_ids[i]
                self.k_cache_cpu[:, cpu_id].copy_(
                    self.k_cache_gpu[:, gpu_slot], non_blocking=True
                )
                self.v_cache_cpu[:, cpu_id].copy_(
                    self.v_cache_gpu[:, gpu_slot], non_blocking=True
                )
            event.record(self.pingpong_stream)

    def offload_slot_to_cpu(
        self,
        gpu_slot: int,
        cpu_block_id: int,
    ) -> None:
        """
        Async offload a single GPU slot's KV to CPU.

        Args:
            gpu_slot: GPU slot ID
            cpu_block_id: Target CPU block ID
        """
        logger.debug(f"Slot offload: GPU[{gpu_slot}] -> CPU[{cpu_block_id}]")

        with torch.cuda.stream(self.pingpong_stream):
            self.pingpong_stream.wait_stream(self.compute_stream)
            self.k_cache_cpu[:, cpu_block_id].copy_(
                self.k_cache_gpu[:, gpu_slot], non_blocking=True
            )
            self.v_cache_cpu[:, cpu_block_id].copy_(
                self.v_cache_gpu[:, gpu_slot], non_blocking=True
            )

    def wait_ping_offload_done(self) -> None:
        """Wait for Ping buffer offload to complete."""
        self.compute_stream.wait_event(self.ping_offload_done)

    def wait_pong_offload_done(self) -> None:
        """Wait for Pong buffer offload to complete."""
        self.compute_stream.wait_event(self.pong_offload_done)

    def wait_all_offload_done(self) -> None:
        """Wait for all offload operations to complete."""
        self.pingpong_stream.synchronize()

    def get_kv_for_ping_slots(
        self,
        layer_id: int,
        num_slots: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for specified number of slots in Ping buffer.

        Args:
            layer_id: Layer ID
            num_slots: Number of slots needed

        Returns:
            (k_cache, v_cache), shape: [1, num_slots * block_size, kv_heads, head_dim]
        """
        slots = self.ping_slots[:num_slots]
        k = self.k_cache_gpu[layer_id, slots]  # [num_slots, block_size, heads, dim]
        v = self.v_cache_gpu[layer_id, slots]
        # Reshape: [num_slots, block_size, heads, dim] -> [1, num_slots*block_size, heads, dim]
        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        return k, v

    def get_kv_for_pong_slots(
        self,
        layer_id: int,
        num_slots: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for specified number of slots in Pong buffer.

        Args:
            layer_id: Layer ID
            num_slots: Number of slots needed

        Returns:
            (k_cache, v_cache), shape: [1, num_slots * block_size, kv_heads, head_dim]
        """
        slots = self.pong_slots[:num_slots]
        k = self.k_cache_gpu[layer_id, slots]
        v = self.v_cache_gpu[layer_id, slots]
        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        return k, v

    def get_kv_for_slots(
        self,
        layer_id: int,
        gpu_slots: List[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for specified GPU slots.

        Args:
            layer_id: Layer ID
            gpu_slots: List of GPU slot IDs

        Returns:
            (k_cache, v_cache), shape: [1, len(slots) * block_size, kv_heads, head_dim]
        """
        if not gpu_slots:
            return None, None
        k = self.k_cache_gpu[layer_id, gpu_slots]
        v = self.v_cache_gpu[layer_id, gpu_slots]
        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        return k, v

    # ========== Three-region GPU Buffer methods ==========

    def load_to_compute(self, cpu_block_ids: List[int]) -> None:
        """
        Async load CPU blocks to Compute region.

        Args:
            cpu_block_ids: List of CPU block IDs to load
        """
        if not cpu_block_ids:
            self.compute_ready.record(self.transfer_stream_main)
            return

        num_to_load = min(len(cpu_block_ids), len(self.compute_slots))
        logger.debug(f"Compute load: CPU{cpu_block_ids[:num_to_load]} -> GPU compute slots {self.compute_slots[:num_to_load]}")

        with torch.cuda.stream(self.transfer_stream_main):
            for i in range(num_to_load):
                cpu_id = cpu_block_ids[i]
                gpu_slot = self.compute_slots[i]
                # Copy all layers together
                self.k_cache_gpu[:, gpu_slot].copy_(
                    self.k_cache_cpu[:, cpu_id], non_blocking=True
                )
                self.v_cache_gpu[:, gpu_slot].copy_(
                    self.v_cache_cpu[:, cpu_id], non_blocking=True
                )
            self.compute_ready.record(self.transfer_stream_main)

    def load_to_prefetch(self, cpu_block_ids: List[int]) -> None:
        """
        Async load CPU blocks to Prefetch region.

        Args:
            cpu_block_ids: List of CPU block IDs to load
        """
        if not cpu_block_ids:
            self.prefetch_ready.record(self.transfer_stream_main)
            return

        num_to_load = min(len(cpu_block_ids), len(self.prefetch_slots))
        logger.debug(f"Prefetch load: CPU{cpu_block_ids[:num_to_load]} -> GPU prefetch slots {self.prefetch_slots[:num_to_load]}")

        with torch.cuda.stream(self.transfer_stream_main):
            for i in range(num_to_load):
                cpu_id = cpu_block_ids[i]
                gpu_slot = self.prefetch_slots[i]
                self.k_cache_gpu[:, gpu_slot].copy_(
                    self.k_cache_cpu[:, cpu_id], non_blocking=True
                )
                self.v_cache_gpu[:, gpu_slot].copy_(
                    self.v_cache_cpu[:, cpu_id], non_blocking=True
                )
            self.prefetch_ready.record(self.transfer_stream_main)

    def wait_compute(self) -> None:
        """Wait for Compute region loading to complete."""
        self.compute_stream.wait_event(self.compute_ready)

    def wait_prefetch(self) -> None:
        """Wait for Prefetch region loading to complete."""
        self.compute_stream.wait_event(self.prefetch_ready)

    def swap_compute_prefetch(self) -> None:
        """Swap roles of Compute region and Prefetch region."""
        self.compute_slots, self.prefetch_slots = self.prefetch_slots, self.compute_slots
        # Also update old ping/pong slots for compatibility
        self.ping_slots, self.pong_slots = self.pong_slots, self.ping_slots

    def offload_decode_slot(self, cpu_block_id: int) -> None:
        """
        Offload KV from Decode region to CPU.

        Args:
            cpu_block_id: Target CPU block ID
        """
        logger.debug(f"Decode offload: GPU[{self.decode_slot}] -> CPU[{cpu_block_id}]")

        with torch.cuda.stream(self.transfer_stream_main):
            self.transfer_stream_main.wait_stream(self.compute_stream)
            self.k_cache_cpu[:, cpu_block_id].copy_(
                self.k_cache_gpu[:, self.decode_slot], non_blocking=True
            )
            self.v_cache_cpu[:, cpu_block_id].copy_(
                self.v_cache_gpu[:, self.decode_slot], non_blocking=True
            )
            self.decode_offload_done.record(self.transfer_stream_main)

    def wait_decode_offload(self) -> None:
        """Wait for Decode region offload to complete."""
        self.compute_stream.wait_event(self.decode_offload_done)

    def get_kv_for_compute(
        self,
        layer_id: int,
        num_blocks: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for specified number of blocks in Compute region.

        Args:
            layer_id: Layer ID
            num_blocks: Number of blocks needed

        Returns:
            (k_cache, v_cache), shape: [1, num_blocks * block_size, kv_heads, head_dim]
        """
        slots = self.compute_slots[:num_blocks]
        k = self.k_cache_gpu[layer_id, slots]  # [num_blocks, block_size, heads, dim]
        v = self.v_cache_gpu[layer_id, slots]
        # Reshape: [num_blocks, block_size, heads, dim] -> [1, num_blocks*block_size, heads, dim]
        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        return k, v

    def get_kv_for_prefetch(
        self,
        layer_id: int,
        num_blocks: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for specified number of blocks in Prefetch region.

        Args:
            layer_id: Layer ID
            num_blocks: Number of blocks needed

        Returns:
            (k_cache, v_cache), shape: [1, num_blocks * block_size, kv_heads, head_dim]
        """
        slots = self.prefetch_slots[:num_blocks]
        k = self.k_cache_gpu[layer_id, slots]
        v = self.v_cache_gpu[layer_id, slots]
        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        return k, v

    def get_kv_for_decode_slot(
        self,
        layer_id: int,
        pos_in_block: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV at specified position in Decode region (for new token during decode).

        Args:
            layer_id: Layer ID
            pos_in_block: Token position within block (0 to block_size-1)

        Returns:
            (k_cache, v_cache), shape: [1, 1, kv_heads, head_dim]
        """
        k = self.k_cache_gpu[layer_id, self.decode_slot, pos_in_block:pos_in_block+1]  # [1, heads, dim]
        v = self.v_cache_gpu[layer_id, self.decode_slot, pos_in_block:pos_in_block+1]
        k = k.unsqueeze(0)  # [1, 1, heads, dim]
        v = v.unsqueeze(0)
        return k, v

    def get_kv_for_decode_slot_accumulated(
        self,
        layer_id: int,
        num_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get accumulated KV in Decode region (all tokens from position 0 to num_tokens-1).

        Used when batching decode offloads - attend to all accumulated tokens,
        not just the current one.

        Args:
            layer_id: Layer ID
            num_tokens: Number of accumulated tokens (1 to block_size)

        Returns:
            (k_cache, v_cache), shape: [1, num_tokens, kv_heads, head_dim]
        """
        k = self.k_cache_gpu[layer_id, self.decode_slot, :num_tokens]  # [num_tokens, heads, dim]
        v = self.v_cache_gpu[layer_id, self.decode_slot, :num_tokens]
        k = k.unsqueeze(0)  # [1, num_tokens, heads, dim]
        v = v.unsqueeze(0)
        return k, v

    def offload_compute_to_cpu(self, cpu_block_ids: List[int]) -> None:
        """
        Offload KV from Compute region to CPU.

        Args:
            cpu_block_ids: Target CPU block IDs list
        """
        if not cpu_block_ids:
            return

        num_to_offload = min(len(cpu_block_ids), len(self.compute_slots))
        logger.debug(f"Compute offload: GPU {self.compute_slots[:num_to_offload]} -> CPU{cpu_block_ids[:num_to_offload]}")

        with torch.cuda.stream(self.transfer_stream_main):
            # Wait for compute to complete
            self.transfer_stream_main.wait_stream(self.compute_stream)

            for i in range(num_to_offload):
                gpu_slot = self.compute_slots[i]
                cpu_id = cpu_block_ids[i]
                self.k_cache_cpu[:, cpu_id].copy_(
                    self.k_cache_gpu[:, gpu_slot], non_blocking=True
                )
                self.v_cache_cpu[:, cpu_id].copy_(
                    self.v_cache_gpu[:, gpu_slot], non_blocking=True
                )