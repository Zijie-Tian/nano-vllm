"""
High-performance CPU-GPU KV cache transfer engine for layer-wise offload.

Key design principles:
1. Layer-wise processing: process entire sequence through one layer at a time
2. Ring-buffered GPU KV cache for decode phase (configurable num_kv_buffers)
3. Async D2H offload during prefill with per-layer streams
4. Async H2D load during decode with ring buffer pipeline
"""

import torch
import torch.cuda.nvtx
from torch import Tensor
from typing import Dict, List, Tuple, Optional

from nanovllm.utils.logger import get_logger

# Import for type hints only (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanovllm.kvcache.sparse import SparsePolicy

logger = get_logger("offload_engine")


class OffloadEngine:
    """
    High-performance CPU-GPU async transfer engine for layer-wise KV cache offloading.

    Memory layout:
    - CPU cache: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim] (pinned)
    - GPU layer buffers: [num_kv_buffers, max_seq_tokens, kv_heads, head_dim] (ring buffer)
    - Decode KV buffer: [num_layers, block_size, kv_heads, head_dim] (per-layer decode)

    Features:
    - Ring buffer for decode H2D pipeline (configurable depth)
    - Per-layer async D2H offload during prefill
    - Stream-based synchronization (no global synchronize)
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
        num_kv_buffers: int = 4,
        max_seq_len: int = 131072,
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
        self.num_kv_buffers = num_kv_buffers
        self.max_seq_len = max_seq_len

        logger.info(f"OffloadEngine initializing: num_layers={num_layers}, "
                    f"num_kv_buffers={num_kv_buffers}, max_seq_len={max_seq_len}")

        # ========== Ring-Buffered GPU KV Cache for Layer-wise Decode ==========
        #
        # Ring Buffer流水线 (以4个buffer为例):
        #   Buffer 0: [Load L0] → [Compute L0] → [Load L4] → ...
        #   Buffer 1:           [Load L1] → [Compute L1] → [Load L5] → ...
        #   Buffer 2:                     [Load L2] → [Compute L2] → ...
        #   Buffer 3:                               [Load L3] → [Compute L3] → ...
        #
        # Shape: [num_kv_buffers, max_seq_len, kv_heads, head_dim]
        self.layer_k_cache = torch.zeros(
            num_kv_buffers, max_seq_len, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        self.layer_v_cache = torch.zeros(
            num_kv_buffers, max_seq_len, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        layer_cache_mb = 2 * num_kv_buffers * max_seq_len * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        logger.info(f"  Ring buffer GPU cache: {layer_cache_mb:.1f} MB "
                    f"({num_kv_buffers} buffers × {max_seq_len} tokens)")

        # ========== Per-layer Decode Buffer ==========
        # During decode, accumulate new tokens' KV per layer until block is full
        # Shape: [num_layers, block_size, kv_heads, head_dim]
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
        cpu_mem_mb = 2 * num_layers * num_cpu_blocks * block_size * num_kv_heads * head_dim * dtype.itemsize / (1024 * 1024)
        logger.info(f"  CPU cache: {cpu_mem_mb:.1f} MB "
                    f"({num_layers} layers × {num_cpu_blocks} blocks)")

        # ========== Compute Stream ==========
        # IMPORTANT: Create a dedicated compute stream (not default stream!)
        # Default stream has implicit synchronization with other streams,
        # which prevents overlap between transfer and compute.
        self.compute_stream = torch.cuda.Stream()

        # ========== Prefill: Per-layer D2H offload streams and events ==========
        # Each layer has its own stream for parallel offloads
        self.prefill_offload_streams = [torch.cuda.Stream() for _ in range(num_layers)]
        self.prefill_offload_events = [torch.cuda.Event() for _ in range(num_layers)]

        # ========== Decode: Ring buffer H2D load streams and events ==========
        # Per-buffer streams for parallel loading
        self.layer_load_streams = [torch.cuda.Stream() for _ in range(num_kv_buffers)]
        self.buffer_load_events = [torch.cuda.Event() for _ in range(num_kv_buffers)]
        self.buffer_compute_done_events = [torch.cuda.Event() for _ in range(num_kv_buffers)]

        # Initialize: mark all buffers as "compute done" (allows first load)
        for event in self.buffer_compute_done_events:
            event.record()

        # ========== Decode offload stream ==========
        self.decode_offload_stream = torch.cuda.Stream()
        self.decode_offload_event = torch.cuda.Event()

        # ========== Sparse attention policy ==========
        self.sparse_policy = sparse_policy

        logger.info(f"OffloadEngine initialized: GPU={self.gpu_memory_bytes()/(1024**2):.1f}MB, "
                    f"CPU={self.cpu_memory_bytes()/(1024**2):.1f}MB")

    # ========== Memory info ==========

    def gpu_memory_bytes(self) -> int:
        """Total GPU memory used by KV caches."""
        return (
            self.layer_k_cache.numel() * self.layer_k_cache.element_size() +
            self.layer_v_cache.numel() * self.layer_v_cache.element_size() +
            self.decode_k_buffer.numel() * self.decode_k_buffer.element_size() +
            self.decode_v_buffer.numel() * self.decode_v_buffer.element_size()
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
            f"  num_kv_buffers={self.num_kv_buffers},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  num_cpu_blocks={self.num_cpu_blocks},\n"
            f"  block_size={self.block_size},\n"
            f"  kv_heads={self.num_kv_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  dtype={self.dtype},\n"
            f"  gpu_memory={self.gpu_memory_bytes() / 1024**2:.1f}MB,\n"
            f"  cpu_memory={self.cpu_memory_bytes() / 1024**2:.1f}MB\n"
            f")"
        )

    # ========== Prefill: Async D2H Offload API ==========

    def offload_layer_kv_async(
        self,
        layer_id: int,
        k: Tensor,
        v: Tensor,
        cpu_block_ids: List[int],
        total_tokens: int,
    ) -> None:
        """
        Async offload layer KV to CPU using per-layer stream.

        This enables overlap: layer N offload overlaps with layer N+1 compute.

        Args:
            layer_id: Layer index
            k: Key tensor [seq_len, kv_heads, head_dim]
            v: Value tensor [seq_len, kv_heads, head_dim]
            cpu_block_ids: List of CPU block IDs to offload to
            total_tokens: Total number of tokens
        """
        stream = self.prefill_offload_streams[layer_id]

        torch.cuda.nvtx.range_push(f"D2H: L{layer_id}")
        with torch.cuda.stream(stream):
            # Wait for compute to finish
            stream.wait_stream(self.compute_stream)

            # Copy to CPU in blocks
            for i, cpu_block_id in enumerate(cpu_block_ids):
                start = i * self.block_size
                end = min(start + self.block_size, total_tokens)
                actual_size = end - start

                self.k_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(
                    k[start:end], non_blocking=True
                )
                self.v_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(
                    v[start:end], non_blocking=True
                )

            # Record completion event
            self.prefill_offload_events[layer_id].record(stream)
        torch.cuda.nvtx.range_pop()

    def wait_layer_offload(self, layer_id: int) -> None:
        """
        Wait for specific layer's offload to complete on compute_stream.

        Call this before reusing the layer's GPU buffer.
        """
        self.compute_stream.wait_event(self.prefill_offload_events[layer_id])

    def wait_all_prefill_offloads(self) -> None:
        """Wait for all prefill offloads to complete."""
        for stream in self.prefill_offload_streams:
            stream.synchronize()

    # ========== Decode: Ring-Buffered H2D Load API ==========

    def load_layer_kv_to_buffer(
        self,
        buffer_idx: int,
        layer_id: int,
        cpu_block_ids: List[int],
        valid_tokens_per_block: List[int],
    ) -> None:
        """
        Async load layer KV from CPU to specified ring buffer slot.

        Args:
            buffer_idx: Ring buffer slot index (0 to num_kv_buffers-1)
            layer_id: Which layer's KV to load
            cpu_block_ids: CPU block IDs containing this layer's KV
            valid_tokens_per_block: Number of valid tokens in each block
        """
        stream = self.layer_load_streams[buffer_idx]

        torch.cuda.nvtx.range_push(f"H2D: L{layer_id}->Buf{buffer_idx}")
        with torch.cuda.stream(stream):
            # Wait for previous compute on this buffer to complete
            stream.wait_event(self.buffer_compute_done_events[buffer_idx])

            offset = 0
            for i, cpu_block_id in enumerate(cpu_block_ids):
                valid_tokens = valid_tokens_per_block[i]
                self.layer_k_cache[buffer_idx, offset:offset+valid_tokens].copy_(
                    self.k_cache_cpu[layer_id, cpu_block_id, :valid_tokens],
                    non_blocking=True
                )
                self.layer_v_cache[buffer_idx, offset:offset+valid_tokens].copy_(
                    self.v_cache_cpu[layer_id, cpu_block_id, :valid_tokens],
                    non_blocking=True
                )
                offset += valid_tokens

            self.buffer_load_events[buffer_idx].record(stream)
        torch.cuda.nvtx.range_pop()

    def wait_buffer_load(self, buffer_idx: int) -> None:
        """Wait for buffer load to complete on compute_stream."""
        self.compute_stream.wait_event(self.buffer_load_events[buffer_idx])

    def get_buffer_kv(self, buffer_idx: int, total_tokens: int) -> Tuple[Tensor, Tensor]:
        """Get KV from specified ring buffer slot."""
        return (
            self.layer_k_cache[buffer_idx, :total_tokens],
            self.layer_v_cache[buffer_idx, :total_tokens]
        )

    def record_buffer_compute_done(self, buffer_idx: int) -> None:
        """Record that compute on this buffer is done (allows next load to reuse it)."""
        self.buffer_compute_done_events[buffer_idx].record(self.compute_stream)

    # ========== Decode Buffer API ==========

    def get_decode_kv(self, layer_id: int, start_pos: int, end_pos: int) -> Tuple[Tensor, Tensor]:
        """
        Get accumulated decode KV for a layer.

        Args:
            layer_id: Layer index
            start_pos: Start position in block
            end_pos: End position in block (exclusive)

        Returns:
            (k, v) tensors with shape [end_pos - start_pos, kv_heads, head_dim]
        """
        return (
            self.decode_k_buffer[layer_id, start_pos:end_pos],
            self.decode_v_buffer[layer_id, start_pos:end_pos]
        )

    def store_decode_kv(
        self,
        layer_id: int,
        pos_in_block: int,
        k: Tensor,
        v: Tensor,
    ) -> None:
        """
        Store new decode token's KV to decode buffer.

        Args:
            layer_id: Layer index
            pos_in_block: Position within block (0 to block_size-1)
            k: Key tensor [1, kv_heads, head_dim]
            v: Value tensor [1, kv_heads, head_dim]
        """
        self.decode_k_buffer[layer_id, pos_in_block].copy_(k.squeeze(0))
        self.decode_v_buffer[layer_id, pos_in_block].copy_(v.squeeze(0))

    def offload_decode_buffer_async(self, cpu_block_id: int) -> None:
        """
        Async offload entire decode buffer to CPU.

        Called when a decode block is full. Also calls sparse policy hooks
        to update metadata (e.g., Quest min/max keys).

        Args:
            cpu_block_id: Target CPU block ID
        """
        torch.cuda.nvtx.range_push(f"D2H: DecBuf->CPU[{cpu_block_id}]")
        with torch.cuda.stream(self.decode_offload_stream):
            self.decode_offload_stream.wait_stream(self.compute_stream)

            for layer_id in range(self.num_layers):
                # Hook: notify sparse policy BEFORE offload (k still on GPU)
                if self.sparse_policy is not None:
                    self.sparse_policy.on_decode_offload(
                        cpu_block_id, layer_id,
                        self.decode_k_buffer[layer_id],
                        self.block_size  # Full block
                    )

                self.k_cache_cpu[layer_id, cpu_block_id].copy_(
                    self.decode_k_buffer[layer_id], non_blocking=True
                )
                self.v_cache_cpu[layer_id, cpu_block_id].copy_(
                    self.decode_v_buffer[layer_id], non_blocking=True
                )

            self.decode_offload_event.record(self.decode_offload_stream)
        torch.cuda.nvtx.range_pop()

    def wait_decode_offload(self) -> None:
        """Wait for decode buffer offload to complete."""
        self.compute_stream.wait_event(self.decode_offload_event)

    # ========== Encapsulated Prefill Offload API (with sparse hooks) ==========

    def offload_layer_kv_sync(
        self,
        layer_id: int,
        k: Tensor,
        v: Tensor,
        cpu_block_ids: List[int],
        total_tokens: int,
    ) -> None:
        """
        Synchronously offload layer KV to CPU with sparse policy hooks.

        This method encapsulates:
        1. Block-wise copy to CPU cache
        2. Sparse policy hooks (on_prefill_offload for Quest metadata)

        Args:
            layer_id: Layer index
            k: Key tensor [seq_len, kv_heads, head_dim]
            v: Value tensor [seq_len, kv_heads, head_dim]
            cpu_block_ids: List of CPU block IDs to offload to
            total_tokens: Total number of tokens
        """
        for i, cpu_block_id in enumerate(cpu_block_ids):
            start = i * self.block_size
            end = min(start + self.block_size, total_tokens)
            actual_size = end - start

            # Hook: notify sparse policy BEFORE offload (k still on GPU)
            if self.sparse_policy is not None:
                self.sparse_policy.on_prefill_offload(
                    cpu_block_id, layer_id, k[start:end], actual_size
                )

            # Synchronous copy to CPU
            self.k_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(k[start:end])
            self.v_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(v[start:end])
