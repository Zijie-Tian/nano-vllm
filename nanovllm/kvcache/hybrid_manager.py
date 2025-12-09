"""
Hybrid CPU-GPU KV cache manager with CUDA Graph support.

Key design for CUDA Graph compatibility:
1. GPU buffer has fixed addresses (allocated once)
2. CPU pool has fixed addresses (pinned memory)
3. gather_indices tensor has fixed address, variable content
4. H2D transfer uses gathered_copy kernel inside CUDA graphs
5. Graph replay only needs index updates (tiny overhead)
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Set, Optional
import torch
from torch import Tensor

from nanovllm.engine.sequence import Sequence
from nanovllm.kvcache.base_manager import KVCacheManager
from nanovllm.kvcache.offload_engine import OffloadEngine
from nanovllm.kvcache.policies.base_policy import EvictionPolicy
from nanovllm.kvcache.policies.lru_policy import LRUPolicy


class BlockLocation(Enum):
    """Where a logical block's data currently resides."""
    GPU = auto()
    CPU = auto()
    INVALID = auto()  # Not yet written / deallocated


@dataclass
class LogicalBlock:
    """
    Logical block that can be mapped to GPU or CPU physical storage.

    Sequences reference logical blocks. Physical blocks are the actual
    storage locations (GPU slots or CPU blocks).
    """
    logical_id: int
    location: BlockLocation = BlockLocation.INVALID
    gpu_slot: int = -1          # GPU buffer slot ID (if on GPU)
    cpu_block_id: int = -1      # CPU pool block ID (if on CPU)
    ref_count: int = 0
    hash: int = -1
    token_ids: List[int] = field(default_factory=list)

    def reset(self):
        self.location = BlockLocation.INVALID
        self.gpu_slot = -1
        self.cpu_block_id = -1
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class HybridKVCacheManager(KVCacheManager):
    """
    Hybrid CPU-GPU KV cache manager with CUDA Graph support.

    Architecture:
    - GPU buffer: Fixed-size working set (num_gpu_slots)
    - CPU pool: Overflow storage (num_cpu_blocks)
    - Logical blocks: What sequences reference (num_gpu_slots + num_cpu_blocks)

    CUDA Graph compatibility:
    - All tensor addresses fixed at init time
    - prepare_for_attention() updates gather_indices (outside graph)
    - gathered_h2d_layer() executes transfer (inside graph)

    Strategy:
    1. New KV data written to GPU slots
    2. Cold blocks evicted to CPU using configurable policy
    3. Needed blocks prefetched back to GPU before attention
    """

    def __init__(
        self,
        num_gpu_slots: int,
        num_cpu_blocks: int,
        block_size: int,
        policy: Optional[EvictionPolicy] = None,
    ):
        """
        Initialize hybrid manager.

        Args:
            num_gpu_slots: Number of GPU buffer slots (working set)
            num_cpu_blocks: Number of CPU pool blocks (overflow)
            block_size: Tokens per block
            policy: Eviction policy (default: LRU)
        """
        self._block_size = block_size
        self.num_gpu_slots = num_gpu_slots
        self.num_cpu_blocks = num_cpu_blocks
        self.total_blocks = num_gpu_slots + num_cpu_blocks

        # Eviction policy
        self.policy = policy or LRUPolicy()

        # Logical blocks (what sequences reference)
        self.logical_blocks: List[LogicalBlock] = [
            LogicalBlock(i) for i in range(self.total_blocks)
        ]
        self.free_logical_ids: deque[int] = deque(range(self.total_blocks))

        # GPU slot management (slots are fixed, mapping is variable)
        self.free_gpu_slots: deque[int] = deque(range(num_gpu_slots))
        self.gpu_slot_to_logical: Dict[int, int] = {}  # gpu_slot -> logical_id

        # CPU block management
        self.free_cpu_blocks: deque[int] = deque(range(num_cpu_blocks))
        self.cpu_block_to_logical: Dict[int, int] = {}  # cpu_block -> logical_id

        # Prefix cache (uses logical block IDs)
        self.hash_to_logical_id: Dict[int, int] = {}

        # Step counter for policy
        self.current_step = 0

        # Offload engine (set by allocate_cache)
        self.offload_engine: Optional[OffloadEngine] = None

        # Track blocks pending GPU load (for decode graph)
        self.pending_gpu_loads: Set[int] = set()  # logical_ids

        # Track blocks that have been prefilled (KV written) for chunked prefill
        self.prefilled_blocks: Set[int] = set()  # logical_ids

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_logical_ids)

    def allocate_cache(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the offload engine with actual cache storage."""
        self.offload_engine = OffloadEngine(
            num_layers=num_layers,
            num_gpu_blocks=self.num_gpu_slots,
            num_cpu_blocks=self.num_cpu_blocks,
            block_size=self._block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    def get_layer_cache(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        """Get GPU K/V cache tensors for a layer."""
        assert self.offload_engine is not None
        return self.offload_engine.get_layer_cache(layer_id)

    def _allocate_gpu_slot(self, protected_logical_ids: Optional[Set[int]] = None) -> int:
        """
        Get a free GPU slot, evicting if necessary.

        Args:
            protected_logical_ids: Logical block IDs that cannot be evicted

        Returns:
            GPU slot ID

        Raises:
            RuntimeError: If no GPU slot is available
        """
        if self.free_gpu_slots:
            return self.free_gpu_slots.popleft()

        # Need to evict - find victim using policy
        return self._evict_to_cpu(protected_logical_ids)

    def _try_allocate_gpu_slot(self, protected_logical_ids: Optional[Set[int]] = None) -> Optional[int]:
        """
        Try to get a free GPU slot, evicting if necessary.

        Unlike _allocate_gpu_slot(), returns None instead of raising if no eviction possible.

        Args:
            protected_logical_ids: Logical block IDs that cannot be evicted

        Returns:
            GPU slot ID, or None if no slot available
        """
        if self.free_gpu_slots:
            return self.free_gpu_slots.popleft()

        # Check if we can evict
        protected = protected_logical_ids or set()
        for gpu_slot, logical_id in self.gpu_slot_to_logical.items():
            if logical_id not in protected:
                block = self.logical_blocks[logical_id]
                if block.ref_count > 0:
                    # Found evictable block
                    return self._evict_to_cpu(protected_logical_ids)

        # No evictable blocks
        return None

    def _evict_to_cpu(self, protected_logical_ids: Optional[Set[int]] = None) -> int:
        """
        Evict a GPU block to CPU to make room.

        Args:
            protected_logical_ids: Logical block IDs that cannot be evicted

        Returns:
            The freed GPU slot ID
        """
        protected = protected_logical_ids or set()

        # Find candidates (blocks currently on GPU with ref_count > 0, excluding protected)
        candidates: Set[int] = set()
        for gpu_slot, logical_id in self.gpu_slot_to_logical.items():
            if logical_id in protected:
                continue  # Skip protected blocks
            block = self.logical_blocks[logical_id]
            if block.ref_count > 0:  # Only evict blocks still in use
                candidates.add(gpu_slot)

        if not candidates:
            raise RuntimeError(
                f"No GPU slots available for eviction. "
                f"GPU slots: {self.num_gpu_slots}, protected: {len(protected)}, "
                f"need more GPU memory or reduce sequence length"
            )

        # Use policy to select victim
        victim_gpu_slot = self.policy.select_victim(candidates)
        logical_id = self.gpu_slot_to_logical[victim_gpu_slot]
        block = self.logical_blocks[logical_id]

        # Allocate CPU block
        if not self.free_cpu_blocks:
            raise RuntimeError("Both GPU and CPU are full")
        cpu_block_id = self.free_cpu_blocks.popleft()

        # Async offload GPU -> CPU
        self.offload_engine.offload_block_async(
            layer_id=0,  # TODO: handle per-layer offloading
            gpu_block_id=victim_gpu_slot,
            cpu_block_id=cpu_block_id,
        )

        # Update mappings
        del self.gpu_slot_to_logical[victim_gpu_slot]
        self.cpu_block_to_logical[cpu_block_id] = logical_id

        block.location = BlockLocation.CPU
        block.gpu_slot = -1
        block.cpu_block_id = cpu_block_id

        # Notify policy
        self.policy.on_block_evicted(victim_gpu_slot)

        return victim_gpu_slot

    def _ensure_on_gpu(
        self,
        logical_id: int,
        protected_logical_ids: Optional[Set[int]] = None,
    ) -> int:
        """
        Ensure a logical block is on GPU.

        Args:
            logical_id: Logical block ID
            protected_logical_ids: Logical block IDs that cannot be evicted

        Returns:
            GPU slot ID where the block is/will be
        """
        block = self.logical_blocks[logical_id]

        if block.location == BlockLocation.GPU:
            # Already on GPU, update policy
            self.policy.on_block_access(block.gpu_slot, self.current_step)
            return block.gpu_slot

        if block.location == BlockLocation.CPU:
            # Need to prefetch from CPU
            gpu_slot = self._allocate_gpu_slot(protected_logical_ids)

            # Async prefetch CPU -> GPU
            self.offload_engine.prefetch_block_async(
                layer_id=0,  # TODO: handle per-layer
                cpu_block_id=block.cpu_block_id,
                gpu_block_id=gpu_slot,
            )

            # Update mappings
            self.free_cpu_blocks.append(block.cpu_block_id)
            del self.cpu_block_to_logical[block.cpu_block_id]

            self.gpu_slot_to_logical[gpu_slot] = logical_id

            block.location = BlockLocation.GPU
            block.gpu_slot = gpu_slot
            block.cpu_block_id = -1

            # Notify policy
            self.policy.on_block_prefetched(gpu_slot, self.current_step)

            return gpu_slot

        raise RuntimeError(f"Block {logical_id} is in invalid state")

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate blocks for a new sequence."""
        return len(self.free_logical_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """
        Allocate logical blocks for prefill.

        New blocks are allocated on GPU when possible. If GPU is full and all
        GPU blocks belong to this sequence (can't evict), remaining blocks
        are allocated to CPU for chunked prefill.
        """
        assert not seq.block_table, "Sequence already has blocks"

        h = -1
        cache_miss = False

        # Track blocks allocated for this sequence to protect them from eviction
        allocated_for_seq: Set[int] = set()

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            # Hash for full blocks only
            if len(token_ids) == self._block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1

            # Check prefix cache
            cached_logical_id = self.hash_to_logical_id.get(h, -1)
            if cached_logical_id != -1:
                cached_block = self.logical_blocks[cached_logical_id]
                if cached_block.token_ids == token_ids and cached_block.ref_count > 0:
                    # Cache hit
                    cached_block.ref_count += 1
                    seq.num_cached_tokens += self._block_size
                    seq.block_table.append(cached_logical_id)
                    allocated_for_seq.add(cached_logical_id)

                    # Ensure block is on GPU (protect already allocated blocks)
                    if cached_block.location == BlockLocation.CPU:
                        self._ensure_on_gpu(cached_logical_id, allocated_for_seq)

                    continue

            cache_miss = True

            # Allocate new logical block
            logical_id = self.free_logical_ids.popleft()
            block = self.logical_blocks[logical_id]
            block.ref_count = 1
            block.hash = h
            block.token_ids = token_ids.copy() if len(token_ids) == self._block_size else []

            # Try to allocate GPU slot
            gpu_slot = self._try_allocate_gpu_slot(allocated_for_seq)
            if gpu_slot is not None:
                # Got GPU slot
                block.location = BlockLocation.GPU
                block.gpu_slot = gpu_slot
                block.cpu_block_id = -1
                self.gpu_slot_to_logical[gpu_slot] = logical_id
            else:
                # GPU full and can't evict (all protected) - allocate to CPU
                # This block will be written via chunked prefill
                if not self.free_cpu_blocks:
                    raise RuntimeError(
                        f"Both GPU and CPU are full. Need {seq.num_blocks} blocks, "
                        f"GPU has {self.num_gpu_slots}, CPU has {self.num_cpu_blocks}"
                    )
                cpu_block_id = self.free_cpu_blocks.popleft()
                block.location = BlockLocation.CPU
                block.gpu_slot = -1
                block.cpu_block_id = cpu_block_id
                self.cpu_block_to_logical[cpu_block_id] = logical_id

            allocated_for_seq.add(logical_id)

            # Update prefix cache
            if h != -1:
                self.hash_to_logical_id[h] = logical_id

            # Notify policy
            self.policy.on_block_allocated(gpu_slot, self.current_step)

            seq.block_table.append(logical_id)

    def deallocate(self, seq: Sequence) -> None:
        """Release all blocks for a sequence."""
        for logical_id in reversed(seq.block_table):
            block = self.logical_blocks[logical_id]
            block.ref_count -= 1

            if block.ref_count == 0:
                # Free physical block
                if block.location == BlockLocation.GPU:
                    self.free_gpu_slots.append(block.gpu_slot)
                    del self.gpu_slot_to_logical[block.gpu_slot]
                    self.policy.on_block_deallocated(block.gpu_slot)
                elif block.location == BlockLocation.CPU:
                    self.free_cpu_blocks.append(block.cpu_block_id)
                    del self.cpu_block_to_logical[block.cpu_block_id]

                # Free logical block
                block.reset()
                self.free_logical_ids.append(logical_id)

                # Remove from prefilled tracking
                self.prefilled_blocks.discard(logical_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a token."""
        need_new_block = (len(seq) % self._block_size == 1)
        return len(self.free_logical_ids) >= int(need_new_block)

    def may_append(self, seq: Sequence) -> None:
        """Handle potential new block allocation during decode."""
        block_table = seq.block_table
        last_logical_id = block_table[-1]
        last_block = self.logical_blocks[last_logical_id]

        seq_len = len(seq)
        pos_in_block = seq_len % self._block_size

        if pos_in_block == 1:
            # Need new block
            assert last_block.hash != -1

            logical_id = self.free_logical_ids.popleft()
            block = self.logical_blocks[logical_id]
            block.ref_count = 1
            block.hash = -1
            block.token_ids = []

            # New decode blocks go to GPU
            gpu_slot = self._allocate_gpu_slot()
            block.location = BlockLocation.GPU
            block.gpu_slot = gpu_slot

            self.gpu_slot_to_logical[gpu_slot] = logical_id
            self.policy.on_block_allocated(gpu_slot, self.current_step)

            block_table.append(logical_id)

        elif pos_in_block == 0:
            # Block is full, update hash for prefix cache
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix_hash = (
                self.logical_blocks[block_table[-2]].hash
                if len(block_table) > 1 else -1
            )
            h = self.compute_hash(token_ids, prefix_hash)
            last_block.hash = h
            last_block.token_ids = token_ids.copy()
            self.hash_to_logical_id[h] = last_logical_id

    def prepare_for_attention(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """
        Prepare KV cache for attention computation.

        For prefill: async prefetch blocks from CPU to GPU.
        For decode: update gather_indices for CUDA graph.
        """
        self.current_step += 1

        # Collect all needed logical blocks
        needed_logical_ids: Set[int] = set()
        for seq in seqs:
            needed_logical_ids.update(seq.block_table)

        if is_prefill:
            # Prefill: ensure all blocks on GPU (async prefetch)
            # Pass needed_logical_ids as protected to prevent evicting blocks we need
            for logical_id in needed_logical_ids:
                block = self.logical_blocks[logical_id]
                if block.location == BlockLocation.CPU:
                    self._ensure_on_gpu(logical_id, needed_logical_ids)

            # Wait for all prefetches to complete
            self.offload_engine.wait_all_transfers()

        else:
            # Decode: Check if we need chunked decode
            cpu_blocks_count = sum(
                1 for lid in needed_logical_ids
                if self.logical_blocks[lid].location == BlockLocation.CPU
            )

            if cpu_blocks_count > self.num_gpu_slots:
                # Too many blocks on CPU - will use chunked decode
                # Don't try to load all blocks now
                return

            # Standard decode: prepare gather_indices for CUDA graph
            # Identify blocks needing transfer
            self.pending_gpu_loads.clear()
            mappings_per_layer: List[List[Tuple[int, int]]] = [
                [] for _ in range(self.offload_engine.num_layers)
            ]

            for logical_id in needed_logical_ids:
                block = self.logical_blocks[logical_id]
                if block.location == BlockLocation.CPU:
                    # Allocate GPU slot (protect needed blocks from eviction)
                    gpu_slot = self._allocate_gpu_slot(needed_logical_ids)

                    # Record mapping for each layer
                    for layer_id in range(self.offload_engine.num_layers):
                        mappings_per_layer[layer_id].append(
                            (block.cpu_block_id, gpu_slot)
                        )

                    # Update block state
                    self.free_cpu_blocks.append(block.cpu_block_id)
                    del self.cpu_block_to_logical[block.cpu_block_id]

                    self.gpu_slot_to_logical[gpu_slot] = logical_id
                    block.location = BlockLocation.GPU
                    block.gpu_slot = gpu_slot
                    block.cpu_block_id = -1

                    self.pending_gpu_loads.add(logical_id)
                    self.policy.on_block_prefetched(gpu_slot, self.current_step)

                elif block.location == BlockLocation.GPU:
                    self.policy.on_block_access(block.gpu_slot, self.current_step)

            # Update gather indices (outside graph)
            self.offload_engine.update_gather_indices_all_layers(mappings_per_layer)
            self.offload_engine.sync_indices()

    def needs_chunked_decode(self, seq: Sequence) -> bool:
        """
        Check if sequence needs chunked decode.

        Returns True if there are blocks on CPU and total blocks exceed GPU capacity.
        """
        cpu_blocks = 0
        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.CPU:
                cpu_blocks += 1
        return cpu_blocks > 0 and len(seq.block_table) > self.num_gpu_slots

    def load_all_kv_for_layer(
        self,
        seq: Sequence,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load ALL KV for a sequence from both GPU and CPU for a layer.

        Used during chunked decode to compute full attention.

        Returns:
            (k, v) tensors with shape [1, total_tokens, kv_heads, head_dim]
        """
        k_chunks = []
        v_chunks = []

        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]

            if block.location == BlockLocation.GPU:
                # Get from GPU cache
                k, v = self.offload_engine.get_layer_cache(layer_id)
                # k, v shape: [num_gpu_blocks, block_size, kv_heads, head_dim]
                k_block = k[block.gpu_slot]  # [block_size, kv_heads, head_dim]
                v_block = v[block.gpu_slot]
                k_chunks.append(k_block)
                v_chunks.append(v_block)

            elif block.location == BlockLocation.CPU:
                # Get from CPU cache
                k_block, v_block = self.offload_engine.get_cpu_block(layer_id, block.cpu_block_id)
                # Already [block_size, kv_heads, head_dim]
                k_chunks.append(k_block.to("cuda", non_blocking=True))
                v_chunks.append(v_block.to("cuda", non_blocking=True))

        # Concatenate all chunks
        k_all = torch.cat(k_chunks, dim=0)  # [total_tokens, kv_heads, head_dim]
        v_all = torch.cat(v_chunks, dim=0)

        # Add batch dimension
        k_all = k_all.unsqueeze(0)  # [1, total_tokens, kv_heads, head_dim]
        v_all = v_all.unsqueeze(0)

        return k_all, v_all

    def get_gpu_block_tables(
        self,
        seqs: List[Sequence],
    ) -> List[List[int]]:
        """
        Get GPU slot tables for sequences.

        Returns GPU slot IDs, which may differ from logical block IDs.
        """
        result = []
        for seq in seqs:
            gpu_table = []
            for logical_id in seq.block_table:
                block = self.logical_blocks[logical_id]
                assert block.location == BlockLocation.GPU, (
                    f"Block {logical_id} not on GPU (location={block.location})"
                )
                gpu_table.append(block.gpu_slot)
            result.append(gpu_table)
        return result

    def post_attention_cleanup(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """
        Cleanup after attention.

        Clear pending loads and optionally proactive offload.
        """
        self.pending_gpu_loads.clear()

    # ========== Chunked Prefill Support ==========

    def needs_chunked_prefill(self, seq: Sequence) -> bool:
        """
        Check if sequence needs chunked prefill.

        Returns True if there are unprefilled blocks that are on CPU.
        This indicates we need to process in chunks because not all blocks fit on GPU.
        """
        for logical_id in seq.block_table:
            if logical_id not in self.prefilled_blocks:
                block = self.logical_blocks[logical_id]
                if block.location == BlockLocation.CPU:
                    return True
        return False

    def get_gpu_block_count(self, seq: Sequence) -> int:
        """Get number of blocks currently on GPU for this sequence."""
        count = 0
        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.GPU:
                count += 1
        return count

    def get_prefill_chunk_info(self, seq: Sequence) -> Tuple[int, int, List[int]]:
        """
        Get information for current prefill chunk.

        Returns:
            (start_block_idx, end_block_idx, gpu_block_ids)
            - start_block_idx: First block index in this chunk
            - end_block_idx: Last block index (exclusive) in this chunk
            - gpu_block_ids: GPU slot IDs for blocks in this chunk
        """
        start_idx = -1
        end_idx = -1
        gpu_block_ids = []

        for i, logical_id in enumerate(seq.block_table):
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.GPU:
                if start_idx == -1:
                    start_idx = i
                end_idx = i + 1
                gpu_block_ids.append(block.gpu_slot)
            elif start_idx != -1:
                # Found CPU block after GPU blocks - stop here
                break

        if start_idx == -1:
            return (0, 0, [])

        return (start_idx, end_idx, gpu_block_ids)

    def complete_prefill_chunk(self, seq: Sequence) -> bool:
        """
        Complete a prefill chunk: mark blocks as prefilled, offload to CPU, load next chunk.

        Returns:
            True if there are more chunks to process, False if done.
        """
        # Find blocks currently on GPU that were just prefilled
        gpu_blocks_to_offload = []
        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.GPU and logical_id not in self.prefilled_blocks:
                # Mark as prefilled
                self.prefilled_blocks.add(logical_id)
                gpu_blocks_to_offload.append(logical_id)

        # Offload prefilled GPU blocks to CPU
        for logical_id in gpu_blocks_to_offload:
            block = self.logical_blocks[logical_id]
            if not self.free_cpu_blocks:
                raise RuntimeError("No free CPU blocks for offload")

            cpu_block_id = self.free_cpu_blocks.popleft()

            # Async offload all layers
            for layer_id in range(self.offload_engine.num_layers):
                self.offload_engine.offload_block_async(
                    layer_id=layer_id,
                    gpu_block_id=block.gpu_slot,
                    cpu_block_id=cpu_block_id,
                )

            # Update mappings
            self.free_gpu_slots.append(block.gpu_slot)
            del self.gpu_slot_to_logical[block.gpu_slot]
            self.cpu_block_to_logical[cpu_block_id] = logical_id

            block.location = BlockLocation.CPU
            block.cpu_block_id = cpu_block_id
            block.gpu_slot = -1

        # Wait for offload to complete
        self.offload_engine.wait_all_transfers()

        # Find next UNPREFILLED CPU blocks and bring them to GPU
        cpu_blocks_to_load = []
        for logical_id in seq.block_table:
            if logical_id in self.prefilled_blocks:
                continue  # Skip already prefilled
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.CPU:
                if len(cpu_blocks_to_load) >= self.num_gpu_slots:
                    break  # GPU is full
                cpu_blocks_to_load.append(logical_id)

        if not cpu_blocks_to_load:
            return False  # All blocks have been prefilled

        # Load unprefilled CPU blocks to GPU
        for logical_id in cpu_blocks_to_load:
            block = self.logical_blocks[logical_id]
            gpu_slot = self.free_gpu_slots.popleft()

            # Note: We're NOT prefetching existing data - these blocks are being
            # loaded for the first time, so we just need to assign GPU slots
            # The model will write new KV cache data to these slots

            # Update mappings
            self.free_cpu_blocks.append(block.cpu_block_id)
            del self.cpu_block_to_logical[block.cpu_block_id]
            self.gpu_slot_to_logical[gpu_slot] = logical_id

            block.location = BlockLocation.GPU
            block.gpu_slot = gpu_slot
            block.cpu_block_id = -1

        return True  # More chunks to process

    def get_gpu_block_tables_partial(
        self,
        seqs: List[Sequence],
    ) -> List[Tuple[List[int], int, int]]:
        """
        Get GPU block tables for chunked prefill.

        Returns list of (gpu_block_ids, start_block_idx, end_block_idx) per sequence.
        Only includes blocks that are currently on GPU AND haven't been prefilled yet.
        """
        result = []
        for seq in seqs:
            gpu_table = []
            start_idx = -1
            end_idx = -1

            for i, logical_id in enumerate(seq.block_table):
                # Skip already prefilled blocks
                if logical_id in self.prefilled_blocks:
                    continue

                block = self.logical_blocks[logical_id]
                if block.location == BlockLocation.GPU:
                    if start_idx == -1:
                        start_idx = i
                    end_idx = i + 1
                    gpu_table.append(block.gpu_slot)
                elif start_idx != -1:
                    # Stop at first non-GPU block after GPU blocks
                    break

            if start_idx == -1:
                start_idx = 0
                end_idx = 0

            result.append((gpu_table, start_idx, end_idx))
        return result

    def get_prefilled_cpu_blocks(self, seq: Sequence) -> List[int]:
        """
        Get list of CPU block IDs for blocks that have been prefilled.

        Used for loading previous KV during chunked prefill.

        Returns:
            List of CPU block IDs in sequence order
        """
        cpu_blocks = []
        for logical_id in seq.block_table:
            if logical_id in self.prefilled_blocks:
                block = self.logical_blocks[logical_id]
                if block.location == BlockLocation.CPU:
                    cpu_blocks.append(block.cpu_block_id)
        return cpu_blocks

    def load_prev_kv_for_layer(
        self,
        seq: Sequence,
        layer_id: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load previous prefilled KV from CPU for a specific layer.

        This concatenates KV from all previously prefilled blocks for use
        during chunked prefill attention.

        Args:
            seq: Sequence to load KV for
            layer_id: Layer index

        Returns:
            (k, v) tensors with shape [1, total_prev_tokens, kv_heads, head_dim]
            or (None, None) if no previous KV exists
        """
        cpu_blocks = self.get_prefilled_cpu_blocks(seq)
        if not cpu_blocks:
            return None, None

        k_chunks = []
        v_chunks = []

        for cpu_block_id in cpu_blocks:
            k, v = self.offload_engine.get_cpu_block(layer_id, cpu_block_id)
            # k, v shape: [block_size, kv_heads, head_dim]
            k_chunks.append(k)
            v_chunks.append(v)

        # Concatenate all chunks
        k_prev = torch.cat(k_chunks, dim=0)  # [total_prev_tokens, kv_heads, head_dim]
        v_prev = torch.cat(v_chunks, dim=0)

        # Move to GPU and add batch dimension
        k_prev = k_prev.to("cuda", non_blocking=True).unsqueeze(0)  # [1, tokens, heads, dim]
        v_prev = v_prev.to("cuda", non_blocking=True).unsqueeze(0)

        return k_prev, v_prev

    def get_chunk_start_position(self, seq: Sequence) -> int:
        """
        Get the starting token position for the current chunk.

        This is the total number of tokens in previously prefilled blocks.

        Returns:
            Token position offset for current chunk
        """
        pos = 0
        for logical_id in seq.block_table:
            if logical_id in self.prefilled_blocks:
                # Full block's worth of tokens
                pos += self._block_size
            else:
                break
        return pos

    def __repr__(self) -> str:
        return (
            f"HybridKVCacheManager(\n"
            f"  num_gpu_slots={self.num_gpu_slots},\n"
            f"  num_cpu_blocks={self.num_cpu_blocks},\n"
            f"  block_size={self._block_size},\n"
            f"  free_logical={len(self.free_logical_ids)},\n"
            f"  free_gpu={len(self.free_gpu_slots)},\n"
            f"  free_cpu={len(self.free_cpu_blocks)},\n"
            f"  policy={self.policy}\n"
            f")"
        )
