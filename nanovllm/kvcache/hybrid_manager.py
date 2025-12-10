"""
Hybrid CPU-GPU KV cache manager with CUDA Graph support.

Key design for CUDA Graph compatibility:
1. GPU buffer has fixed addresses (allocated once)
2. CPU pool has fixed addresses (pinned memory)
3. gather_indices tensor has fixed address, variable content
4. H2D transfer uses gathered_copy kernel inside CUDA graphs
5. Graph replay only needs index updates (tiny overhead)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Set, Optional
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

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
        cpu_primary: bool = True,
        num_prefetch_blocks: int = 2,
    ):
        """
        Initialize hybrid manager.

        Args:
            num_gpu_slots: Number of GPU buffer slots (working set)
            num_cpu_blocks: Number of CPU pool blocks (overflow or primary storage)
            block_size: Tokens per block
            policy: Eviction policy (default: LRU)
            cpu_primary: If True, use CPU as primary storage with 三区域 GPU buffer.
                        If False, use GPU as primary with CPU as overflow (legacy mode).
            num_prefetch_blocks: Number of prefetch blocks for 三区域 GPU buffer design
        """
        self._block_size = block_size
        self.num_gpu_slots = num_gpu_slots
        self.num_cpu_blocks = num_cpu_blocks
        self.total_blocks = num_gpu_slots + num_cpu_blocks
        self.cpu_primary = cpu_primary  # 三区域 mode flag
        self.num_prefetch_blocks = num_prefetch_blocks  # 三区域设计参数

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
            num_prefetch_blocks=self.num_prefetch_blocks,
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

        In cpu_primary mode (Ping-Pong): All blocks are allocated to CPU.
        In legacy mode: Blocks are allocated to GPU when possible, overflow to CPU.
        """
        assert not seq.block_table, "Sequence already has blocks"

        # Ping-Pong模式：所有blocks都分配到CPU
        if self.cpu_primary:
            return self.allocate_cpu_only(seq)

        # Legacy模式：GPU为主，CPU为overflow
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

            if self.cpu_primary:
                # Ping-Pong模式：新block分配到CPU
                if not self.free_cpu_blocks:
                    raise RuntimeError("No free CPU blocks for decode")
                cpu_block_id = self.free_cpu_blocks.popleft()
                block.location = BlockLocation.CPU
                block.cpu_block_id = cpu_block_id
                block.gpu_slot = -1
                self.cpu_block_to_logical[cpu_block_id] = logical_id
            else:
                # Legacy模式：新block分配到GPU
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

    # ========== Chunked Decode Support ==========

    def get_decode_chunk_info(self, seq: Sequence) -> Tuple[List[int], List[int], int]:
        """
        Get information for chunked decode.

        Returns:
            (cpu_block_ids, cpu_logical_ids, num_chunks)
            - cpu_block_ids: List of CPU block IDs in sequence order
            - cpu_logical_ids: Corresponding logical block IDs
            - num_chunks: Number of chunks needed
        """
        cpu_block_ids = []
        cpu_logical_ids = []

        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.CPU:
                cpu_block_ids.append(block.cpu_block_id)
                cpu_logical_ids.append(logical_id)

        # Each chunk uses available GPU slots minus 1 (reserved for write block)
        usable_slots = self.num_gpu_slots - 1
        num_chunks = (len(cpu_block_ids) + usable_slots - 1) // usable_slots if usable_slots > 0 else 0

        return cpu_block_ids, cpu_logical_ids, num_chunks

    def load_decode_chunk(
        self,
        seq: Sequence,
        cpu_block_ids: List[int],
        cpu_logical_ids: List[int],
        chunk_idx: int,
    ) -> List[int]:
        """
        Load one chunk of CPU blocks to GPU for chunked decode.

        Similar to chunked prefill: uses GPU slots to hold a batch of blocks.

        Args:
            seq: Sequence being decoded
            cpu_block_ids: All CPU block IDs for this sequence
            cpu_logical_ids: Corresponding logical block IDs
            chunk_idx: Which chunk to load (0-indexed)

        Returns:
            List of GPU slot IDs where the chunk was loaded
        """
        chunk_size = self.num_gpu_slots
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(cpu_block_ids))

        chunk_cpu_ids = cpu_block_ids[start:end]
        chunk_logical_ids = cpu_logical_ids[start:end]

        # Use GPU slots 0, 1, 2, ... for this chunk
        gpu_slots = list(range(len(chunk_cpu_ids)))

        # Load all layers at once using offload_engine
        self.offload_engine.load_cpu_blocks_to_gpu_slots_all_layers(
            chunk_cpu_ids, gpu_slots
        )

        return gpu_slots

    def get_gpu_blocks_for_decode(self, seq: Sequence) -> Tuple[List[int], List[int]]:
        """
        Get blocks currently on GPU for this sequence.

        Returns:
            (gpu_slots, logical_ids) - GPU slot IDs and corresponding logical block IDs
        """
        gpu_slots = []
        logical_ids = []

        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.GPU:
                gpu_slots.append(block.gpu_slot)
                logical_ids.append(logical_id)

        return gpu_slots, logical_ids

    def get_kv_for_gpu_slots(
        self,
        layer_id: int,
        gpu_slots: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV tensors for specific GPU slots.

        Args:
            layer_id: Layer index
            gpu_slots: List of GPU slot IDs

        Returns:
            (k, v) tensors with shape [1, num_tokens, kv_heads, head_dim]
        """
        k_cache, v_cache = self.offload_engine.get_layer_cache(layer_id)
        # k_cache, v_cache shape: [num_gpu_blocks, block_size, kv_heads, head_dim]

        k_chunks = [k_cache[slot] for slot in gpu_slots]
        v_chunks = [v_cache[slot] for slot in gpu_slots]

        # Concatenate and add batch dimension
        k = torch.cat(k_chunks, dim=0).unsqueeze(0)  # [1, tokens, heads, dim]
        v = torch.cat(v_chunks, dim=0).unsqueeze(0)

        return k, v

    def ensure_last_block_on_gpu(self, seq: Sequence) -> int:
        """
        Ensure the last block is on GPU for writing new KV.

        Uses a RESERVED slot (last slot) to avoid conflicts with chunked decode
        which uses slots 0, 1, 2, ... for loading CPU blocks.

        Returns:
            GPU slot ID for the last block
        """
        last_logical_id = seq.block_table[-1]
        block = self.logical_blocks[last_logical_id]

        if block.location == BlockLocation.GPU:
            return block.gpu_slot

        # Use last slot as reserved slot for write block
        # This avoids conflicts with chunked decode which uses slots 0, 1, 2...
        reserved_slot = self.num_gpu_slots - 1

        # Load this block to GPU for all layers
        self.offload_engine.load_cpu_blocks_to_gpu_slots_all_layers(
            [block.cpu_block_id], [reserved_slot]
        )

        # Update block state
        self.free_cpu_blocks.append(block.cpu_block_id)
        del self.cpu_block_to_logical[block.cpu_block_id]

        self.gpu_slot_to_logical[reserved_slot] = last_logical_id
        block.location = BlockLocation.GPU
        block.gpu_slot = reserved_slot
        block.cpu_block_id = -1

        return reserved_slot

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
        logger.debug(
            f"get_prefilled_cpu_blocks: prefilled_blocks={list(self.prefilled_blocks)}, "
            f"returned cpu_blocks={cpu_blocks}"
        )
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

    # ========== Ping-Pong 双缓冲支持 ==========

    def allocate_cpu_only(self, seq: Sequence) -> None:
        """
        为序列分配 CPU blocks（用于 Ping-Pong 模式）。

        与 allocate() 不同，这里所有 blocks 都分配到 CPU，
        GPU 只用作工作缓冲区。

        Args:
            seq: 要分配的序列
        """
        assert not seq.block_table, "Sequence already has blocks"

        for i in range(seq.num_blocks):
            # 分配 CPU block
            if not self.free_cpu_blocks:
                raise RuntimeError(
                    f"No free CPU blocks. Need {seq.num_blocks}, "
                    f"available: {len(self.free_cpu_blocks)}"
                )

            cpu_block_id = self.free_cpu_blocks.popleft()

            # 分配逻辑块
            logical_id = self.free_logical_ids.popleft()
            block = self.logical_blocks[logical_id]
            block.ref_count = 1
            block.location = BlockLocation.CPU
            block.cpu_block_id = cpu_block_id
            block.gpu_slot = -1

            self.cpu_block_to_logical[cpu_block_id] = logical_id
            seq.block_table.append(logical_id)

    def get_cpu_block_table(self, seq: Sequence) -> List[int]:
        """
        获取序列的 CPU block ID 列表。

        Args:
            seq: 序列

        Returns:
            CPU block IDs 列表，按序列顺序
        """
        cpu_blocks = []
        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.CPU:
                cpu_blocks.append(block.cpu_block_id)
            else:
                # 如果 block 在 GPU 上，它应该有一个对应的 CPU block
                # 在 Ping-Pong 模式下，所有数据最终都在 CPU 上
                raise RuntimeError(
                    f"Block {logical_id} not on CPU (location={block.location}). "
                    f"In Ping-Pong mode, all blocks should be on CPU."
                )
        return cpu_blocks

    def get_all_cpu_blocks(self, seq: Sequence) -> Tuple[List[int], List[int]]:
        """
        获取序列的所有 CPU blocks 及其逻辑 ID。

        Args:
            seq: 序列

        Returns:
            (cpu_block_ids, logical_ids)
        """
        cpu_block_ids = []
        logical_ids = []
        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.CPU:
                cpu_block_ids.append(block.cpu_block_id)
                logical_ids.append(logical_id)
        return cpu_block_ids, logical_ids

    def allocate_next_cpu_block(self, seq: Sequence) -> int:
        """
        为序列分配下一个 CPU block（用于 decode 时新 token）。

        Args:
            seq: 序列

        Returns:
            新分配的 CPU block ID
        """
        if not self.free_cpu_blocks:
            raise RuntimeError("No free CPU blocks")

        cpu_block_id = self.free_cpu_blocks.popleft()
        logical_id = self.free_logical_ids.popleft()

        block = self.logical_blocks[logical_id]
        block.ref_count = 1
        block.location = BlockLocation.CPU
        block.cpu_block_id = cpu_block_id
        block.gpu_slot = -1

        self.cpu_block_to_logical[cpu_block_id] = logical_id
        seq.block_table.append(logical_id)

        return cpu_block_id

    def get_last_cpu_block(self, seq: Sequence) -> int:
        """
        获取序列最后一个 block 的 CPU block ID。

        如果最后一个 block 不在 CPU 上，返回 -1。

        Args:
            seq: 序列

        Returns:
            CPU block ID，如果不在 CPU 上则返回 -1
        """
        if not seq.block_table:
            return -1

        last_logical_id = seq.block_table[-1]
        block = self.logical_blocks[last_logical_id]

        if block.location == BlockLocation.CPU:
            return block.cpu_block_id
        return -1

    def get_write_slot_for_pingpong(self, seq: Sequence) -> int:
        """
        获取三区域 decode 时新 KV 写入的 GPU slot。

        在三区域设计中，永远使用 Decode区 (slot 0) 写入新 KV。
        这样可以避免与 Compute/Prefetch区 的加载操作冲突。

        Args:
            seq: 序列

        Returns:
            GPU slot ID (永远是 decode_slot = 0)
        """
        return self.offload_engine.decode_slot

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
