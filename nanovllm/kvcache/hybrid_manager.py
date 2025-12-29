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

# Type checking import for sparse policy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanovllm.kvcache.sparse.policy import SparsePolicy


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
    Hybrid CPU-GPU KV cache manager with ring buffer design.

    Architecture (CPU-primary mode):
    - CPU pool: Primary storage for all KV cache (num_cpu_blocks)
    - GPU buffer: Ring buffer for computation (num_gpu_slots)
    - Logical blocks: What sequences reference (num_gpu_slots + num_cpu_blocks)

    Design:
    - All KV cache is stored on CPU as primary storage
    - GPU is used as a ring buffer for computation only
    - During prefill: KV is written to GPU ring slot, then offloaded to CPU
    - During decode: Previous KV is loaded from CPU to GPU for attention
    - Ring buffer enables pipelined H2D transfers overlapped with computation
    """

    def __init__(
        self,
        num_gpu_slots: int,
        num_cpu_blocks: int,
        block_size: int,
        policy: Optional[EvictionPolicy] = None,
    ):
        """
        Initialize hybrid manager with CPU-primary ring buffer design.

        All KV cache is stored on CPU as primary storage. GPU slots are used
        as a ring buffer for computation only.

        Args:
            num_gpu_slots: Number of GPU buffer slots (ring buffer for computation)
            num_cpu_blocks: Number of CPU pool blocks (primary storage)
            block_size: Tokens per block
            policy: Eviction policy (default: LRU, used for prefix cache management)
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

        # Track decode starting position within block (for batched offload optimization)
        # Key: sequence id, Value: starting position where decode began in current block
        self._decode_start_pos: Dict[int, int] = {}

        # Sparse attention policy (optional)
        self.sparse_policy: Optional["SparsePolicy"] = None

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

    def set_sparse_policy(self, policy: "SparsePolicy") -> None:
        """
        Set sparse attention policy for block selection.

        The sparse policy determines which KV blocks to load from CPU
        for each query chunk during chunked attention computation.

        Args:
            policy: SparsePolicy instance (e.g., VerticalSlashPolicy, QuestPolicy)

        Example:
            from nanovllm.kvcache.sparse import VerticalSlashPolicy, VerticalSlashConfig
            policy = VerticalSlashPolicy(VerticalSlashConfig(num_sink_blocks=2))
            manager.set_sparse_policy(policy)
        """
        self.sparse_policy = policy
        logger.info(f"Sparse attention policy set: {policy}")

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate blocks for a new sequence."""
        return len(self.free_logical_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """
        Allocate logical blocks for prefill.

        All blocks are allocated to CPU (primary storage).
        GPU is used as ring buffer for computation only.
        """
        return self.allocate_cpu_only(seq)

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

            # Allocate new block to CPU (ring buffer mode)
            if not self.free_cpu_blocks:
                raise RuntimeError("No free CPU blocks for decode")
            cpu_block_id = self.free_cpu_blocks.popleft()
            block.location = BlockLocation.CPU
            block.cpu_block_id = cpu_block_id
            block.gpu_slot = -1
            self.cpu_block_to_logical[cpu_block_id] = logical_id

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

        In ring buffer mode, this is a no-op because chunked offload
        paths handle H2D transfers directly in the attention layer.
        """
        pass

    def get_gpu_block_tables(
        self,
        seqs: List[Sequence],
    ) -> List[List[int]]:
        """
        Get GPU slot tables for sequences.

        In ring buffer mode, all blocks are on CPU, so this raises an error
        if called. Use run_chunked_offload_* methods instead.
        """
        raise RuntimeError(
            "get_gpu_block_tables should not be called in ring buffer mode. "
            "Use run_chunked_offload_prefill/decode instead."
        )

    def post_attention_cleanup(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """
        Cleanup after attention.

        In ring buffer mode, this is a no-op because offload is handled
        directly in the chunked prefill/decode paths.
        """
        pass

    # ========== Ring Buffer CPU-primary Chunked Prefill Support ==========

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
        # logger.debug(
        #     f"get_prefilled_cpu_blocks: prefilled_blocks={list(self.prefilled_blocks)}, "
        #     f"returned cpu_blocks={cpu_blocks}"
        # )
        return cpu_blocks

    # ========== Ring Buffer CPU-primary support ==========

    def allocate_cpu_only(self, seq: Sequence) -> None:
        """
        Allocate CPU blocks for sequence (for ring buffer mode).

        Unlike allocate(), here all blocks are allocated to CPU,
        GPU is only used as ring buffer for computation.

        Args:
            seq: Sequence to allocate
        """
        assert not seq.block_table, "Sequence already has blocks"

        h = -1  # Running hash for prefix cache

        for i in range(seq.num_blocks):
            # Allocate CPU block
            if not self.free_cpu_blocks:
                raise RuntimeError(
                    f"No free CPU blocks. Need {seq.num_blocks}, "
                    f"available: {len(self.free_cpu_blocks)}"
                )

            cpu_block_id = self.free_cpu_blocks.popleft()

            # Get token IDs for this block and compute hash
            token_ids = seq.block(i)
            if len(token_ids) == self._block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1  # Incomplete block

            # Allocate logical block
            logical_id = self.free_logical_ids.popleft()
            block = self.logical_blocks[logical_id]
            block.ref_count = 1
            block.hash = h
            block.token_ids = token_ids.copy() if len(token_ids) == self._block_size else []
            block.location = BlockLocation.CPU
            block.cpu_block_id = cpu_block_id
            block.gpu_slot = -1

            self.cpu_block_to_logical[cpu_block_id] = logical_id
            seq.block_table.append(logical_id)

            # Update prefix cache
            if h != -1:
                self.hash_to_logical_id[h] = logical_id

    def get_cpu_block_table(self, seq: Sequence) -> List[int]:
        """
        Get CPU block ID list for sequence.

        Args:
            seq: Sequence

        Returns:
            List of CPU block IDs in sequence order
        """
        cpu_blocks = []
        for logical_id in seq.block_table:
            block = self.logical_blocks[logical_id]
            if block.location == BlockLocation.CPU:
                cpu_blocks.append(block.cpu_block_id)
            else:
                # If block is on GPU, it should have a corresponding CPU block
                # In ring buffer mode, all data ultimately resides on CPU
                raise RuntimeError(
                    f"Block {logical_id} not on CPU (location={block.location}). "
                    f"In ring buffer mode, all blocks should be on CPU."
                )
        return cpu_blocks

    def get_all_cpu_blocks(self, seq: Sequence) -> Tuple[List[int], List[int]]:
        """
        Get all CPU blocks and their logical IDs for sequence.

        Args:
            seq: Sequence

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
        Allocate next CPU block for sequence (for new token during decode).

        Args:
            seq: Sequence

        Returns:
            Newly allocated CPU block ID
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
        Get CPU block ID of the last block in sequence.

        Returns -1 if the last block is not on CPU.

        Args:
            seq: Sequence

        Returns:
            CPU block ID, or -1 if not on CPU
        """
        if not seq.block_table:
            return -1

        last_logical_id = seq.block_table[-1]
        block = self.logical_blocks[last_logical_id]

        if block.location == BlockLocation.CPU:
            return block.cpu_block_id
        return -1

    def get_write_slot_for_chunked_offload(self, seq: Sequence) -> int:
        """
        Get GPU slot for writing new KV during chunked offload decode.

        In ring buffer design, always use decode_slot (slot[0]) to write new KV.
        This avoids conflicts with loading operations which use slots[1:].

        Args:
            seq: Sequence

        Returns:
            GPU slot ID (always decode_slot = 0)
        """
        return self.offload_engine.decode_slot

    def get_decode_start_pos(self, seq: Sequence) -> int:
        """
        Get the starting position within block where decode tokens began.

        This is used for batched offload optimization - we need to attend to all
        accumulated tokens in decode slot, not just the current one.

        Args:
            seq: Sequence

        Returns:
            Starting position within block (0 to block_size-1)
        """
        seq_id = id(seq)
        if seq_id not in self._decode_start_pos:
            # First decode step - compute starting position
            # After prefill, the last block has some tokens filled
            # Decode starts at the next position
            prefill_len = len(seq) - 1  # Current len includes the new decode token
            self._decode_start_pos[seq_id] = prefill_len % self._block_size
        return self._decode_start_pos[seq_id]

    def reset_decode_start_pos(self, seq: Sequence) -> None:
        """
        Reset decode start position for sequence.

        Called when block is full and offloaded - next decode starts at position 0.

        Args:
            seq: Sequence
        """
        seq_id = id(seq)
        self._decode_start_pos[seq_id] = 0

    def clear_decode_tracking(self, seq: Sequence) -> None:
        """
        Clear decode position tracking for sequence.

        Called when sequence is deallocated.

        Args:
            seq: Sequence
        """
        seq_id = id(seq)
        self._decode_start_pos.pop(seq_id, None)

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
