"""
GPU-only KV cache manager.

This is the default manager when CPU offload is disabled.
Refactored from the original block_manager.py to implement
the KVCacheManager interface.
"""

from collections import deque
from typing import List, Tuple, Dict, Optional
import torch
from torch import Tensor

from nanovllm.engine.sequence import Sequence
from nanovllm.kvcache.base_manager import KVCacheManager


class Block:
    """Physical block in GPU memory."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: List[int] = []

    def update(self, hash: int, token_ids: List[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class GPUOnlyManager(KVCacheManager):
    """
    Pure GPU KV cache manager.

    This is the default implementation when enable_cpu_offload=False.
    All KV cache resides in GPU memory.

    Features:
    - Paged attention with configurable block size
    - Prefix caching via xxhash
    - Reference counting for block sharing
    - Contiguous cache for single-sequence layer-wise prefill (optional)

    This manager is fully compatible with CUDA graphs since
    all data stays on GPU at fixed addresses.
    """

    def __init__(self, num_blocks: int, block_size: int, max_seq_len: int = 0):
        """
        Initialize GPU-only manager.

        Args:
            num_blocks: Total number of blocks to manage
            block_size: Tokens per block (default 256)
            max_seq_len: Max sequence length for contiguous cache (0 to disable)
        """
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._max_seq_len = max_seq_len

        # Block metadata
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]

        # Prefix cache: hash -> block_id
        self.hash_to_block_id: Dict[int, int] = {}

        # Free/used tracking
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

        # KV cache tensors (set by allocate_cache)
        self.kv_cache: Optional[Tensor] = None
        self.num_layers: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0

        # Contiguous cache for single-seq layer-wise prefill (set by allocate_cache)
        self.contiguous_k_cache: Optional[Tensor] = None
        self.contiguous_v_cache: Optional[Tensor] = None
        self.contiguous_seq_len: int = 0  # Current sequence length in contiguous cache

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def allocate_cache(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Allocate GPU KV cache tensor."""
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Shape: [2, num_layers, num_blocks, block_size, kv_heads, head_dim]
        # 2 for K and V
        self.kv_cache = torch.empty(
            2, num_layers, self._num_blocks, self._block_size,
            num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )

        # Allocate contiguous cache for single-seq layer-wise prefill
        # Only allocate if there's enough free memory (at least 2GB margin)
        if self._max_seq_len > 0:
            contiguous_cache_bytes = 2 * num_layers * self._max_seq_len * num_kv_heads * head_dim * dtype.itemsize
            free_memory = torch.cuda.mem_get_info()[0]

            if free_memory > contiguous_cache_bytes + 2 * 1024**3:  # 2GB margin
                # Shape: [num_layers, max_seq_len, kv_heads, head_dim]
                self.contiguous_k_cache = torch.empty(
                    num_layers, self._max_seq_len, num_kv_heads, head_dim,
                    dtype=dtype, device="cuda"
                )
                self.contiguous_v_cache = torch.empty(
                    num_layers, self._max_seq_len, num_kv_heads, head_dim,
                    dtype=dtype, device="cuda"
                )

    def get_layer_cache(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        """Get K/V cache for a layer."""
        assert self.kv_cache is not None, "Cache not allocated"
        return self.kv_cache[0, layer_id], self.kv_cache[1, layer_id]

    def _allocate_block(self, block_id: int) -> Block:
        """Internal: allocate a specific block."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is not free"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Internal: deallocate a block."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we have enough blocks for the sequence."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """
        Allocate blocks for a sequence during prefill.

        Implements prefix caching: if a block's content matches
        a previously cached block, reuse it instead of allocating new.
        """
        assert not seq.block_table, "Sequence already has blocks allocated"

        h = -1  # Hash chain
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            # Only compute hash for full blocks
            if len(token_ids) == self._block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1

            # Try prefix cache lookup
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                # Cache miss: allocate new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache hit: reuse existing block
                seq.num_cached_tokens += self._block_size
                if block_id in self.used_block_ids:
                    # Block is in use, increment ref count
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block was freed but hash still valid
                    block = self._allocate_block(block_id)

            # Update hash mapping for full blocks
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        """Release all blocks for a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a token (may need new block)."""
        # Need new block only if current position is at block boundary
        need_new_block = (len(seq) % self._block_size == 1)
        return len(self.free_block_ids) >= int(need_new_block)

    def may_append(self, seq: Sequence) -> None:
        """Handle potential new block allocation during decode."""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        seq_len = len(seq)
        pos_in_block = seq_len % self._block_size

        if pos_in_block == 1:
            # Just crossed into new block, need to allocate
            assert last_block.hash != -1, "Previous block should be complete"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif pos_in_block == 0:
            # Just filled a block, compute hash for prefix cache
            assert last_block.hash == -1, "Block should not have hash yet"
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:
            # Middle of block, nothing to do
            assert last_block.hash == -1

    def prepare_for_attention(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """
        No-op for GPU-only manager.

        All blocks are already on GPU, no preparation needed.
        """
        pass

    def get_gpu_block_tables(
        self,
        seqs: List[Sequence],
    ) -> List[List[int]]:
        """
        Return block tables directly (logical = physical for GPU-only).
        """
        return [list(seq.block_table) for seq in seqs]

    def post_attention_cleanup(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """No-op for GPU-only manager."""
        pass

    def __repr__(self) -> str:
        return (
            f"GPUOnlyManager("
            f"num_blocks={self._num_blocks}, "
            f"block_size={self._block_size}, "
            f"free={len(self.free_block_ids)}, "
            f"used={len(self.used_block_ids)}"
            f")"
        )
