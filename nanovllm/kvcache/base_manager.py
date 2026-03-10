"""
Abstract base class for KV cache managers.

This interface allows pluggable implementations:
- GPUOnlyManager: Pure GPU (current nano-vllm behavior)
- HybridKVCacheManager: CPU offload with CUDA Graph support
- Future: Disk offload, distributed cache, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from torch import Tensor

from nanovllm.engine.sequence import Sequence


class KVCacheManager(ABC):
    """
    Abstract base class for KV cache management strategies.

    A KVCacheManager handles:
    1. Physical memory allocation (GPU and optionally CPU)
    2. Logical block management (allocation, deallocation, prefix caching)
    3. Data transfer between devices (for hybrid managers)
    4. Integration with CUDA graphs

    Key design principles:
    - Sequences reference logical block IDs
    - Physical block IDs (GPU slots) may differ from logical IDs
    - CUDA Graph compatibility requires fixed tensor addresses
    """

    @property
    @abstractmethod
    def block_size(self) -> int:
        """Number of tokens per block."""
        pass

    @property
    @abstractmethod
    def num_free_blocks(self) -> int:
        """Number of free logical blocks available for allocation."""
        pass

    @abstractmethod
    def allocate_cache(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """
        Allocate KV cache storage.

        Called once during initialization to allocate GPU (and optionally CPU)
        memory for the KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value heads per layer
            head_dim: Dimension per head
            dtype: Data type for cache (e.g., torch.float16)
        """
        pass

    @abstractmethod
    def get_layer_cache(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        """
        Get K and V cache tensors for a specific layer.

        The returned tensors must be on GPU and have fixed addresses
        for CUDA Graph compatibility.

        Args:
            layer_id: Layer index

        Returns:
            (k_cache, v_cache) tensors
            Shape depends on implementation, typically:
            [num_blocks, block_size, kv_heads, head_dim]
        """
        pass

    @abstractmethod
    def can_allocate(self, seq: Sequence) -> bool:
        """
        Check if blocks can be allocated for a new sequence.

        Called before allocate() to ensure sufficient resources.

        Args:
            seq: Sequence to check

        Returns:
            True if allocation is possible
        """
        pass

    @abstractmethod
    def allocate(self, seq: Sequence) -> None:
        """
        Allocate blocks for a sequence during prefill.

        This method:
        1. Checks prefix cache for matching blocks
        2. Allocates new blocks as needed
        3. Updates seq.block_table with logical block IDs
        4. Updates seq.num_cached_tokens for prefix cache hits

        Args:
            seq: Sequence to allocate blocks for
        """
        pass

    @abstractmethod
    def deallocate(self, seq: Sequence) -> None:
        """
        Release blocks for a finished sequence.

        This method:
        1. Decrements reference counts
        2. Frees blocks with zero references
        3. Clears seq.block_table

        Args:
            seq: Sequence whose blocks to release
        """
        pass

    @abstractmethod
    def can_append(self, seq: Sequence) -> bool:
        """
        Check if a new block can be allocated for decode.

        Called before may_append() to check if resources are available.

        Args:
            seq: Sequence to check

        Returns:
            True if append is possible (or no new block needed)
        """
        pass

    @abstractmethod
    def may_append(self, seq: Sequence) -> None:
        """
        Potentially allocate a new block during decode.

        Called after each decode step. If the current block is full,
        allocates a new block and updates seq.block_table.

        Args:
            seq: Sequence that may need a new block
        """
        pass

    @abstractmethod
    def prepare_for_attention(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """
        Prepare KV cache for attention computation.

        For GPU-only managers: typically a no-op.
        For hybrid managers: ensures all needed blocks are on GPU,
        may trigger prefetching from CPU.

        Called before attention computation. For decode with CUDA graphs,
        this should update gather_indices but not perform actual transfers
        (transfers happen inside the graph).

        Args:
            seqs: Sequences that will be processed
            is_prefill: True for prefill phase, False for decode
        """
        pass

    @abstractmethod
    def get_gpu_block_tables(
        self,
        seqs: List[Sequence],
    ) -> List[List[int]]:
        """
        Get GPU physical block tables for sequences.

        For GPU-only managers: returns seq.block_table directly.
        For hybrid managers: returns GPU slot IDs (may differ from logical IDs).

        The returned block tables are used to compute slot_mapping
        in ModelRunner.prepare_prefill/decode.

        Args:
            seqs: Sequences to get block tables for

        Returns:
            List of GPU block tables, one per sequence
        """
        pass

    def post_attention_cleanup(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
    ) -> None:
        """
        Cleanup after attention computation.

        Optional hook for managers to perform post-attention tasks:
        - Offloading cold blocks to CPU
        - Updating access statistics
        - etc.

        Default implementation does nothing.

        Args:
            seqs: Sequences that were processed
            is_prefill: True for prefill phase, False for decode
        """
        pass

    def get_num_blocks_needed(self, num_tokens: int) -> int:
        """
        Calculate number of blocks needed for given token count.

        Args:
            num_tokens: Number of tokens

        Returns:
            Number of blocks needed
        """
        return (num_tokens + self.block_size - 1) // self.block_size

    @staticmethod
    def compute_hash(token_ids: list, prefix: int = -1) -> int:
        """
        Compute hash for prefix caching.

        Uses xxhash for fast hashing. The hash includes the prefix hash
        to create a chain of hashes for multi-block sequences.

        Args:
            token_ids: Token IDs in the block
            prefix: Hash of previous block, or -1 for first block

        Returns:
            Hash value
        """
        import xxhash
        import numpy as np

        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
