"""
Base class for sparse attention policies.

Sparse attention policies determine which KV cache blocks to load
from CPU for each query chunk during chunked attention computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, TYPE_CHECKING
import torch

# Import SparsePolicyType from config to avoid circular imports
from nanovllm.config import SparsePolicyType

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence


@dataclass
class PolicyContext:
    """
    Context passed to sparse policy for block selection.

    This dataclass contains all information needed by a sparse policy
    to decide which blocks to load for the current query chunk.
    """

    query_chunk_idx: int
    """Index of the current query chunk (0-indexed)."""

    num_query_chunks: int
    """Total number of query chunks in this prefill."""

    layer_id: int
    """Current transformer layer index."""

    query: Optional[torch.Tensor]
    """
    Query tensor for current chunk.
    Shape: [1, num_heads, head_dim] for decode, [seq_len, num_heads, head_dim] for prefill.
    Available for both prefill and decode phases.
    """

    is_prefill: bool
    """True if in prefill phase, False if in decode phase."""

    block_size: int = 1024
    """Number of tokens per block."""

    total_kv_len: int = 0
    """Total KV sequence length so far (for reference)."""


class SparsePolicy(ABC):
    """
    Abstract base class for sparse attention policies.

    Subclass this and implement select_blocks() to create custom
    sparse attention patterns. The policy receives context about
    the current query chunk and returns which KV blocks to load.

    Attributes:
        supports_prefill: Whether this policy can be used for prefill phase.
        supports_decode: Whether this policy can be used for decode phase.

    Example:
        class MySparsePolicy(SparsePolicy):
            supports_prefill = False  # decode-only policy
            supports_decode = True

            def select_blocks(self, available_blocks, ctx):
                # Load first block and last 2 blocks
                if len(available_blocks) <= 3:
                    return available_blocks
                return [available_blocks[0]] + available_blocks[-2:]
    """

    # Compatibility flags - override in subclasses
    supports_prefill: bool = True
    supports_decode: bool = True

    def initialize(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
        device: torch.device = None,
    ) -> None:
        """
        Initialize policy resources.

        Called by the framework after KV cache is allocated. Override this
        to create metadata structures (e.g., BlockMetadataManager for Quest).
        Default implementation does nothing.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension per head
            num_cpu_blocks: Number of CPU blocks allocated
            dtype: Data type for tensors
            device: Device for metadata storage (GPU recommended for performance)
        """
        pass

    def alloc_policy_metadata(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        enable_cpu_offload: bool = False,
    ) -> None:
        """
        Pre-allocate GPU buffers for policy computation.

        Called by the framework after KV cache allocation. Implementations should
        use enable_cpu_offload to decide which buffers to allocate:
        - Offload mode: allocate chunked prefill buffers (mask, KV chunking stats)
        - GPU-only mode: additionally allocate GQA expansion buffers

        This is separate from initialize() which is used for CPU offload metadata.

        Args:
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length (for buffer sizing)
            dtype: Data type (typically float16/bfloat16)
            device: Target device (cuda)
            enable_cpu_offload: Whether CPU offload is enabled
        """
        pass

    @abstractmethod
    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select which KV blocks to load for the current query chunk.

        This is the core method that defines the sparse attention pattern.
        The returned blocks will be loaded from CPU to GPU for attention
        computation against the current query chunk.

        Args:
            available_blocks: List of CPU block IDs that contain KV cache
                             from previous chunks. These are ordered by
                             their position in the sequence.
            offload_engine: OffloadEngine for loading KV (some policies need
                           to load KV to make selection decisions).
            ctx: PolicyContext with information about the current query
                 chunk, layer, phase (prefill/decode), etc.
            q: Query tensor [seq_len, num_heads, head_dim] for current chunk
            k: Key tensor [seq_len, num_kv_heads, head_dim] for current chunk

        Returns:
            List of block IDs to load (must be a subset of available_blocks).
            The order may affect performance (sequential access is faster).
            Returning [] means no previous blocks will be loaded.
        """
        pass

    def on_prefill_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """
        Hook called when a block is offloaded during prefill phase.

        Called BEFORE GPU→CPU copy, while k_cache is still on GPU.
        Override this to collect metadata about blocks (e.g., min/max keys
        for Quest-style selection). Default implementation does nothing.

        Args:
            cpu_block_id: The CPU block ID that will be written
            layer_id: Transformer layer index
            k_cache: Key cache tensor [block_size, num_kv_heads, head_dim] (on GPU)
            num_valid_tokens: Number of valid tokens in this block
        """
        pass

    def on_decode_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """
        Hook called when a block is offloaded during decode phase.

        Called BEFORE GPU→CPU copy, while k_cache is still on GPU.
        Override this to update metadata about blocks. Default implementation
        does nothing.

        Args:
            cpu_block_id: The CPU block ID that will be written
            layer_id: Transformer layer index
            k_cache: Key cache tensor [block_size, num_kv_heads, head_dim] (on GPU)
            num_valid_tokens: Number of valid tokens in this block
        """
        pass

    def reset(self) -> None:
        """
        Reset policy state.

        Called when starting a new sequence or clearing state.
        Default implementation does nothing.
        """
        pass

    # =========================================================================
    # GPU-only methods (non-chunked)
    # These methods are used when all KV cache is on GPU, no CPU offload needed.
    # =========================================================================

    def compute_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        layer_id: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute GPU-only prefill attention (non-chunked).

        This method is used when all KV cache resides on GPU (no CPU offload).
        Override this to implement sparse prefill attention for GPU-only mode.
        Default implementation raises NotImplementedError.

        Args:
            q: [total_q, num_heads, head_dim] query tensor (packed variable length)
            k: [total_kv, num_kv_heads, head_dim] key tensor
            v: [total_kv, num_kv_heads, head_dim] value tensor
            cu_seqlens_q: [batch+1] cumulative sequence lengths for queries
            cu_seqlens_k: [batch+1] cumulative sequence lengths for keys
            max_seqlen_q: maximum query sequence length
            max_seqlen_k: maximum key sequence length
            softmax_scale: softmax scaling factor (1/sqrt(head_dim))
            layer_id: transformer layer index
            block_tables: [batch, max_blocks] paged attention block tables (optional)

        Returns:
            [total_q, num_heads, head_dim] attention output
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_prefill for GPU-only mode"
        )

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute GPU-only decode attention (non-chunked).

        This method is used when all KV cache resides on GPU (no CPU offload).
        Override this to implement sparse decode attention for GPU-only mode.
        Default implementation raises NotImplementedError.

        Args:
            q: [batch, num_heads, head_dim] query tensor (single token per sequence)
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim] paged key cache
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim] paged value cache
            cache_seqlens: [batch] sequence lengths in cache
            softmax_scale: softmax scaling factor (1/sqrt(head_dim))
            layer_id: transformer layer index
            block_tables: [batch, max_blocks] paged attention block tables (optional)

        Returns:
            [batch, 1, num_heads, head_dim] attention output
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_decode for GPU-only mode"
        )

    # =========================================================================
    # Chunked offload methods (for CPU offload mode)
    # =========================================================================

    @abstractmethod
    def compute_chunked_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        current_chunk_idx: int,
        seq: "Sequence",
        num_tokens: int,
        selected_blocks: List[int],
    ) -> torch.Tensor:
        """
        Compute chunked prefill attention (complete flow).

        This is the main entry point for prefill attention computation.
        It defines the complete prefill flow:
        1. Load and compute historical blocks via offload_engine (using selected_blocks)
        2. Get current chunk KV from offload_engine, compute attention
        3. Merge all results

        Note: Block selection (select_blocks) is called by the caller (attention.py)
        before invoking this method. The selected_blocks parameter contains the
        filtered block IDs to process.

        Args:
            q: [seq_len, num_heads, head_dim] query for current chunk
            k: [seq_len, num_kv_heads, head_dim] key for current chunk (in prefill buffer)
            v: [seq_len, num_kv_heads, head_dim] value for current chunk (in prefill buffer)
            layer_id: transformer layer index
            softmax_scale: softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            current_chunk_idx: current chunk index
            seq: Sequence object
            num_tokens: number of tokens in current chunk
            selected_blocks: list of CPU block IDs to process (already filtered by select_blocks)

        Returns:
            [seq_len, num_heads, head_dim] final attention output
        """
        pass

    @abstractmethod
    def compute_chunked_decode(
        self,
        q: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        seq: "Sequence",
        selected_blocks: List[int],
    ) -> torch.Tensor:
        """
        Compute chunked decode attention (complete flow).

        This is the main entry point for decode attention computation.
        It defines the complete decode flow:
        1. Load blocks via pipeline using selected_blocks (ring buffer or cross-layer)
        2. Read accumulated decode tokens from decode buffer
        3. Merge all results

        Note: Block selection (select_blocks) is called by the caller (attention.py)
        before invoking this method. The selected_blocks parameter contains the
        filtered block IDs to process.

        The decode position information can be computed internally:
        - decode_start_pos = kvcache_manager.get_decode_start_pos(seq)
        - decode_pos_in_block = (len(seq) - 1) % kvcache_manager.block_size

        Args:
            q: [batch_size, num_heads, head_dim] query for decode token
            layer_id: transformer layer index
            softmax_scale: softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            seq: Sequence object
            selected_blocks: list of CPU block IDs to process (already filtered by select_blocks)

        Returns:
            [batch_size, 1, num_heads, head_dim] final attention output
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
