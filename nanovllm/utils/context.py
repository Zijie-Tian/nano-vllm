from dataclasses import dataclass, field
from typing import List, Tuple, Any
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    # Chunked prefill support
    is_chunked_prefill: bool = False
    # Previous KV chunks info: List of (start_pos, end_pos) for blocks on CPU
    prev_kv_ranges: List[Tuple[int, int]] = field(default_factory=list)
    # Current chunk's position offset (for causal mask)
    chunk_offset: int = 0
    # Reference to kvcache manager for loading previous KV (HybridKVCacheManager)
    kvcache_manager: Any = None
    # Current sequence being processed (for chunked prefill to load KV)
    chunked_seq: Any = None
    # Position within block for decode (used for reading from Decode region)
    decode_pos_in_block: int = 0
    # Starting position within block where decode tokens began (for accumulated token tracking)
    # Used when batching decode offloads - we need to attend to all accumulated tokens
    decode_start_pos_in_block: int = 0
    # Current chunk index for ring buffer pipeline (prefill only)
    current_chunk_idx: int = 0


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    is_chunked_prefill=False,
    prev_kv_ranges=None,
    chunk_offset=0,
    kvcache_manager=None,
    chunked_seq=None,
    decode_pos_in_block=0,
    decode_start_pos_in_block=0,
    current_chunk_idx=0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_chunked_prefill=is_chunked_prefill,
        prev_kv_ranges=prev_kv_ranges or [],
        chunk_offset=chunk_offset,
        kvcache_manager=kvcache_manager,
        chunked_seq=chunked_seq,
        decode_pos_in_block=decode_pos_in_block,
        decode_start_pos_in_block=decode_start_pos_in_block,
        current_chunk_idx=current_chunk_idx,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
