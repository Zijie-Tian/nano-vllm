"""
PostRoPE full-attention policy.

This policy is an explicit named alias for the existing FULL semantics where
RoPE is applied in the model layer before Attention and the KV cache stores the
post-RoPE key states.
"""

from typing import TYPE_CHECKING, List, Optional

import torch

from .full_policy import FullAttentionPolicy
from .policy import SparsePolicy

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.kvcache.offload_engine import OffloadEngine


class PostRoPEPolicy(SparsePolicy):
    """Explicit post-RoPE full-attention policy."""

    supports_prefill = True
    supports_decode = True
    apply_rope_in_attention = False

    def __init__(self):
        self._delegate = FullAttentionPolicy()

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        return self._delegate.select_blocks(
            available_blocks=available_blocks,
            offload_engine=offload_engine,
            ctx=ctx,
            q=q,
            k=k,
        )

    def reset_stats(self) -> None:
        self._delegate.reset_stats()

    def get_density_stats(self) -> dict:
        return self._delegate.get_density_stats()

    def print_density_stats(self) -> None:
        self._delegate.print_density_stats()

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
        return self._delegate.compute_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            layer_id=layer_id,
            block_tables=block_tables,
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
        return self._delegate.compute_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            layer_id=layer_id,
            block_tables=block_tables,
        )

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
        return self._delegate.compute_chunked_prefill(
            q=q,
            k=k,
            v=v,
            layer_id=layer_id,
            softmax_scale=softmax_scale,
            offload_engine=offload_engine,
            kvcache_manager=kvcache_manager,
            current_chunk_idx=current_chunk_idx,
            seq=seq,
            num_tokens=num_tokens,
            selected_blocks=selected_blocks,
        )

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
        return self._delegate.compute_chunked_decode(
            q=q,
            layer_id=layer_id,
            softmax_scale=softmax_scale,
            offload_engine=offload_engine,
            kvcache_manager=kvcache_manager,
            seq=seq,
            selected_blocks=selected_blocks,
        )

    def __repr__(self) -> str:
        return "PostRoPEPolicy()"
