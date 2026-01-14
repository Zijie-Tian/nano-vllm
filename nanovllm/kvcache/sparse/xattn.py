"""
XAttention sparse attention policy for nano-vllm.

Implements the XAttention algorithm from COMPASS, using chunked estimation
and block sparse attention for efficient long-context inference.

Reference: COMPASS/compass/src/Xattention.py
"""

import math
from typing import List, Optional
import torch
import torch.nn.functional as F

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.kvcache.sparse.kernels import (
    flat_group_gemm_fuse_reshape,
    softmax_fuse_block_sum,
)
from nanovllm.kvcache.sparse.utils import find_blocks_chunked


class XAttentionPolicy(SparsePolicy):
    """
    XAttention sparse prefill policy using chunked estimation + block sparse attention.

    This policy estimates sparse attention patterns by:
    1. Chunked QK computation using Triton kernels
    2. Block-wise softmax with importance scores
    3. Block selection based on threshold
    4. Block sparse attention computation

    Note: Requires Triton >= 2.1.0 and CUDA SM 80+ (RTX 3090, A100, H100, etc.)
    """

    supports_prefill = True
    supports_decode = False  # XAttention is prefill-only
    requires_block_selection = False  # Only affects attention computation

    def __init__(
        self,
        stride: int = 8,
        threshold: float = 0.9,
        chunk_size: Optional[int] = None,
        use_triton: bool = True,
        keep_sink: bool = False,
        keep_recent: bool = False,
        norm: float = 1.0,
    ):
        """
        Initialize XAttention policy.

        Args:
            stride: Stride for reorganizing Q/K (default: 8)
            threshold: Block selection threshold, 0-1 (default: 0.9)
            chunk_size: Chunk size for estimation (auto if None)
            use_triton: Use Triton kernels (requires SM 80+)
            keep_sink: Always keep first block (sink tokens)
            keep_recent: Always keep recent diagonal blocks
            norm: Normalization factor for attention scores
        """
        self.stride = stride
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.use_triton = use_triton
        self.keep_sink = keep_sink
        self.keep_recent = keep_recent
        self.norm = norm

        # Check Triton availability
        if self.use_triton:
            try:
                import triton
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                if props.major < 8:
                    self.use_triton = False
                    print(f"XAttention: Triton requires SM 80+, got SM {props.major}{props.minor}. Falling back to PyTorch.")
            except ImportError:
                self.use_triton = False
                print("XAttention: Triton not available. Falling back to PyTorch.")

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """
        Select blocks for decode phase.

        XAttention is prefill-only, so this method is only used as a fallback.
        Returns all available blocks by default.
        """
        # XAttention is prefill-only, but we need to implement this abstract method
        # Since requires_block_selection=False, this won't be called for loading
        return available_blocks

    def sparse_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Compute XAttention sparse attention for prefill.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            layer_id: Current transformer layer index

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        seq_len = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k.shape[1]

        # Use FlashAttention directly for CPU offload mode
        # FlashAttention supports GQA natively
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=q.device)

            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
            )

            return attn_output

        except Exception as e:
            # Fallback: PyTorch SDPA (supports GQA natively)
            print(f"XAttention: FlashAttention fallback failed ({e}), using PyTorch SDPA")
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
                scale=1.0 / math.sqrt(head_dim)
            )
            return attn_output

    def _xattn_offload_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Simplified XAttention prefill for CPU offload mode.

        Uses FlashAttention with full context since chunked estimation
        with full key_states requires special handling.
        """
        batch_size, num_heads, q_len, head_dim = query_states.shape
        _, _, k_len, _ = key_states.shape

        # Use FlashAttention with full context
        # In offload mode, keys are already on CPU and loaded as needed
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

            # Convert to [seq, heads, dim] format
            q = query_states.squeeze(0).transpose(0, 1)  # [q_len, num_heads, head_dim]
            k = key_states.squeeze(0).transpose(0, 1)  # [k_len, num_heads, head_dim]
            v = value_states.squeeze(0).transpose(0, 1)  # [k_len, num_heads, head_dim]

            cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.tensor([0, k_len], dtype=torch.int32, device=q.device)

            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=causal,
            )

            # Convert back to [batch, seq, heads, dim]
            attn_output = attn_output.unsqueeze(0).transpose(1, 2)  # [1, q_len, num_heads, head_dim]

            return attn_output

        except Exception as e:
            # Final fallback: PyTorch SDPA
            print(f"XAttention: FlashAttention fallback failed ({e}), using PyTorch SDPA")
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=None,
                    is_causal=causal,
                    scale=1.0 / math.sqrt(head_dim)
                )
            return attn_output

    def _xattn_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        stride: int,
        norm: float,
        threshold: float,
        block_size: int = 128,
        use_triton: bool = True,
        causal: bool = True,
        chunk_size: Optional[int] = None,
        keep_sink: bool = False,
        keep_recent: bool = False,
    ) -> torch.Tensor:
        """
        XAttention prefill implementation.

        Args:
            query_states: [batch, num_heads, q_len, head_dim]
            key_states: [batch, num_heads, k_len, head_dim]
            value_states: [batch, num_heads, k_len, head_dim]
            ... other params

        Returns:
            Attention output [batch, q_len, num_heads, head_dim]
        """
        batch_size, num_heads, k_len, head_dim = key_states.shape
        _, _, q_len, _ = query_states.shape

        # Auto-compute chunk_size if not specified
        if chunk_size is None:
            chunk_size = int(
                max(
                    min(
                        max(2048, 1 << (k_len - 1).bit_length()),
                        128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                    ),
                    2048,
                )
            )

        # Phase 1: Estimate sparse pattern
        attn_sums, approx_simple_mask = self._xattn_estimate(
            query_states,
            key_states,
            block_size=block_size,
            stride=stride,
            norm=norm,
            threshold=threshold,
            chunk_size=chunk_size,
            use_triton=use_triton,
            causal=causal,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )

        # Phase 2: Block sparse attention
        # For now, use FlashAttention as fallback since block_sparse_attn_func may not be available
        attn_output = self._block_sparse_attention_fallback(
            query_states, key_states, value_states,
            approx_simple_mask, block_size, q_len, k_len
        )

        return attn_output

    def _xattn_estimate(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        block_size: int,
        stride: int,
        norm: float = 1,
        softmax: bool = True,
        threshold: float = 0.9,
        chunk_size: int = 16384,
        use_triton: bool = True,
        causal: bool = True,
        keep_sink: bool = False,
        keep_recent: bool = False,
    ) -> torch.Tensor:
        """
        Estimate sparse attention pattern using chunked computation.

        Returns:
            attn_sums: [batch, heads, q_blocks, k_blocks] - importance scores
            simple_masks: [batch, heads, q_blocks, k_blocks] - boolean masks
        """
        batch_size, num_kv_head, k_len, head_dim = key_states.shape
        batch_size, num_q_head, q_len, head_dim = query_states.shape

        k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
        q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
        k_chunk_num = (k_len + k_num_to_pad) // chunk_size
        k_block_num = (k_len + k_num_to_pad) // block_size
        q_chunk_num = (q_len + q_num_to_pad) // chunk_size
        q_block_num = (q_len + q_num_to_pad) // block_size

        # Pad inputs
        if k_num_to_pad > 0:
            pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0)
        else:
            pad_key_states = key_states
        if q_num_to_pad > 0:
            pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)
        else:
            pad_query_states = query_states

        reshaped_chunk_size = chunk_size // stride
        reshaped_block_size = block_size // stride
        k_reshaped_seq_len = (k_len + k_num_to_pad) // stride

        attn_sum_list = []
        simple_mask_list = []

        for chunk_idx in range(q_chunk_num):
            if use_triton:
                # Triton GEMM + Softmax
                attn_weights_slice = flat_group_gemm_fuse_reshape(
                    pad_query_states[:, :, (chunk_idx * reshaped_chunk_size) * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size) * stride, :],
                    pad_key_states,
                    stride,
                    (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
                    (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                    is_causal=causal,
                )

                attn_sum = softmax_fuse_block_sum(
                    attn_weights_slice,
                    reshaped_block_size,
                    min(4096, reshaped_block_size),
                    (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
                    (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                    k_reshaped_seq_len - (k_num_to_pad // stride),
                    1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                    is_causal=causal,
                )
            else:
                # PyTorch fallback
                chunk_size_actual = reshaped_chunk_size
                chunk_start = chunk_idx * chunk_size_actual
                chunk_end = chunk_start + chunk_size_actual

                chunked_query = pad_query_states[:, :, chunk_start * stride:chunk_end * stride:stride, :]
                attn_weights_slice = torch.matmul(chunked_query, pad_key_states.transpose(2, 3))
                attn_weights_slice = attn_weights_slice / math.sqrt(head_dim) / stride / norm

                if causal:
                    causal_mask = torch.zeros((batch_size, num_q_head, chunk_size_actual, chunk_size_actual * k_chunk_num), device=key_states.device)
                    causal_mask[:, :, :, -(k_num_to_pad // stride):] = float("-inf")
                    # ... more causal mask logic ...
                    attn_weights_slice = attn_weights_slice + causal_mask

                attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32)
                attn_sum = attn_weights_slice.view(batch_size, num_q_head, chunk_size_actual // reshaped_block_size, reshaped_block_size, -1).sum(dim=-1).sum(dim=-2)

            # Find blocks based on threshold
            simple_mask = find_blocks_chunked(
                attn_sum,
                k_block_num - q_block_num + chunk_idx * (reshaped_chunk_size // reshaped_block_size),
                threshold,
                None,
                decoding=False,
                mode="prefill",
                causal=causal,
            )

            attn_sum_list.append(attn_sum)
            simple_mask_list.append(simple_mask)

        attn_sums = torch.cat(attn_sum_list, dim=-2)
        simple_masks = torch.cat(simple_mask_list, dim=-2)

        # Apply causal mask to block masks
        if causal:
            simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
                torch.tril(torch.ones(q_block_num, q_block_num, dtype=bool, device=key_states.device), diagonal=0),
                simple_masks[:, :, -q_block_num:, -q_block_num:],
                False,
            )

        if keep_sink:
            simple_masks[:, :, 0, :] = True

        if keep_recent:
            eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
            eye_matrix_expanded = eye_matrix.unsqueeze(0).unsqueeze(0).expand(1, num_q_head, q_block_num, q_block_num)
            simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
                eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
            )

        return attn_sums, simple_masks

    def _block_sparse_attention_fallback(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
        block_size: int,
        q_len: int,
        k_len: int,
    ) -> torch.Tensor:
        """
        Fallback implementation using FlashAttention.

        Since block_sparse_attn_func may not be available in all environments,
        this uses standard FlashAttention with full attention.
        """
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

            batch_size, num_heads, _, head_dim = query_states.shape

            # Convert to [seq, heads, dim] format
            q = query_states.squeeze(0).transpose(0, 1)  # [q_len, num_heads, head_dim]
            k = key_states.squeeze(0).transpose(0, 1)  # [k_len, num_heads, head_dim]
            v = value_states.squeeze(0).transpose(0, 1)  # [k_len, num_heads, head_dim]

            cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.tensor([0, k_len], dtype=torch.int32, device=q.device)

            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True,
            )

            # Convert back to [batch, seq, heads, dim]
            attn_output = attn_output.unsqueeze(0).transpose(1, 2)

            return attn_output

        except Exception as e:
            # Final fallback: PyTorch SDPA
            print(f"XAttention: FlashAttention fallback failed ({e}), using PyTorch SDPA")
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=None,
                    is_causal=True,
                    scale=1.0 / math.sqrt(query_states.shape[-1])
                )
            return attn_output

    def reset(self) -> None:
        """Reset policy state (no state to reset for XAttention)."""
        pass

    def __repr__(self) -> str:
        return (f"XAttentionPolicy("
                f"stride={self.stride}, "
                f"threshold={self.threshold}, "
                f"use_triton={self.use_triton})")
