"""
XAttention Block Sparse Attention (BSA) Policy for nano-vllm.

This module implements XAttention-inspired block sparse attention for chunked prefill,
using block-level estimation to select important KV blocks for computation.

Reference: COMPASS/compass/src/Xattention.py
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from nanovllm.kvcache.sparse.policy import SparsePolicy, PolicyContext
from nanovllm.utils.context import get_context


class XAttentionBSAPolicy(SparsePolicy):
    """
    XAttention Block Sparse Attention policy for chunked prefill.

    This policy uses block-level estimation to determine which KV blocks
    are important for the current chunk's queries, enabling sparse computation.

    Key features:
    - Double-loading design: estimate phase loads samples, compute phase loads selected blocks
    - Block-level granularity: 128-token blocks for estimation and computation
    - Triton kernels for efficient estimation (optional, falls back to PyTorch)

    Architecture:
        1. Estimate Phase: Load samples from all historical chunks, compute importance scores
        2. Selection Phase: Select top chunks by cumulative attention threshold
        3. Compute Phase: Load selected chunks fully, apply block sparse attention
    """

    supports_prefill = True
    supports_decode = False  # BSA is prefill-only
    requires_block_selection = False  # Selection happens at chunk level, not block level

    def __init__(
        self,
        block_size: int = 128,
        samples_per_chunk: int = 128,
        threshold: float = 0.9,
        use_triton: bool = True,
        stride: int = 8,
    ):
        """
        Initialize XAttention BSA policy.

        Args:
            block_size: Number of tokens per block (default: 128)
            samples_per_chunk: Number of tokens to sample from each historical chunk for estimation
            threshold: Cumulative attention threshold for chunk selection (0-1)
            use_triton: Use Triton kernels for estimation (requires SM 80+)
            stride: Stride for Q/K downsampling in estimation
        """
        self.block_size = block_size
        self.samples_per_chunk = samples_per_chunk
        self.threshold = threshold
        self.use_triton = use_triton
        self.stride = stride

        # Check Triton availability
        if self.use_triton:
            try:
                import triton
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                if props.major < 8:
                    self.use_triton = False
                    print(f"[XAttentionBSA] Triton requires SM 80+, got SM {props.major}{props.minor}. Falling back to PyTorch.")
            except ImportError:
                self.use_triton = False
                print("[XAttentionBSA] Triton not available. Using PyTorch implementation.")

    def select_blocks(self, available_blocks: List[int], ctx: PolicyContext) -> List[int]:
        """
        Select blocks to load from CPU (for decode compatibility, not used in prefill).

        For prefill, BSA handles chunk-level selection internally.
        """
        # For prefill, we return all blocks - selection happens in sparse_prefill_attention
        return available_blocks

    def sparse_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        """
        Compute XAttention block sparse attention for current chunk.

        This implements a simplified version that loads all historical chunks
        (sparse selection to be implemented in next phase).

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim] (unused, we use prefill buffer)
            v: Value tensor [seq_len, num_kv_heads, head_dim] (unused, we use prefill buffer)
            layer_id: Current transformer layer index
            softmax_scale: Softmax scaling factor from attention layer

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

        context = get_context()
        kvcache_manager = context.kvcache_manager
        offload_engine = kvcache_manager.offload_engine if kvcache_manager else None

        if offload_engine is None:
            # No offload engine, use standard attention with provided k, v
            return self._full_attention(q, k, v, causal=True)

        current_chunk_idx = getattr(context, 'current_chunk_idx', 0)
        seq = getattr(context, 'chunked_seq', None)
        num_tokens = q.shape[0]

        if seq is None:
            # No chunked sequence, fallback to full attention on current chunk only
            return self._full_attention(q, k, v, causal=True)

        # Get prefilled CPU blocks (historical chunks)
        cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None

        # Get compute stream for all attention operations
        compute_stream = offload_engine.compute_stream

        # Step 1: Load historical chunks from CPU using slot mechanism
        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            # Load ALL historical blocks (not just min(num_blocks, num_slots))
            # Use synchronous mode like standard flow when pipeline_depth=1
            if len(load_slots) == 1:
                # Only 1 slot available, cannot pipeline - use synchronous mode
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
                    offload_engine.wait_slot_layer(slot)

                    with torch.cuda.stream(compute_stream):
                        # Get KV from slot - returns [1, block_size, kv_heads, head_dim]
                        prev_k, prev_v = offload_engine.get_kv_for_slot(slot)

                        # Compute attention to historical chunk (non-causal, already processed)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_batched, prev_k, prev_v,
                            softmax_scale=softmax_scale,
                            causal=False,
                        )

                        # Merge results
                        if o_acc is None:
                            o_acc, lse_acc = prev_o, prev_lse
                        else:
                            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

                        # Record compute done so slot can be reused
                        offload_engine.record_slot_compute_done(slot)
            else:
                # Multiple slots available - use pipeline
                num_slots = len(load_slots)

                # Phase 1: Pre-load up to num_slots blocks to fill the pipeline
                num_preload = min(num_slots, num_blocks)
                for i in range(num_preload):
                    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

                # Phase 2: Main loop - compute and immediately reuse slot for next transfer
                for block_idx in range(num_blocks):
                    # Cycle through slots: slot[block_idx % num_slots]
                    current_slot = load_slots[block_idx % num_slots]
                    cpu_block_id = cpu_block_table[block_idx]

                    # Wait for current slot's transfer to complete
                    offload_engine.wait_slot_layer(current_slot)

                    # Compute attention on current slot's data
                    with torch.cuda.stream(compute_stream):
                        # Get KV from slot - returns [1, block_size, kv_heads, head_dim]
                        prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)

                        # Compute attention to historical chunk (non-causal, already processed)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_batched, prev_k, prev_v,
                            softmax_scale=softmax_scale,
                            causal=False,
                        )

                        # Merge results
                        if o_acc is None:
                            o_acc, lse_acc = prev_o, prev_lse
                        else:
                            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

                    # Record compute done so slot can be reused
                    offload_engine.record_slot_compute_done(current_slot)

                    # Issue next transfer if there are more blocks
                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id)

        # Step 2: Compute attention to current chunk (causal mask) - use prefill buffer on compute_stream
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)

            current_o, current_lse = flash_attn_with_lse(
                q_batched,
                k_curr,
                v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        # Step 3: Merge historical and current attention
        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                # No historical chunks processed
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)

        # Sync default stream with compute_stream before returning
        torch.cuda.default_stream().wait_stream(compute_stream)

        # Remove batch dimension: [1, seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim]
        return final_o.squeeze(0)

    def _estimate_historical_chunks(
        self,
        q: torch.Tensor,
        historical_blocks: List[int],
        layer_id: int,
        current_chunk_idx: int,
    ) -> Tuple[List[float], bool]:
        """
        Estimate importance of each historical chunk for current Q.

        First load: Load samples from each historical chunk for estimation.

        Args:
            q: Current chunk queries [chunk_size, num_heads, head_dim]
            historical_blocks: List of historical CPU block IDs
            layer_id: Current layer index
            current_chunk_idx: Current chunk index

        Returns:
            (List of importance scores (one per historical chunk), has_valid_data flag)
            has_valid_data is True if at least one block had non-zero data
        """
        chunk_estimates = []
        has_valid_data = False

        for block_idx, cpu_block_id in enumerate(historical_blocks):
            # First load: Load sample from this historical chunk
            k_sample, v_sample = self._load_block_sample(
                cpu_block_id, layer_id, self.samples_per_chunk
            )

            # Check if loaded data is valid (non-zero)
            if k_sample.abs().max().item() > 0:
                has_valid_data = True

            # Quick estimation: Compute Q attention to this chunk's sample
            # q [chunk_size, H, D] @ k_sample [samples, H, D]
            # Result: Aggregate to chunk-level score
            estimate = self._compute_chunk_estimate(q, k_sample)
            chunk_estimates.append(estimate)

        return chunk_estimates, has_valid_data

    def _select_important_chunks(
        self,
        chunk_estimates: List[float],
    ) -> List[int]:
        """
        Select important chunks based on cumulative attention threshold.

        Args:
            chunk_estimates: Importance scores for each historical chunk

        Returns:
            Indices of selected chunks
        """
        if not chunk_estimates:
            return []

        scores = torch.tensor(chunk_estimates, device='cpu')
        threshold_value = scores.max() * self.threshold

        # Select chunks that contribute to cumulative attention threshold
        selected_indices = []
        cumulative = 0.0
        sorted_indices = torch.argsort(scores, descending=True)

        for idx in sorted_indices:
            cumulative += scores[idx].item()
            selected_indices.append(idx.item())
            if cumulative >= threshold_value:
                break

        return selected_indices

    def _compute_with_selected_chunks(
        self,
        q: torch.Tensor,
        historical_blocks: List[int],
        selected_indices: List[int],
        layer_id: int,
        current_chunk_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute attention to selected historical chunks.

        Second load: Load full data for selected chunks.

        Args:
            q: Current chunk queries
            historical_blocks: All historical block IDs
            selected_indices: Indices of selected blocks
            layer_id: Current layer index
            current_chunk_idx: Current chunk index

        Returns:
            (accumulated_output, accumulated_lse) or (None, None)
        """
        if not selected_indices:
            return None, None

        o_acc = None
        lse_acc = None

        for chunk_idx in selected_indices:
            cpu_block_id = historical_blocks[chunk_idx]

            # Second load: Load full data for this selected chunk
            k_full, v_full = self._load_block_full(
                cpu_block_id, layer_id
            )

            # Compute attention (non-causal, already processed)
            o, lse = self._full_attention(
                q.unsqueeze(0), k_full.unsqueeze(0),
                v_full.unsqueeze(0), causal=False, return_lse=True
            )

            # Merge results
            if o_acc is None:
                o_acc, lse_acc = o.squeeze(0), lse
            else:
                from nanovllm.kvcache.chunked_attention import merge_attention_outputs
                o_acc, lse_acc = merge_attention_outputs(
                    o_acc.unsqueeze(0), lse_acc,
                    o.unsqueeze(0), lse
                )
                o_acc = o_acc.squeeze(0)

        return o_acc, lse_acc

    def _load_block_sample(
        self,
        cpu_block_id: int,
        layer_id: int,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load sample tokens from a CPU block."""
        offload_engine = get_context().kvcache_manager.offload_engine

        k_sample, v_sample = offload_engine.load_block_sample_from_cpu(
            cpu_block_id, layer_id, num_samples
        )
        return k_sample, v_sample

    def _load_block_full(
        self,
        cpu_block_id: int,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load full tokens from a CPU block."""
        offload_engine = get_context().kvcache_manager.offload_engine
        return offload_engine.load_block_full_from_cpu(
            cpu_block_id, layer_id
        )

    def _compute_chunk_estimate(
        self,
        q: torch.Tensor,
        k_sample: torch.Tensor,
    ) -> float:
        """
        Compute chunk-level importance estimate.

        Args:
            q: [chunk_size, num_heads, head_dim]
            k_sample: [num_samples, num_kv_heads, head_dim]

        Returns:
            Aggregate importance score for this chunk
        """
        # Expand K to match Q's head count (GQA support)
        num_heads = q.shape[1]
        num_kv_heads = k_sample.shape[1]
        head_dim = q.shape[2]  # Last dimension is head_dim
        if num_heads != num_kv_heads:
            repeat_factor = num_heads // num_kv_heads
            k_sample = k_sample.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores: Q @ K.T with proper scaling
        # q [chunk_size, H, D], k [samples, H, D] -> need to compute per-head attention
        # Use scaled dot-product attention: (Q @ K.T) / sqrt(D)
        scale = 1.0 / (head_dim ** 0.5)

        # Reshape to 2D: [chunk_size * H, D] @ [D, samples * H] then aggregate
        chunk_size = q.shape[0]
        num_samples = k_sample.shape[0]

        # Reshape for batched matmul: merge heads and seq dims
        q_2d = q.reshape(chunk_size * num_heads, head_dim)  # [chunk_size*H, D]
        k_2d = k_sample.reshape(num_samples * num_heads, head_dim)  # [samples*H, D]

        # Compute scaled Q @ K.T: [chunk_size*H, D] @ [D, samples*H] = [chunk_size*H, samples*H]
        attn_scores_2d = torch.matmul(q_2d, k_2d.T) * scale

        # Use max absolute value as importance (captures both positive and negative attention)
        importance = attn_scores_2d.abs().max().item()

        return importance

    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        return_lse: bool = False,
    ) -> torch.Tensor:
        """
        Compute full FlashAttention (fallback when sparse not applicable).

        Args:
            q: [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
            k, v: Same shape as q
            causal: Apply causal mask
            return_lse: Whether to return log-sum-exp

        Returns:
            attention output [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        """
        from nanovllm.kvcache.chunked_attention import flash_attn_with_lse

        # Handle 3D input: add batch dimension
        input_3d = q.dim() == 3
        if input_3d:
            q = q.unsqueeze(0)  # [seq_len, H, D] -> [1, seq_len, H, D]
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        if return_lse:
            o, lse = flash_attn_with_lse(q, k, v, softmax_scale=self.scale, causal=causal)
            result = (o, lse)
        else:
            o, _ = flash_attn_with_lse(q, k, v, softmax_scale=self.scale, causal=causal)
            result = o

        # Remove batch dimension if input was 3D
        if input_3d:
            if return_lse:
                result = (result[0].squeeze(0), result[1])
            else:
                result = result.squeeze(0)

        return result

    @property
    def scale(self) -> float:
        """Get softmax scale factor from Attention layer."""
        context = get_context()
        # Get scale from current Attention layer in the model
        if hasattr(context, 'current_attention') and context.current_attention is not None:
            return context.current_attention.scale
        # Fallback: try to get from model runner
        if hasattr(context, 'model_runner') and context.model_runner is not None:
            model_runner = context.model_runner
            if hasattr(model_runner, 'model') and hasattr(model_runner.model, 'layers'):
                # Get scale from first attention layer
                first_layer = model_runner.model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    return first_layer.self_attn.scaling
        # Default: 1 / sqrt(128) for Qwen models
        return 1.0 / 128.0 ** 0.5

    def reset(self) -> None:
        """Reset policy state."""
        pass
