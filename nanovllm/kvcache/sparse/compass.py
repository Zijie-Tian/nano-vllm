"""
COMPASS sparse attention policy for chunked prefill.

This is a placeholder implementation that:
1. select_blocks returns empty list (no historical blocks selected)
2. compute_chunked_prefill uses full attention computation on current chunk only

This serves as a baseline for development and testing of the chunked prefill flow
without loading any historical KV blocks.
"""

import logging
import torch
from typing import List, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext
from ..quant.kcache_quant import quantize_kcache_per_token, pack_kvcache_tmac

try:
    from nanovllm.ops.tvm_qgemm.qgemm import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
    HAS_TMAC = True
except ImportError:
    HAS_TMAC = False

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class COMPASSPolicy(SparsePolicy):
    """
    COMPASS sparse attention policy.

    Current implementation:
    - select_blocks: Returns empty list (no historical blocks loaded)
    - compute_chunked_prefill: Only computes attention on current chunk (causal)

    This is essentially a "current chunk only" baseline for chunked prefill.
    """

    # COMPASS supports both prefill and decode for now
    supports_prefill = True
    supports_decode = True

    def __init__(self, lambda_threshold: float = 0.1, bits: int = 2, group_size: int = 128, act_group_size: int = 64):
        """Initialize with statistics tracking and parameters for TMAC prediction."""
        self._stats_num_chunks = 0
        self.lambda_threshold = lambda_threshold
        self.bits = bits
        self.group_size = group_size
        self.act_group_size = act_group_size
        
        # TMAC Codegen
        self.tmac_codegen = None
        self.func_qgemm = None
        
        # Metadata buffers for TMAC verification
        self._q_buffer: torch.Tensor | None = None
        self._k_packed_buffer: torch.Tensor | None = None
        self._k_fp16_verify_buffer: torch.Tensor | None = None
        self._max_q_chunks: int = 0
        self._q_chunk_sizes: list[int] = []

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
        Allocate pinned CPU memory buffers for COMPASS metadata (TMAC prediction).
        
        Allocates buffers for:
        1. Packed K-cache (2-bit interleaved format for TMAC)
        2. Q tensors (all chunks for initial verification)
        """
        if not enable_cpu_offload:
            return
            
        logger.info("[COMPASS] Allocating metadata buffers for TMAC verification...")
        
        # K-cache packing parameters (TMAC 2-bit format)
        # We need to store this per layer and per CPU block
        # The packed size depends on the block_size and quantization strategy.
        # TMAC uses INT4 (2-bit per element effectively) with specific interleaving.
        # For simplicity in this phase, we'll allocate a generic byte buffer that can 
        # hold the packed data. 
        # 16-bit to 2-bit is an 8x reduction in size.
        # Shape: [num_layers, max_cpu_blocks, block_size, kv_heads, head_dim // 8]
        # We'll determine the exact size needed during Phase 2 implementation.
        
        # Estimate max chunks based on sequence length and a typical chunk size
        # Assume minimum chunk size is 512 for estimation
        self._max_q_chunks = max_seq_len // 512 + 1
        
        # Allocate Q buffer: [num_layers, max_chunks, max_chunk_len, num_heads, head_dim]
        # For now, we'll just allocate a flat buffer and keep track of sizes, 
        # or a large enough tensor to hold max seq length.
        # Shape: [num_layers, max_seq_len, num_heads, head_dim]
        # This is pinned memory on CPU.
        # For this verification phase, we'll allocate up to 64 layers to be safe for larger models like GLM-4 (40 layers).
        num_layers = 64
        
        self._q_buffer = torch.zeros(
            (num_layers, max_seq_len, num_heads, head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=True
        )
        
        # Packed K buffer
        # Shape: [num_layers, max_seq_len, num_kv_heads, head_dim // 4] (assuming 8x reduction from fp16)
        self._k_packed_buffer = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim // 4),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True
        )
        
        # Unquantized K buffer for verification
        self._k_fp16_verify_buffer = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=True
        )
        
        q_mb = self._q_buffer.numel() * self._q_buffer.element_size() / (1024 * 1024)
        k_mb = self._k_packed_buffer.numel() * self._k_packed_buffer.element_size() / (1024 * 1024)
        logger.info(f"[COMPASS] Allocated Q buffer: {q_mb:.1f} MB, Packed K buffer: {k_mb:.1f} MB (Pinned CPU)")
        
        self._q_chunk_sizes = []
        
        # Instantiate TMAC Codegen
        if HAS_TMAC:
            logger.info("[COMPASS] Initializing TMAC Codegen...")
            try:
                self.tmac_codegen = QGeMMLUTBitsCodegen(
                    dtype="int8", # input dtype for Q (before packing)
                    target="llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2",
                    name="qgemm_lut",
                    tune=False,
                    verify=False,
                    num_threads=1,
                    bits=self.bits,
                    g=4, # TMAC 2-bit default group
                    group_size=self.group_size,
                    act_group_size=self.act_group_size,
                    out_dtype="float32",
                    m_groups=-1,
                )
                logger.info("[COMPASS] TMAC Codegen initialized successfully.")
            except Exception as e:
                logger.warning(f"[COMPASS] Failed to initialize TMAC Codegen: {e}")
        else:
            logger.warning("[COMPASS] TMAC not available. CPU prediction will not run.")

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select blocks - currently returns all blocks (FullAttention behavior).
        Also offloads the current Q chunk to CPU pinned memory for later verification.
        """
        if q is not None and self._q_buffer is not None:
            # q shape is [seq_len, num_heads, head_dim]
            # Convert to [1, num_heads, seq_len, head_dim] for easier concatenation later,
            # but wait, _q_buffer is [num_layers, max_seq_len, num_heads, head_dim]
            # So we can just copy it directly using start and end indices
            
            # Since q is only the current chunk, we need to know its start index.
            # We can compute it from total_kv_len which represents the sequence length before this chunk
            start_idx = ctx.total_kv_len
            seq_len = q.shape[0]
            
            # Copy to pinned CPU buffer on a separate stream if possible, or default stream
            # Wait for offload stream maybe? For now, simple blocking copy is fine for verification
            self._q_buffer[ctx.layer_id, start_idx:start_idx + seq_len].copy_(q, non_blocking=True)
            
            if ctx.layer_id == 0:
                self._q_chunk_sizes.append(seq_len)

        # For now, return all blocks to test basic chunked prefill flow
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1
            logger.debug(
                f"[COMPASS] chunk={ctx.query_chunk_idx}, "
                f"available={len(available_blocks)}, selected={len(available_blocks)}"
            )
        return available_blocks

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats_num_chunks = 0

    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            "num_chunks": self._stats_num_chunks,
        }

    # ========================================================================
    # GPU-only methods (non-chunked) - Not supported for now
    # ========================================================================

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
        block_tables=None,
    ) -> torch.Tensor:
        """GPU-only prefill - not supported, use FullAttentionPolicy instead."""
        raise NotImplementedError(
            "COMPASS policy only supports chunked prefill mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables=None,
    ) -> torch.Tensor:
        """GPU-only decode - not supported, use FullAttentionPolicy instead."""
        raise NotImplementedError(
            "COMPASS policy only supports chunked decode mode. "
            "Use FullAttentionPolicy for GPU-only mode."
        )

    # ========================================================================
    # Chunked offload methods
    # ========================================================================

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
        Compute attention for chunked prefill.

        Currently implements full attention (loads all selected_blocks).
        TODO: Implement COMPASS sparse attention computation.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim] (unused, from prefill buffer)
            v: Value tensor [seq_len, num_kv_heads, head_dim] (unused, from prefill buffer)
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            current_chunk_idx: Current chunk index
            seq: Sequence object
            num_tokens: Number of tokens in current chunk
            selected_blocks: List of CPU block IDs to load

        Returns:
            Attention output [seq_len, num_heads, head_dim]
        """
        # Use FlashInfer-based implementations (same as FullAttentionPolicy)
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        logger.debug(
            f"[COMPASS] compute_chunked_prefill called, "
            f"layer={layer_id}, chunk={current_chunk_idx}, "
            f"num_tokens={num_tokens}, selected_blocks={len(selected_blocks)}"
        )

        q_batched = q.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
        o_acc = None
        lse_acc = None
        compute_stream = offload_engine.compute_stream

        # Load and compute attention on selected historical blocks
        cpu_block_table = selected_blocks

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            if len(load_slots) == 1:
                # Only 1 slot - use synchronous mode
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(
                        slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id
                    )
                    offload_engine.wait_slot_layer(slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_batched,
                            prev_k,
                            prev_v,
                            softmax_scale=softmax_scale,
                            causal=False,
                        )
                        if o_acc is None:
                            o_acc, lse_acc = prev_o, prev_lse
                        else:
                            o_acc, lse_acc = merge_attention_outputs(
                                o_acc, lse_acc, prev_o, prev_lse
                            )
                        offload_engine.record_slot_compute_done(slot)
            else:
                # Multiple slots - use pipeline
                num_slots = len(load_slots)
                num_preload = min(num_slots, num_blocks)
                for i in range(num_preload):
                    cpu_block_id = cpu_block_table[i]
                    offload_engine.load_to_slot_layer(
                        load_slots[i], layer_id, cpu_block_id, chunk_idx=cpu_block_id
                    )

                for block_idx in range(num_blocks):
                    current_slot = load_slots[block_idx % num_slots]

                    offload_engine.wait_slot_layer(current_slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_batched,
                            prev_k,
                            prev_v,
                            softmax_scale=softmax_scale,
                            causal=False,
                        )
                        offload_engine.record_slot_compute_done(current_slot)

                        if o_acc is None:
                            o_acc, lse_acc = prev_o, prev_lse
                        else:
                            o_acc, lse_acc = merge_attention_outputs(
                                o_acc, lse_acc, prev_o, prev_lse
                            )

                    # Issue next transfer
                    next_block_idx = block_idx + num_slots
                    if next_block_idx < num_blocks:
                        next_slot = load_slots[next_block_idx % num_slots]
                        next_cpu_block_id = cpu_block_table[next_block_idx]
                        offload_engine.load_to_slot_layer(
                            next_slot,
                            layer_id,
                            next_cpu_block_id,
                            chunk_idx=next_cpu_block_id,
                        )

        # Compute attention to current chunk (causal mask)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(
                layer_id, num_tokens
            )
            current_o, current_lse = flash_attn_with_lse(
                q_batched,
                k_curr,
                v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        # Merge historical and current attention
        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(
                    o_acc, lse_acc, current_o, current_lse
                )

        # Sync default stream with compute_stream before returning
        torch.cuda.default_stream().wait_stream(compute_stream)

        # Remove batch dimension: [1, seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim]
        return final_o.squeeze(0)

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
        Compute attention for chunked decode.

        COMPASS uses full attention on all prefilled blocks during decode
        (since select_blocks returns all available blocks in decode phase).

        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            layer_id: Current layer index
            softmax_scale: Softmax scaling factor
            offload_engine: OffloadEngine for loading blocks
            kvcache_manager: KVCacheManager for block management
            seq: Sequence object
            selected_blocks: List of CPU block IDs to process (already filtered)

        Returns:
            Attention output [batch_size, 1, num_heads, head_dim]
        """
        # COMPASS uses FullAttentionPolicy for decode computation
        # (select_blocks already returns all blocks in decode phase)
        from .full_policy import FullAttentionPolicy

        logger.debug(
            f"[COMPASS] compute_chunked_decode using FullAttentionPolicy, "
            f"layer={layer_id}, selected_blocks={len(selected_blocks)}"
        )

        fallback_policy = FullAttentionPolicy()
        return fallback_policy.compute_chunked_decode(
            q,
            layer_id,
            softmax_scale,
            offload_engine,
            kvcache_manager,
            seq,
            selected_blocks,
        )

    def offload_prefill_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
        **kwargs,
    ) -> None:
        # Get the prefill K buffer data that is about to be offloaded
        # Shape: [num_tokens, kv_heads, head_dim]
        k_curr = offload_engine.prefill_k_buffer[layer_id, :num_tokens]

        if self._k_packed_buffer is not None:
            # We must quantize and pack
            # quantize_kcache_per_token expects [..., head_dim]
            k_q, _, _ = quantize_kcache_per_token(k_curr, bits=2, sym=False)
            
            # pack_kvcache_tmac expects [batch, n_head, M, K]
            # Here batch=1, n_head=kv_heads, M=num_tokens, K=head_dim
            # So we need to reshape k_q from [num_tokens, kv_heads, head_dim] to [1, kv_heads, num_tokens, head_dim]
            k_q_tmac = k_q.transpose(0, 1).unsqueeze(0)
            
            # Note: pack_kvcache_tmac expects M (num_tokens) to be a multiple of 128 for bits=2 and bm=256
            # If num_tokens is not aligned, we need to pad it
            pad_m = (128 - (num_tokens % 128)) % 128
            if pad_m > 0:
                k_q_tmac = torch.nn.functional.pad(k_q_tmac, (0, 0, 0, pad_m))
            
            packed_k = pack_kvcache_tmac(k_q_tmac, is_key=True, bits=2)
            
            # packed_k shape is [1, kv_heads, (num_tokens+pad_m)*2//256, head_dim//4, 128]
            # We want to store it sequentially based on the cpu_block_id or token index
            # For verification, we can just flatten the packed_k into bytes per token 
            # and store it in our linear _k_packed_buffer. 
            # The size per token of packed data is (head_dim * 2 bits) / 8 bits/byte = head_dim / 4 bytes.
            # So packed_k has total size = (num_tokens_padded) * kv_heads * head_dim / 4 bytes.
            # Let's reshape to [kv_heads, num_tokens_padded, head_dim // 4]
            # Then transpose back to [num_tokens_padded, kv_heads, head_dim // 4] to match our buffer layout
            kv_heads = k_curr.shape[1]
            head_dim = k_curr.shape[2]
            
            packed_k_flat = packed_k.view(kv_heads, -1, head_dim // 4).transpose(0, 1)
            
            # We need to know where to write it in _k_packed_buffer
            # Since cpu_block_id gives us the block index, and block_size is fixed
            block_size = offload_engine.block_size
            start_idx = cpu_block_id * block_size
            
            # We write the valid tokens part
            write_len = min(num_tokens, block_size)
            
            # Copy to pinned CPU buffer
            # Since packed_k is padded, we only copy the write_len part, 
            # though TMAC packing scrambles tokens in groups of 128.
            # We copy the aligned part.
            stream = offload_engine.prefill_offload_streams[layer_id]
            with torch.cuda.stream(stream):
                # We need to copy to CPU buffer asynchronously
                self._k_packed_buffer[layer_id, start_idx:start_idx + write_len].copy_(
                    packed_k_flat[:write_len], non_blocking=True
                )

                # Also copy original FP16 K cache for verification
                if self._k_fp16_verify_buffer is not None:
                    self._k_fp16_verify_buffer[layer_id, start_idx:start_idx + write_len].copy_(
                        k_curr[:write_len], non_blocking=True
                    )

        super().offload_prefill_chunk(
            offload_engine, layer_id, cpu_block_id, num_tokens, **kwargs
        )

    def offload_decode_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        **kwargs,
    ) -> None:
        super().offload_decode_chunk(offload_engine, layer_id, cpu_block_id, **kwargs)

    def __repr__(self) -> str:
        return "COMPASSPolicy()"
