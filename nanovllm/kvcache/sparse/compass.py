"""
COMPASS sparse attention policy for chunked prefill.

Two-tier sparse attention pipeline:
- Tier 1 (CPU, coarse): TMAC 2-bit qGEMM filters which 4096-token blocks to load
- Tier 2 (GPU, fine): BLASST dynamically prunes sub-blocks within loaded blocks

select_blocks uses TMAC CPU prediction to filter available blocks.
compute_chunked_prefill uses BLASST-like attention with dynamic pruning on
the CPU-pre-filtered block subset.
"""

import logging
import math
import numpy as np
import torch
from typing import List, Optional, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext
from ..quant.kcache_quant import quantize_kcache_per_token, pack_kvcache_tmac, pack_scales_tmac

try:
    from nanovllm.ops.tvm_qgemm.qgemm import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
    import tvm
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
    COMPASS sparse attention policy with two-tier sparsity.

    Tier 1 (CPU): TMAC 2-bit qGEMM in select_blocks filters coarse 4096-token blocks.
    Tier 2 (GPU): BLASST dynamic pruning in compute_chunked_prefill on loaded blocks.
    """

    supports_prefill = True
    supports_decode = True

    # TMAC estimation granularity (tokens per fine-grained sub-block)
    FINE_GRAIN = 128

    def __init__(
        self,
        lambda_threshold: float = 0.001,
        bits: int = 2,
        group_size: int = 128,
        act_group_size: int = 64,
    ):
        """Initialize with parameters for TMAC prediction and BLASST GPU attention."""
        self._stats_num_chunks = 0
        self._stats_selected_blocks = 0
        self._stats_total_blocks = 0
        self.lambda_threshold = lambda_threshold
        self.bits = bits
        self.group_size = group_size
        self.act_group_size = act_group_size

        # TMAC compiled operators (set during alloc_policy_metadata)
        self._func_qgemm = None
        self._func_preprocessor = None
        self._tvm_device = None
        self._tmac_codegen = None
        self._tmac_preproc_codegen = None

        # Model dimensions (set during alloc_policy_metadata)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        # Metadata buffers for TMAC CPU prediction
        self._q_buffer: Optional[torch.Tensor] = None
        self._k_packed_buffer: Optional[torch.Tensor] = None
        self._k_scales_buffer: Optional[torch.Tensor] = None
        self._k_fp16_verify_buffer: Optional[torch.Tensor] = None
        self._q_chunk_sizes: list[int] = []

    # ========================================================================
    # Initialization
    # ========================================================================

    def alloc_policy_metadata(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        enable_cpu_offload: bool = False,
        num_layers: int = 64,
    ) -> None:
        """
        Allocate CPU buffers and compile TMAC operators for CPU prediction.
        """
        if not enable_cpu_offload:
            return

        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        logger.info("[COMPASS] Allocating metadata buffers for TMAC prediction...")

        # num_layers is now passed from model config (e.g., 32 for Llama-3.1-8B)

        # Q buffer: [num_layers, max_seq_len, num_heads, head_dim]
        self._q_buffer = torch.zeros(
            (num_layers, max_seq_len, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True,
        )

        # Packed K buffer: [num_layers, max_seq_len, num_kv_heads, head_dim // 4]
        self._k_packed_buffer = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim // 4),
            dtype=torch.uint8, device="cpu", pin_memory=True,
        )

        # K scales buffer for quantization: [num_layers, max_seq_len, num_kv_heads, 1]
        self._k_scales_buffer = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, 1),
            dtype=dtype, device="cpu", pin_memory=True,
        )

        # FP16 verification buffer
        self._k_fp16_verify_buffer = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True,
        )

        q_mb = self._q_buffer.numel() * self._q_buffer.element_size() / (1024 * 1024)
        k_mb = self._k_packed_buffer.numel() * self._k_packed_buffer.element_size() / (1024 * 1024)
        logger.info(f"[COMPASS] Allocated Q buffer: {q_mb:.1f} MB, Packed K buffer: {k_mb:.1f} MB (Pinned CPU)")

        self._q_chunk_sizes = []

        # Compile TMAC operators
        if HAS_TMAC:
            self._compile_tmac_operators(num_heads, num_kv_heads, head_dim)
        else:
            logger.warning("[COMPASS] TMAC not available. CPU prediction will not run.")

    def _compile_tmac_operators(self, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        """Compile TMAC preprocessor and qGEMM operators for CPU sparse estimation."""
        target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
        K = head_dim
        # N=1: we process one KV head at a time (each has independent K data)
        N = 1
        # M = fine_grain tokens * bits (TMAC bit-expansion)
        M = self.FINE_GRAIN * self.bits  # 128 * 2 = 256

        logger.info(f"[COMPASS] Compiling TMAC operators: M={M}, N={N}, K={K}...")

        try:
            # 1. Compile preprocessor: converts one Q head → QLUT
            self._tmac_preproc_codegen = QGeMMLUTBitsPreprocessorCodegen(
                dtype="int8",
                target=target,
                name="build/compass_preproc",
                tune=False,
                verify=False,
                num_threads=1,
                g=4,
                act_group_size=self.act_group_size,
                out_dtype="float32",
                bits=self.bits,
            )
            func_preproc, arrays_preproc = self._tmac_preproc_codegen.compile(N, K)
            logger.info("[COMPASS] TMAC Preprocessor compiled successfully.")

            # 2. Compile qGEMM: packed_K_head × QLUT_head → attention scores
            self._tmac_codegen = QGeMMLUTBitsCodegen(
                dtype="int8",
                target=target,
                name="build/compass_qgemm",
                tune=False,
                verify=False,
                num_threads=1,
                bits=self.bits,
                g=4,
                group_size=self.group_size,
                act_group_size=self.act_group_size,
                out_dtype="float32",
                m_groups=-1,
                zero_point=True,
            )
            func_qgemm, arrays_qgemm = self._tmac_codegen.compile(M, N, K)
            logger.info("[COMPASS] TMAC QGEMM compiled successfully.")

            # Store compiled functions
            self._func_preprocessor = func_preproc
            self._func_qgemm = func_qgemm
            self._tvm_device = tvm.cpu(0)

        except Exception as e:
            logger.warning(f"[COMPASS] Failed to compile TMAC operators: {e}")
            import traceback
            traceback.print_exc()
            self._func_preprocessor = None
            self._func_qgemm = None

    # ========================================================================
    # TMAC CPU Estimation Helpers
    # ========================================================================

    def _preprocess_q_per_head(self, q_chunk: torch.Tensor, kv_head_idx: int) -> tuple:
        """
        Run compiled TMAC preprocessor for one KV head's Q representative.

        Args:
            q_chunk: [seq_len, num_heads, head_dim] on CPU
            kv_head_idx: which KV head to process

        Returns:
            (lut_scales, lut_biases, qlut) as tvm.nd.NDArray, each with N=1
        """
        head_dim = q_chunk.shape[2]
        num_heads = q_chunk.shape[1]
        heads_per_group = num_heads // self._num_kv_heads
        K = head_dim
        N = 1
        g = 4

        # Average Q across seq_len, then across Q heads mapping to this KV head
        q_avg = q_chunk.float().mean(dim=0)  # [num_heads, head_dim]
        start_h = kv_head_idx * heads_per_group
        end_h = start_h + heads_per_group
        q_head = q_avg[start_h:end_h].mean(dim=0, keepdim=True)  # [1, head_dim]

        q_np = q_head.numpy().astype("float32")

        # Allocate TVM arrays matching preprocessor tensors: [B, LUT_Scales, LUT_Biases, QLUT]
        dev = self._tvm_device
        B_tvm = tvm.nd.array(q_np, dev)
        LUT_Scales_tvm = tvm.nd.array(
            np.zeros((N, K // self.act_group_size), dtype="float32"), dev
        )
        LUT_Biases_tvm = tvm.nd.array(
            np.zeros((N, K // self.act_group_size), dtype="float32"), dev
        )
        QLUT_tvm = tvm.nd.array(
            np.zeros((N, K // g, 1 << g), dtype="int8"), dev
        )

        # Call compiled preprocessor
        self._func_preprocessor(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)

        return LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm

    def _estimate_subblock_score(
        self,
        k_fp16_sub: torch.Tensor,
        kv_head_idx: int,
        qlut_tvm: "tvm.nd.NDArray",
        lut_scales_tvm: "tvm.nd.NDArray",
        lut_biases_tvm: "tvm.nd.NDArray",
    ) -> float:
        """
        Estimate attention score for a 128-token K sub-block of one KV head.

        Quantizes, packs, and runs compiled TMAC qGEMM to get score.

        Args:
            k_fp16_sub: [fine_grain, num_kv_heads, head_dim] FP16 K data
            kv_head_idx: which KV head to process
            qlut_tvm, lut_scales_tvm, lut_biases_tvm: from preprocessor for this head

        Returns:
            Max absolute score for this head and sub-block
        """
        # Extract single head: [fine_grain, 1, head_dim]
        # Convert bf16→fp16 if needed: TVM/numpy don't support bf16
        if k_fp16_sub.dtype == torch.bfloat16:
            k_fp16_sub = k_fp16_sub.to(torch.float16)
        k_head = k_fp16_sub[:, kv_head_idx:kv_head_idx+1, :]

        # Quantize per-token: k_q [fine_grain, 1, head_dim], k_s [fine_grain, 1, 1]
        k_q, k_s, k_z = quantize_kcache_per_token(k_head, bits=self.bits, sym=False)

        # Pack for TMAC: [1, 1, fine_grain, head_dim] → packed
        k_q_tmac = k_q.transpose(0, 1).unsqueeze(0)  # [1, 1, fine_grain, head_dim]
        packed_k = pack_kvcache_tmac(k_q_tmac, is_key=True, bits=self.bits)
        A_np = packed_k[0, 0].numpy()  # [M//bm, K//g, bm//ngroups_per_elem]

        # Pack scales
        k_s_tmac = k_s.transpose(0, 1).unsqueeze(0)  # [1, 1, fine_grain, 1]
        k_z_tmac = k_z.transpose(0, 1).unsqueeze(0) if k_z is not None else None
        packed_scales = pack_scales_tmac(k_s_tmac, zeros=k_z_tmac, bits=self.bits)
        Scales_np = packed_scales[0, 0].numpy()  # [M//bm, K//group_size, ...]

        M = self.FINE_GRAIN * self.bits
        N = 1
        dev = self._tvm_device

        try:
            # Allocate TVM arrays matching qGEMM tensors: [A, LUT, Scales, LUT_Scales, LUT_Biases, C]
            A_tvm = tvm.nd.array(A_np.astype("uint8"), dev)
            Scales_tvm = tvm.nd.array(Scales_np.astype("float32"), dev)
            C_tvm = tvm.nd.array(
                np.zeros((N, M // self.bits), dtype="float32"), dev
            )

            # Call compiled qGEMM
            self._func_qgemm(A_tvm, qlut_tvm, Scales_tvm, lut_scales_tvm, lut_biases_tvm, C_tvm)

            # Read output scores
            scores = C_tvm.numpy()  # [N=1, M//bits=FINE_GRAIN]
            return float(np.max(np.abs(scores)))
        except Exception as e:
            logger.debug(f"[COMPASS] TMAC estimation error: {e}")
            return float('inf')  # On error, select block to be safe

    # ========================================================================
    # Block selection
    # ========================================================================

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        """
        Select blocks using TMAC CPU prediction during prefill.

        For each available 4096-token block, scans 128-token sub-blocks using
        TMAC qGEMM. If any sub-block's score exceeds the threshold, the entire
        block is selected for GPU attention.

        During decode, returns all blocks (falls back to full attention).
        """
        # Save Q to buffer for metadata tracking
        if q is not None and self._q_buffer is not None:
            start_idx = ctx.total_kv_len
            seq_len = q.shape[0]
            self._q_buffer[ctx.layer_id, start_idx:start_idx + seq_len].copy_(q, non_blocking=True)
            if ctx.layer_id == 0:
                self._q_chunk_sizes.append(seq_len)

        # Track statistics
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1
            self._stats_total_blocks += len(available_blocks)

        # During decode or when TMAC not available: return all blocks
        if not ctx.is_prefill or not available_blocks or self._tmac_codegen is None:
            if ctx.layer_id == 0:
                self._stats_selected_blocks += len(available_blocks)
            return available_blocks

        # --- CPU TMAC Sparse Estimation ---
        # Synchronize to ensure previous offload_prefill_chunk has written
        # FP16 K data to the verify buffer (it's async on offload streams)
        torch.cuda.synchronize()

        block_size = ctx.block_size
        fine_grain = self.FINE_GRAIN
        num_fine_per_block = block_size // fine_grain
        threshold = abs(math.log(self.lambda_threshold))  # |ln(0.001)| ≈ 6.9

        # Step 1: Preprocess Q → QLUT for each KV head
        q_cpu = q.cpu()
        try:
            per_head_qluts = []
            for kv_h in range(self._num_kv_heads):
                lut_s, lut_b, qlut = self._preprocess_q_per_head(q_cpu, kv_h)
                per_head_qluts.append((lut_s, lut_b, qlut))
        except Exception as e:
            logger.warning(f"[COMPASS] Preprocessor failed: {e}, selecting all blocks")
            if ctx.layer_id == 0:
                self._stats_selected_blocks += len(available_blocks)
            return available_blocks

        # Step 2: For each block, scan sub-blocks across all heads
        selected_indices = []
        block_max_scores = []  # Track max score per block for diagnostics
        for block_idx, cpu_block_id in enumerate(available_blocks):
            start_token = cpu_block_id * block_size
            block_selected = False
            block_max_score = -float('inf')

            for sub_idx in range(num_fine_per_block):
                sub_start = start_token + sub_idx * fine_grain
                sub_end = sub_start + fine_grain

                # Get FP16 K sub-block for estimation
                k_fp16_sub = self._k_fp16_verify_buffer[ctx.layer_id, sub_start:sub_end]
                abssum = k_fp16_sub.abs().sum().item()
                if block_idx == 0 and sub_idx == 0 and ctx.layer_id == 0:
                    logger.info(
                        f"[COMPASS-DEBUG] read L{ctx.layer_id}: block={cpu_block_id}, "
                        f"sub_start={sub_start}, sub_end={sub_end}, "
                        f"abssum={abssum:.2f}, shape={k_fp16_sub.shape}, "
                        f"block_size={block_size}"
                    )
                if abssum == 0:
                    continue

                try:
                    # Check each KV head
                    for kv_h in range(self._num_kv_heads):
                        lut_s, lut_b, qlut = per_head_qluts[kv_h]
                        score = self._estimate_subblock_score(
                            k_fp16_sub, kv_h, qlut, lut_s, lut_b,
                        )
                        block_max_score = max(block_max_score, score)
                        if score > threshold:
                            block_selected = True
                            break

                    if block_selected:
                        break
                except Exception as e:
                    logger.warning(f"[COMPASS] Sub-block estimation error (L{ctx.layer_id} kv_h={kv_h}): {e}")
                    block_selected = True  # Select on error to be safe
                    break

            block_max_scores.append(block_max_score)
            if block_selected:
                selected_indices.append(block_idx)

        selected = [available_blocks[i] for i in selected_indices]

        # Log density and score distribution
        self._stats_selected_blocks += len(selected)
        score_min = min(block_max_scores) if block_max_scores else 0
        score_max = max(block_max_scores) if block_max_scores else 0
        score_mean = sum(block_max_scores) / len(block_max_scores) if block_max_scores else 0
        logger.info(
            f"[COMPASS] layer={ctx.layer_id}, chunk={ctx.query_chunk_idx}, "
            f"available={len(available_blocks)}, selected={len(selected)} "
            f"(density={len(selected)/max(len(available_blocks),1)*100:.1f}%), "
            f"threshold={threshold:.2f}, scores=[min={score_min:.2f}, mean={score_mean:.2f}, max={score_max:.2f}]"
        )

        return selected

    # ========================================================================
    # Statistics
    # ========================================================================

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats_num_chunks = 0
        self._stats_selected_blocks = 0
        self._stats_total_blocks = 0

    def get_stats(self) -> dict:
        """Get statistics."""
        select_rate = 0.0
        if self._stats_total_blocks > 0:
            select_rate = self._stats_selected_blocks / self._stats_total_blocks
        return {
            "num_chunks": self._stats_num_chunks,
            "selected_blocks": self._stats_selected_blocks,
            "total_blocks": self._stats_total_blocks,
            "select_rate": select_rate,
        }

    # ========================================================================
    # GPU-only methods (non-chunked) - Not supported
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
    # Chunked offload methods — BLASST-like GPU attention
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
        Compute attention with BLASST-like dynamic pruning on CPU-pre-filtered blocks.

        The selected_blocks have already been filtered by TMAC CPU prediction.
        This method applies BLASST dynamic pruning on top for fine-grained sparsity.
        """
        from nanovllm.ops.chunked_attention import (
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )
        from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill

        total_seq_len = len(seq) if seq else num_tokens
        lambda_val = self.lambda_threshold
        ln_lambda = math.log(lambda_val)

        q_len = q.shape[0]
        num_heads = q.shape[1]
        compute_stream = offload_engine.compute_stream
        q_input = q.unsqueeze(0).transpose(1, 2).contiguous()  # [1, num_heads, q_len, head_dim]

        historical_o = None
        historical_lse = None
        historical_m_global = None  # Running max for BLASST pruning

        TRITON_BLOCK_M, TRITON_BLOCK_N = 128, 64
        grid_0 = (q_len + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
        grid_1 = num_heads
        num_kv_subblocks = kvcache_manager.block_size // TRITON_BLOCK_N

        def get_mask_buffer(is_causal=False, kv_len_override=None):
            num_sub = (
                num_kv_subblocks
                if kv_len_override is None
                else (kv_len_override + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
            )
            mask = torch.ones(
                (grid_0, grid_1, num_sub), device=q.device, dtype=torch.int8
            )
            if is_causal:
                for q_idx in range(grid_0):
                    q_end_pos = (q_idx + 1) * TRITON_BLOCK_M
                    for kv_idx in range(num_sub):
                        kv_start_pos = kv_idx * TRITON_BLOCK_N
                        if kv_start_pos >= q_end_pos:
                            mask[q_idx, :, kv_idx] = 0
            return mask

        logger.debug(
            f"[COMPASS] compute_chunked_prefill: layer={layer_id}, "
            f"chunk={current_chunk_idx}, num_tokens={num_tokens}, "
            f"selected_blocks={len(selected_blocks)}"
        )

        # 1. Process CPU-pre-filtered historical blocks with BLASST pruning
        cpu_block_table = selected_blocks
        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_slots = len(load_slots)
            num_blocks = len(cpu_block_table)

            num_preload = min(num_slots, num_blocks)
            for i in range(num_preload):
                offload_engine.load_to_slot_layer(
                    load_slots[i], layer_id, cpu_block_table[i]
                )

            for block_idx in range(num_blocks):
                current_slot = load_slots[block_idx % num_slots]
                offload_engine.wait_slot_layer(current_slot)

                with torch.cuda.stream(compute_stream):
                    prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                    k_input = prev_k.transpose(1, 2).contiguous()
                    v_input = prev_v.transpose(1, 2).contiguous()

                    mask_buffer = get_mask_buffer(is_causal=False)

                    out, lse, m_global_out = blasst_chunked_prefill(
                        q=q_input,
                        k=k_input,
                        v=v_input,
                        threshold_ln_lambda=ln_lambda,
                        m_global_in=historical_m_global,
                        mask_buffer=mask_buffer,
                    )

                    # Update running max for next kernel call
                    if historical_m_global is None:
                        historical_m_global = m_global_out
                    else:
                        historical_m_global = torch.maximum(historical_m_global, m_global_out)

                    compute_stream.synchronize()

                    block_o = out.transpose(1, 2).contiguous()
                    block_lse = lse

                    if historical_o is None:
                        historical_o, historical_lse = block_o, block_lse
                    else:
                        historical_o, historical_lse = merge_attention_outputs(
                            historical_o, historical_lse, block_o, block_lse
                        )

                offload_engine.record_slot_compute_done(current_slot)

                next_block_idx = block_idx + num_slots
                if next_block_idx < num_blocks:
                    offload_engine.load_to_slot_layer(
                        load_slots[next_block_idx % num_slots],
                        layer_id,
                        cpu_block_table[next_block_idx],
                    )

        # 2. Process current prefill chunk (GPU buffer, causal)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(
                layer_id, num_tokens
            )
            k_curr_input = k_curr.transpose(1, 2).contiguous()
            v_curr_input = v_curr.transpose(1, 2).contiguous()

            kv_offset = len(selected_blocks) * kvcache_manager.block_size

            curr_mask_buffer = get_mask_buffer(
                is_causal=False, kv_len_override=num_tokens
            )

            out_curr, lse_curr, _ = blasst_chunked_prefill(
                q=q_input,
                k=k_curr_input,
                v=v_curr_input,
                threshold_ln_lambda=ln_lambda,
                m_global_in=historical_m_global,
                mask_buffer=curr_mask_buffer,
                is_causal=True,
                kv_offset=kv_offset,
            )

            compute_stream.synchronize()

            block_o = out_curr.transpose(1, 2).contiguous()
            block_lse = lse_curr

            if historical_o is None:
                final_o = block_o
            else:
                final_o, _ = merge_attention_outputs(
                    historical_o, historical_lse, block_o, block_lse
                )

        # 3. Finalize
        torch.cuda.default_stream().wait_stream(compute_stream)
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
        Falls back to FullAttentionPolicy for decode phase.
        """
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

    # ========================================================================
    # Offload hooks
    # ========================================================================

    def offload_prefill_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
        **kwargs,
    ) -> None:
        """Offload prefill chunk: quantize and pack K-cache for TMAC prediction."""
        k_curr = offload_engine.prefill_k_buffer[layer_id, :num_tokens]

        if self._k_packed_buffer is not None:
            # Quantize
            k_q, k_scales, k_zeros = quantize_kcache_per_token(k_curr, bits=2, sym=False)

            # Pack for TMAC: [1, kv_heads, num_tokens, head_dim]
            k_q_tmac = k_q.transpose(0, 1).unsqueeze(0)

            # Pad to multiple of 128
            pad_m = (128 - (num_tokens % 128)) % 128
            if pad_m > 0:
                k_q_tmac = torch.nn.functional.pad(k_q_tmac, (0, 0, 0, pad_m))

            packed_k = pack_kvcache_tmac(k_q_tmac, is_key=True, bits=2)

            kv_heads = k_curr.shape[1]
            head_dim = k_curr.shape[2]
            packed_k_flat = packed_k.view(kv_heads, -1, head_dim // 4).transpose(0, 1)

            block_size = offload_engine.block_size
            start_idx = cpu_block_id * block_size
            write_len = min(num_tokens, block_size)

            stream = offload_engine.prefill_offload_streams[layer_id]
            with torch.cuda.stream(stream):
                self._k_packed_buffer[layer_id, start_idx:start_idx + write_len].copy_(
                    packed_k_flat[:write_len], non_blocking=True
                )

                # Save scales for TMAC estimation
                if self._k_scales_buffer is not None:
                    self._k_scales_buffer[layer_id, start_idx:start_idx + write_len].copy_(
                        k_scales[:write_len], non_blocking=True
                    )

                # Save FP16 K for verification and CPU estimation
                if self._k_fp16_verify_buffer is not None:
                    self._k_fp16_verify_buffer[layer_id, start_idx:start_idx + write_len].copy_(
                        k_curr[:write_len], non_blocking=True
                    )
                    if layer_id == 0:
                        stream.synchronize()
                        buf_check = self._k_fp16_verify_buffer[layer_id, start_idx:start_idx + write_len]
                        logger.info(
                            f"[COMPASS-DEBUG] offload L0: block={cpu_block_id}, "
                            f"start_idx={start_idx}, write_len={write_len}, "
                            f"k_curr_abssum={k_curr[:write_len].abs().sum():.2f}, "
                            f"buf_abssum={buf_check.abs().sum():.2f}, "
                            f"k_curr_device={k_curr.device}, k_curr_shape={k_curr.shape}"
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
        return f"COMPASSPolicy(lambda={self.lambda_threshold})"
