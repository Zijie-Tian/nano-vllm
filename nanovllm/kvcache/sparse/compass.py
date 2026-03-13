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
        self.g = 4  # TMAC group size (fixed)
        self.bm = 256  # TMAC tile size = FINE_GRAIN * bits

        # TMAC compiled operators (set during alloc_policy_metadata)
        self._func_qgemm = None
        self._func_preprocessor = None
        self._tvm_device = None
        self._tmac_codegen = None
        self._tmac_preproc_codegen = None

        # L1 Batched TMAC operators: {num_blocks -> compiled func}
        self._func_qgemm_batched: dict = {}

        # Model dimensions (set during alloc_policy_metadata)
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        # Metadata buffers for TMAC CPU prediction
        self._q_buffer: Optional[torch.Tensor] = None
        self._k_packed_buffer: Optional[torch.Tensor] = None
        self._k_scales_buffer: Optional[torch.Tensor] = None
        self._k_zeros_buffer: Optional[torch.Tensor] = None
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

        # K zeros buffer for quantization: [num_layers, max_seq_len, num_kv_heads, 1]
        self._k_zeros_buffer = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, 1),
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
        """Compile TMAC preprocessor and qGEMM operators for CPU sparse estimation.

        L1 batching: compiles qGEMM for multiple M values with N=1,
        so all sub-blocks for one head are estimated in a single call.
        """
        target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
        K = head_dim
        N = 1
        M_single = self.FINE_GRAIN * self.bits  # 256
        subs_per_block = 4096 // self.FINE_GRAIN  # 32

        logger.info(f"[COMPASS] Compiling TMAC operators (L1 batched)...")

        try:
            # 1. Compile preprocessor: N=1
            self._tmac_preproc_codegen = QGeMMLUTBitsPreprocessorCodegen(
                dtype="int8", target=target,
                name="build/compass_preproc",
                tune=False, verify=False, num_threads=1,
                g=4, act_group_size=self.act_group_size,
                out_dtype="float32", bits=self.bits,
            )
            func_preproc, _ = self._tmac_preproc_codegen.compile(N, K)
            self._func_preprocessor = func_preproc
            logger.info("[COMPASS] TMAC Preprocessor (N=1) compiled.")

            # 2. Compile single qGEMM (legacy): M=256, N=1
            self._tmac_codegen = QGeMMLUTBitsCodegen(
                dtype="int8", target=target,
                name="build/compass_qgemm",
                tune=False, verify=False, num_threads=32,
                bits=self.bits, g=4,
                group_size=self.group_size,
                act_group_size=self.act_group_size,
                out_dtype="float32", m_groups=-1,
                zero_point=True,
            )
            func_qgemm, _ = self._tmac_codegen.compile(M_single, N, K)
            self._func_qgemm = func_qgemm
            logger.info(f"[COMPASS] TMAC QGEMM (M={M_single}, N=1, threads=32) compiled.")

            # 3. Compile batched qGEMM for 1-10 blocks: M=subs*256, N=1
            #    Each head has different K data, so N must stay 1.
            #    We batch across the M dimension (sub-blocks) only.
            self._func_qgemm_batched = {}
            for nb in range(1, 11):
                M_big = nb * subs_per_block * M_single
                codegen_big = QGeMMLUTBitsCodegen(
                    dtype="int8", target=target,
                    name=f"build/compass_qgemm_L1_b{nb}",
                    tune=False, verify=False, num_threads=32,
                    bits=self.bits, g=4,
                    group_size=self.group_size,
                    act_group_size=self.act_group_size,
                    out_dtype="float32", m_groups=-1,
                    zero_point=True,
                )
                func_big, _ = codegen_big.compile(M_big, N, K)
                self._func_qgemm_batched[nb] = func_big
                logger.info(f"[COMPASS] TMAC QGEMM L1 (M={M_big}, N=1, threads=32) for {nb} blocks compiled.")

            self._tvm_device = tvm.cpu(0)

        except Exception as e:
            logger.warning(f"[COMPASS] Failed to compile TMAC operators: {e}")
            import traceback
            traceback.print_exc()
            self._func_preprocessor = None
            self._func_qgemm = None
            self._func_qgemm_batched = {}

    # ========================================================================
    # TMAC CPU Estimation Helpers
    # ========================================================================

    def _preprocess_q_per_head(self, q_chunk: torch.Tensor, kv_head_idx: int) -> tuple:
        """Run compiled TMAC preprocessor for one KV head's Q representative."""
        head_dim = q_chunk.shape[2]
        num_heads = q_chunk.shape[1]
        heads_per_group = num_heads // self._num_kv_heads
        K = head_dim
        N = 1
        g = 4

        q_avg = q_chunk.float().mean(dim=0)  # [num_heads, head_dim]
        start_h = kv_head_idx * heads_per_group
        end_h = start_h + heads_per_group
        q_head = q_avg[start_h:end_h].mean(dim=0, keepdim=True)  # [1, head_dim]

        q_np = q_head.numpy().astype("float32")
        dev = self._tvm_device
        B_tvm = tvm.nd.array(q_np, dev)
        LUT_Scales_tvm = tvm.nd.array(np.zeros((N, K // self.act_group_size), dtype="float32"), dev)
        LUT_Biases_tvm = tvm.nd.array(np.zeros((N, K // self.act_group_size), dtype="float32"), dev)
        QLUT_tvm = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype="int8"), dev)
        self._func_preprocessor(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)
        return LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm

    def _estimate_subblock_score_from_packed(
        self, layer_id, sub_start, sub_end, kv_head_idx,
        qlut_tvm, lut_scales_tvm, lut_biases_tvm,
    ) -> float:
        """Legacy single sub-block estimation (fallback)."""
        fine_grain = sub_end - sub_start
        nge = 8 // self.g
        M_exp = fine_grain * self.bits
        packed_sub = self._k_packed_buffer[layer_id, sub_start:sub_end, kv_head_idx]
        A_np = packed_sub.numpy().reshape(M_exp // self.bm, self._head_dim // self.g, self.bm // nge)

        k_s = self._k_scales_buffer[layer_id, sub_start:sub_end, kv_head_idx:kv_head_idx+1]
        k_z = self._k_zeros_buffer[layer_id, sub_start:sub_end, kv_head_idx:kv_head_idx+1]
        if k_s.dtype == torch.bfloat16:
            k_s = k_s.to(torch.float16)
            k_z = k_z.to(torch.float16)
        k_s_tmac = k_s.transpose(0, 1).unsqueeze(0)
        k_z_tmac = k_z.transpose(0, 1).unsqueeze(0)
        packed_scales = pack_scales_tmac(k_s_tmac, zeros=k_z_tmac, bits=self.bits)
        Scales_np = packed_scales[0, 0].numpy()

        N = 1
        dev = self._tvm_device
        try:
            A_tvm = tvm.nd.array(A_np.astype("uint8"), dev)
            Scales_tvm = tvm.nd.array(Scales_np.astype("float32"), dev)
            C_tvm = tvm.nd.array(np.zeros((N, fine_grain), dtype="float32"), dev)
            self._func_qgemm(A_tvm, qlut_tvm, Scales_tvm, lut_scales_tvm, lut_biases_tvm, C_tvm)
            scores = C_tvm.numpy()
            return float(np.max(np.abs(scores)))
        except Exception as e:
            logger.warning(f"[COMPASS] TMAC packed estimation error (L{layer_id} h={kv_head_idx}): {e}")
            return float('inf')

    def _estimate_head_batched(
        self,
        layer_id: int,
        kv_head_idx: int,
        k_subblocks: list,
        num_blocks: int,
        qlut_tvm: "tvm.nd.NDArray",
        lut_scales_tvm: "tvm.nd.NDArray",
        lut_biases_tvm: "tvm.nd.NDArray",
        head_data: dict = None,
    ) -> list:
        """
        Estimate all sub-block scores for ONE KV head using a single batched qGEMM call.

        L1 batching: all sub-blocks concatenated into large M, N=1.
        Uses pre-computed A_tvm and Scales_tvm from _prepare_head_data.

        Returns:
            List of max-abs scores, one per sub-block.
        """
        num_subs = len(k_subblocks)
        fine_grain = self.FINE_GRAIN

        # Use pre-computed head data if available
        if head_data is not None:
            func_batched = head_data['func']
            A_tvm = head_data['A_tvm']
            Scales_tvm = head_data['Scales_tvm']
            C_tvm = head_data['C_tvm']
            M_compiled = head_data['M_compiled']

            try:
                func_batched(A_tvm, qlut_tvm, Scales_tvm, lut_scales_tvm, lut_biases_tvm, C_tvm)

                # Extract per-sub-block scores from the output
                C_np = C_tvm.numpy()[0]  # [M_compiled // bits]
                scores = []
                for i in range(num_subs):
                    start_idx = i * fine_grain
                    end_idx = start_idx + fine_grain
                    scores.append(float(np.max(np.abs(C_np[start_idx:end_idx]))))
                return scores
            except Exception as e:
                logger.warning(f"[COMPASS] Batched estimation error (L{layer_id} h={kv_head_idx}): {e}")
                return [float('inf')] * num_subs

        # Fallback: compute everything inline (legacy path)
        nge = 8 // self.g
        dev = self._tvm_device

        func_key = num_blocks
        if func_key not in self._func_qgemm_batched:
            valid_keys = [k for k in self._func_qgemm_batched.keys() if k >= num_blocks]
            if valid_keys:
                func_key = min(valid_keys)
            else:
                func_key = max(self._func_qgemm_batched.keys())
        func_batched = self._func_qgemm_batched[func_key]

        subs_per_block = 4096 // fine_grain
        M_compiled = func_key * subs_per_block * self.bm

        first_start = k_subblocks[0][1]
        last_end = k_subblocks[-1][2]
        total_tokens = last_end - first_start
        M_actual = total_tokens * self.bits

        try:
            packed_all = self._k_packed_buffer[layer_id, first_start:last_end, kv_head_idx]
            A_np = packed_all.numpy().reshape(
                M_actual // self.bm, self._head_dim // self.g, self.bm // nge
            )

            k_s = self._k_scales_buffer[layer_id, first_start:last_end, kv_head_idx:kv_head_idx+1]
            k_z = self._k_zeros_buffer[layer_id, first_start:last_end, kv_head_idx:kv_head_idx+1]
            if k_s.dtype == torch.bfloat16:
                k_s = k_s.to(torch.float16)
                k_z = k_z.to(torch.float16)
            k_s_tmac = k_s.transpose(0, 1).unsqueeze(0)
            k_z_tmac = k_z.transpose(0, 1).unsqueeze(0)
            packed_scales = pack_scales_tmac(k_s_tmac, zeros=k_z_tmac, bits=self.bits)
            Scales_np = packed_scales[0, 0].numpy()

            if M_actual < M_compiled:
                A_padded = np.zeros(
                    (M_compiled // self.bm, self._head_dim // self.g, self.bm // nge),
                    dtype="uint8"
                )
                A_padded[:M_actual // self.bm] = A_np
                Scales_padded = np.zeros(
                    (M_compiled // self.bm, self._head_dim // self.group_size,
                     self.bm // self.bits * 2),
                    dtype="float32"
                )
                Scales_padded[:M_actual // self.bm] = Scales_np
            else:
                A_padded = A_np.astype("uint8")
                Scales_padded = Scales_np.astype("float32")

            A_tvm = tvm.nd.array(A_padded, dev)
            Scales_tvm = tvm.nd.array(Scales_padded, dev)
            C_tvm = tvm.nd.array(np.zeros((1, M_compiled // self.bits), dtype="float32"), dev)

            func_batched(A_tvm, qlut_tvm, Scales_tvm, lut_scales_tvm, lut_biases_tvm, C_tvm)

            C_np = C_tvm.numpy()[0]
            scores = []
            for i in range(num_subs):
                start_idx = i * fine_grain
                end_idx = start_idx + fine_grain
                scores.append(float(np.max(np.abs(C_np[start_idx:end_idx]))))
            return scores

        except Exception as e:
            logger.warning(f"[COMPASS] Batched estimation error (L{layer_id} h={kv_head_idx}): {e}")
            return [float('inf')] * num_subs

    def _prepare_head_data(
        self,
        layer_id: int,
        kv_head_idx: int,
        k_subblocks: list,
        num_blocks: int,
    ) -> dict:
        """
        Pre-compute A_tvm, Scales_tvm, C_tvm for one KV head.
        Called once per head before the Q-group loop to avoid redundant work.

        Returns:
            dict with keys: func, A_tvm, Scales_tvm, C_tvm, M_compiled
        """
        fine_grain = self.FINE_GRAIN
        nge = 8 // self.g
        dev = self._tvm_device

        # Select compiled function
        func_key = num_blocks
        if func_key not in self._func_qgemm_batched:
            valid_keys = [k for k in self._func_qgemm_batched.keys() if k >= num_blocks]
            if valid_keys:
                func_key = min(valid_keys)
            else:
                func_key = max(self._func_qgemm_batched.keys())
        func_batched = self._func_qgemm_batched[func_key]

        subs_per_block = 4096 // fine_grain
        M_compiled = func_key * subs_per_block * self.bm

        first_start = k_subblocks[0][1]
        last_end = k_subblocks[-1][2]
        total_tokens = last_end - first_start
        M_actual = total_tokens * self.bits

        # Read packed K for this head (contiguous)
        packed_all = self._k_packed_buffer[layer_id, first_start:last_end, kv_head_idx]
        A_np = packed_all.numpy().reshape(
            M_actual // self.bm, self._head_dim // self.g, self.bm // nge
        )

        # Batch scales/zeros for this head
        k_s = self._k_scales_buffer[layer_id, first_start:last_end, kv_head_idx:kv_head_idx+1]
        k_z = self._k_zeros_buffer[layer_id, first_start:last_end, kv_head_idx:kv_head_idx+1]
        if k_s.dtype == torch.bfloat16:
            k_s = k_s.to(torch.float16)
            k_z = k_z.to(torch.float16)
        k_s_tmac = k_s.transpose(0, 1).unsqueeze(0)
        k_z_tmac = k_z.transpose(0, 1).unsqueeze(0)
        packed_scales = pack_scales_tmac(k_s_tmac, zeros=k_z_tmac, bits=self.bits)
        Scales_np = packed_scales[0, 0].numpy()

        # Pad to compiled M if needed
        if M_actual < M_compiled:
            A_padded = np.zeros(
                (M_compiled // self.bm, self._head_dim // self.g, self.bm // nge),
                dtype="uint8"
            )
            A_padded[:M_actual // self.bm] = A_np
            Scales_padded = np.zeros(
                (M_compiled // self.bm, self._head_dim // self.group_size,
                 self.bm // self.bits * 2),
                dtype="float32"
            )
            Scales_padded[:M_actual // self.bm] = Scales_np
        else:
            A_padded = A_np.astype("uint8")
            Scales_padded = Scales_np.astype("float32")

        A_tvm = tvm.nd.array(A_padded, dev)
        Scales_tvm = tvm.nd.array(Scales_padded, dev)
        C_tvm = tvm.nd.array(np.zeros((1, M_compiled // self.bits), dtype="float32"), dev)

        return {
            'func': func_batched,
            'A_tvm': A_tvm,
            'Scales_tvm': Scales_tvm,
            'C_tvm': C_tvm,
            'M_compiled': M_compiled,
        }

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
        Select blocks using BLASST-style TMAC CPU prediction during prefill.

        Splits Q chunk (4096 tokens) into 32 Q-groups of 128 tokens.
        Each Q-group independently estimates importance of all K sub-blocks
        (128 tokens each) using TMAC qGEMM with BLASST relative threshold:
            score >= m_global + ln(λ)

        Takes union across all Q-groups, then aggregates to 4096-token IO blocks.
        During decode, returns all blocks.
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

        # --- BLASST-style CPU TMAC Sparse Estimation ---
        # Synchronize to ensure offload_prefill_chunk has written packed K data
        torch.cuda.synchronize()

        block_size = ctx.block_size       # 4096
        fine_grain = self.FINE_GRAIN       # 128
        num_q_groups = q.shape[0] // fine_grain  # typically 32
        ln_lambda = math.log(self.lambda_threshold)  # e.g. ln(0.001) = -6.91
        num_fine_per_block = block_size // fine_grain  # 32

        # Build K sub-block index: [(cpu_block_id, sub_start, sub_end), ...]
        k_subblocks = []
        for bid in available_blocks:
            for si in range(num_fine_per_block):
                start = bid * block_size + si * fine_grain
                k_subblocks.append((bid, start, start + fine_grain))

        if not k_subblocks:
            return available_blocks

        selected_subblock_indices = set()
        q_cpu = q.cpu()
        num_blocks = len(available_blocks)

        # Try L1 batched path: batch all sub-blocks per head per Q-group
        use_batched = bool(self._func_qgemm_batched) and (
            num_blocks in self._func_qgemm_batched or
            num_blocks <= max(self._func_qgemm_batched.keys(), default=0)
        )
        # Pre-compute per-head TVM data ONCE (A_tvm, Scales_tvm, C_tvm)
        # This avoids repeating pack_scales_tmac/numpy/tvm.nd.array 256 times
        head_data_cache = {}  # kv_h -> dict
        if use_batched:
            for kv_h in range(self._num_kv_heads):
                try:
                    head_data_cache[kv_h] = self._prepare_head_data(
                        ctx.layer_id, kv_h, k_subblocks, num_blocks
                    )
                except Exception as e:
                    logger.warning(f"[COMPASS] Head data prep failed (h={kv_h}): {e}")
                    head_data_cache[kv_h] = None

        for q_grp in range(num_q_groups):
            q_group = q_cpu[q_grp * fine_grain : (q_grp + 1) * fine_grain]

            # Preprocess Q group → QLUT per KV head
            try:
                per_head_qluts = [
                    self._preprocess_q_per_head(q_group, h)
                    for h in range(self._num_kv_heads)
                ]
            except Exception as e:
                logger.warning(f"[COMPASS] Q preprocess failed (q_grp={q_grp}): {e}")
                selected_subblock_indices.update(range(len(k_subblocks)))
                continue

            # Per-head estimation
            for kv_h in range(self._num_kv_heads):
                lut_s, lut_b, qlut = per_head_qluts[kv_h]

                if use_batched and head_data_cache.get(kv_h) is not None:
                    # L1 Batched with pre-computed head data
                    scores = self._estimate_head_batched(
                        ctx.layer_id, kv_h, k_subblocks, num_blocks,
                        qlut, lut_s, lut_b,
                        head_data=head_data_cache[kv_h],
                    )
                elif use_batched:
                    # L1 Batched fallback (prep failed, compute inline)
                    scores = self._estimate_head_batched(
                        ctx.layer_id, kv_h, k_subblocks, num_blocks,
                        qlut, lut_s, lut_b,
                    )
                else:
                    # Legacy: per-sub-block qGEMM
                    scores = []
                    for _, sub_s, sub_e in k_subblocks:
                        score = self._estimate_subblock_score_from_packed(
                            ctx.layer_id, sub_s, sub_e, kv_h,
                            qlut, lut_s, lut_b,
                        )
                        scores.append(score)

                # BLASST relative threshold: score >= m_global + ln(λ)
                valid_scores = [s for s in scores if s > -float('inf') and s < float('inf')]
                if not valid_scores:
                    selected_subblock_indices.update(range(len(k_subblocks)))
                    continue

                m_global = max(valid_scores)
                threshold = m_global + ln_lambda

                for idx, score in enumerate(scores):
                    if score >= threshold:
                        selected_subblock_indices.add(idx)

        # Aggregate: 128-token sub-block → 4096-token IO chunk
        selected_chunks = set()
        for idx in selected_subblock_indices:
            selected_chunks.add(k_subblocks[idx][0])

        selected = [b for b in available_blocks if b in selected_chunks]

        # Logging
        self._stats_selected_blocks += len(selected)
        n_total_sub = len(k_subblocks)
        n_sel_sub = len(selected_subblock_indices)
        density_block = len(selected) / max(len(available_blocks), 1) * 100
        density_sub = n_sel_sub / max(n_total_sub, 1) * 100
        logger.info(
            f"[COMPASS] layer={ctx.layer_id}, chunk={ctx.query_chunk_idx}, "
            f"blocks: {len(selected)}/{len(available_blocks)} ({density_block:.1f}%), "
            f"sub-blocks: {n_sel_sub}/{n_total_sub} ({density_sub:.1f}%), "
            f"λ={self.lambda_threshold}"
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

                # Save zeros for TMAC estimation
                if self._k_zeros_buffer is not None and k_zeros is not None:
                    self._k_zeros_buffer[layer_id, start_idx:start_idx + write_len].copy_(
                        k_zeros[:write_len], non_blocking=True
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
