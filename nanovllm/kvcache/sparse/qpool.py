"""
QPoolPolicy: Query-pooled multi-subblock DMAS sparse attention.

Built on PREROPE semantics (pre-RoPE KV cache) with sub-block level sparse
attention via Distance-Modulated Amplitude Scoring (DMAS).

Key features:
- 256-token sub-block K summaries cached at offload time
- Single-Q (last-8-token mean) DMAS scoring for block/sub-block selection
- Fixed 70% sub-block density per block for compute-phase sparse attention
- Multi-Q sub-block union density instrumentation (stats only)
"""

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import torch

from nanovllm.layers.rotary_embedding import (
    apply_rotary_emb,
    apply_rotary_emb_interleaved,
)
from nanovllm.utils.context import get_context

from .policy import SparsePolicy

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.kvcache.offload_engine import OffloadEngine

logger = logging.getLogger(__name__)


class QPoolPolicy(SparsePolicy):
    """
    PREROPE semantics with sub-block DMAS sparse attention.
    """

    supports_prefill = True
    supports_decode = True
    apply_rope_in_attention = True

    def __init__(self, top_p: float = 0.95):
        self._top_p = top_p
        self._stats_total_blocks = 0
        self._stats_num_chunks = 0
        self._stats_selected_blocks = 0

        # DMAS: per-layer, per-block K summary cache
        self._k_summary_cache: dict = {}
        self._num_layers = 0
        self._num_kv_heads = 0
        self._head_dim = 0
        self._thetas: Optional[torch.Tensor] = None

        # Sub-block selection masks for compute-phase sparse attention
        # {layer_id: {cpu_block_id: torch.BoolTensor [num_sub_blocks]}}
        self._sub_block_masks: dict = {}
        self._sub_block_size = 256

        # Global Q union masks: accumulate multi-Q sub-block selections across
        # ALL query chunks. {layer_id: {cpu_block_id: torch.BoolTensor}}
        self._global_q_union_masks: dict = {}

    def _get_rope_context(self):
        context = get_context()
        if context.positions is None or context.rotary_emb is None:
            raise RuntimeError(
                "QPoolPolicy requires context.positions and context.rotary_emb"
            )
        return context, context.positions, context.rotary_emb

    def initialize(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
        device: torch.device = None,
    ) -> None:
        del num_cpu_blocks, dtype, device
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._k_summary_cache = {layer_id: {} for layer_id in range(num_layers)}

    def _get_thetas(self, device: torch.device) -> torch.Tensor:
        """Lazy-load RoPE frequency thetas from context."""
        if self._thetas is not None:
            return self._thetas.to(device)
        context = get_context()
        if hasattr(context, "rotary_emb") and context.rotary_emb is not None:
            rotary_emb = context.rotary_emb
            if hasattr(rotary_emb, "inv_freq"):
                self._thetas = rotary_emb.inv_freq.detach().cpu()
            else:
                d_pairs = self._head_dim // 2
                self._thetas = 10000.0 ** (
                    -torch.arange(0, d_pairs, dtype=torch.float32) * 2.0 / self._head_dim
                )
        else:
            d_pairs = self._head_dim // 2
            self._thetas = 10000.0 ** (
                -torch.arange(0, d_pairs, dtype=torch.float32) * 2.0 / self._head_dim
            )
        return self._thetas.to(device)

    def _apply_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x

        if hasattr(rotary_emb, "rotary_dim"):
            cache = rotary_emb.cos_sin_cache[positions]
            cos = cache[..., 0]
            sin = cache[..., 1]
            x_rot = x[..., : rotary_emb.rotary_dim]
            x_pass = x[..., rotary_emb.rotary_dim :]
            x_rot = apply_rotary_emb_interleaved(x_rot, cos, sin)
            return torch.cat([x_rot, x_pass], dim=-1)

        cos_sin = rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return apply_rotary_emb(x, cos, sin)

    def _apply_rope_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._apply_rope(q, positions, rotary_emb),
            self._apply_rope(k, positions, rotary_emb),
        )

    def _block_positions(
        self,
        block_index: int,
        num_tokens: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        start = block_index * block_size
        return torch.arange(start, start + num_tokens, device=device, dtype=dtype)

    def _apply_sub_block_mask(
        self,
        prev_k: torch.Tensor,
        prev_v: torch.Tensor,
        cpu_block_id: int,
        layer_id: int,
        block_idx: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply sub-block mask to extract important tokens from a loaded block."""
        sb_masks = self._sub_block_masks.get(layer_id, {})
        mask = sb_masks.get(cpu_block_id)
        if mask is None or not mask.any() or mask.all():
            num_tokens = prev_k.shape[1]
            positions = self._block_positions(
                block_idx, num_tokens, block_size, device, dtype
            )
            return prev_k, prev_v, positions

        num_sub_blocks = mask.shape[0]
        selected_indices = []
        for sb_idx in range(num_sub_blocks):
            if mask[sb_idx]:
                start = sb_idx * self._sub_block_size
                end = min(start + self._sub_block_size, prev_k.shape[1])
                selected_indices.extend(range(start, end))

        if not selected_indices:
            num_tokens = prev_k.shape[1]
            positions = self._block_positions(
                block_idx, num_tokens, block_size, device, dtype
            )
            return prev_k, prev_v, positions

        indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
        prev_k_sel = prev_k[:, indices, :, :]
        prev_v_sel = prev_v[:, indices, :, :]

        block_start = block_idx * block_size
        positions = torch.tensor(
            [block_start + i for i in selected_indices],
            device=device,
            dtype=dtype,
        )
        return prev_k_sel, prev_v_sel, positions

    def _materialize_kv_range(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        start_pos: int,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if end_pos <= start_pos:
            empty_shape = (0, k_cache.shape[2], k_cache.shape[3])
            return (
                k_cache.new_empty(empty_shape),
                v_cache.new_empty(empty_shape),
            )

        block_size = k_cache.shape[1]
        start_block = start_pos // block_size
        end_block = (end_pos - 1) // block_size
        block_ids = block_table[start_block : end_block + 1].to(dtype=torch.long)
        gathered_k = k_cache.index_select(0, block_ids).reshape(
            -1, k_cache.shape[2], k_cache.shape[3]
        )
        gathered_v = v_cache.index_select(0, block_ids).reshape(
            -1, v_cache.shape[2], v_cache.shape[3]
        )
        start_offset = start_pos - start_block * block_size
        length = end_pos - start_pos
        return (
            gathered_k[start_offset : start_offset + length],
            gathered_v[start_offset : start_offset + length],
        )

    def _materialize_prefilled_cpu_history(
        self,
        *,
        cpu_block_table: List[int],
        offload_engine: "OffloadEngine",
        layer_id: int,
        block_size: int,
        last_block_valid_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cpu_block_table:
            empty = offload_engine.k_cache_gpu.new_empty(
                (
                    0,
                    offload_engine.k_cache_gpu.shape[-2],
                    offload_engine.k_cache_gpu.shape[-1],
                )
            )
            return empty, empty.clone()

        slot = offload_engine.decode_load_slots[0]
        compute_stream = offload_engine.compute_stream
        hist_k_parts = []
        hist_v_parts = []
        for block_idx, cpu_block_id in enumerate(cpu_block_table):
            offload_engine.load_to_slot_layer(
                slot,
                layer_id,
                cpu_block_id,
                chunk_idx=cpu_block_id,
                is_prefill=False,
            )
            offload_engine.wait_slot_layer(slot)
            with torch.cuda.stream(compute_stream):
                block_k, block_v = offload_engine.get_kv_for_slot(slot)
                valid_tokens = (
                    last_block_valid_tokens
                    if block_idx == len(cpu_block_table) - 1
                    and last_block_valid_tokens < block_size
                    else block_k.shape[1]
                )
                hist_k_parts.append(block_k.squeeze(0)[:valid_tokens].clone())
                hist_v_parts.append(block_v.squeeze(0)[:valid_tokens].clone())
                offload_engine.record_slot_compute_done(slot)
        torch.cuda.default_stream().wait_stream(compute_stream)
        return torch.cat(hist_k_parts, dim=0), torch.cat(hist_v_parts, dim=0)

    def on_prefill_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """Cache sub-block (256-token) pre-RoPE K summaries for DMAS scoring."""
        SUB_BLOCK_SIZE = 256
        k_block = k_cache[:num_valid_tokens].float()
        d_pairs = self._head_dim // 2
        num_sub_blocks = (num_valid_tokens + SUB_BLOCK_SIZE - 1) // SUB_BLOCK_SIZE
        sub_block_summaries = []

        for sb_idx in range(num_sub_blocks):
            start = sb_idx * SUB_BLOCK_SIZE
            end = min(start + SUB_BLOCK_SIZE, num_valid_tokens)
            sb_k = k_block[start:end]
            sb_len = end - start
            if sb_len == 0:
                continue
            sb_pairs = sb_k.reshape(sb_len, self._num_kv_heads, d_pairs, 2)
            pooled = sb_pairs.mean(dim=0)
            sub_block_summaries.append(pooled)

        if sub_block_summaries:
            self._k_summary_cache[layer_id][cpu_block_id] = torch.stack(
                sub_block_summaries, dim=0
            ).detach().cpu()

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        del offload_engine, k
        layer_id = ctx.layer_id
        num_available = len(available_blocks)
        SUB_BLOCK_SIZE = 256

        if ctx.layer_id == 0 and available_blocks:
            self._stats_total_blocks += num_available
            self._stats_num_chunks += 1

        if not available_blocks or not self._k_summary_cache.get(layer_id):
            if ctx.layer_id == 0 and available_blocks:
                self._stats_selected_blocks += num_available
            return available_blocks

        device = q.device
        d_pairs = self._head_dim // 2
        num_heads = q.shape[1]
        heads_per_group = max(1, num_heads // max(1, self._num_kv_heads))

        # 1. Pool Q into complex pairs [num_heads, d_pairs, 2]
        q_last = q.float()[-8:]
        q_pairs = q_last.mean(dim=0)
        q_pairs = q_pairs.reshape(
            heads_per_group, self._num_kv_heads, d_pairs, 2
        )

        # 2. Load cached sub-block K summaries
        layer_cache = self._k_summary_cache[layer_id]
        k_sub_blocks = []
        sb_counts = []
        missing = False
        for block_id in available_blocks:
            if block_id not in layer_cache:
                missing = True
                break
            sb_tensor = layer_cache[block_id].to(device)
            k_sub_blocks.append(sb_tensor)
            sb_counts.append(sb_tensor.shape[0])

        if missing or not k_sub_blocks:
            if ctx.layer_id == 0:
                self._stats_selected_blocks += num_available
            return available_blocks

        k_stack = torch.cat(k_sub_blocks, dim=0)
        total_sb = k_stack.shape[0]

        # 3. Compute amplitude M and phase phi
        q0 = q_pairs[..., 0]
        q1 = q_pairs[..., 1]
        k0 = k_stack[..., 0]
        k1 = k_stack[..., 1]

        A = torch.einsum("ghp,shp->ghsp", q0, k0) + torch.einsum(
            "ghp,shp->ghsp", q1, k1
        )
        B = torch.einsum("ghp,shp->ghsp", q1, k0) - torch.einsum(
            "ghp,shp->ghsp", q0, k1
        )
        M = torch.sqrt(A**2 + B**2)
        phi = torch.atan2(B, A)

        # 4. EXACT post-RoPE QK dot product
        thetas = self._get_thetas(device)
        block_size = ctx.block_size
        total_kv_len = ctx.total_kv_len
        seq_len = q.shape[0]
        q_pos_val = total_kv_len + seq_len - 1

        sb_deltas = []
        for b_idx in range(num_available):
            num_sb = sb_counts[b_idx]
            for s_idx in range(num_sb):
                sb_mid = b_idx * block_size + s_idx * SUB_BLOCK_SIZE + SUB_BLOCK_SIZE / 2.0
                delta = q_pos_val - sb_mid
                sb_deltas.append(delta)
        deltas = torch.tensor(sb_deltas, device=device, dtype=torch.float32)
        deltas = deltas.view(1, 1, total_sb, 1)
        thetas_exp = thetas.view(1, 1, 1, d_pairs)

        exact_dot = M * torch.cos(deltas * thetas_exp + phi)
        scores = exact_dot.sum(dim=-1)
        max_scores = scores.max(dim=0).values.max(dim=0).values

        # 5. Block selection: keep ALL blocks
        selected = available_blocks

        # ---- Multi-Q sub-block union density analysis ----
        num_q_sub_blocks = (seq_len + SUB_BLOCK_SIZE - 1) // SUB_BLOCK_SIZE
        union_density = None
        per_block_mq_union = {}
        if num_q_sub_blocks > 1:
            q_sb_list = []
            q_sb_mids = []
            for q_sb_idx in range(num_q_sub_blocks):
                start = q_sb_idx * SUB_BLOCK_SIZE
                end = min(start + SUB_BLOCK_SIZE, seq_len)
                q_sb = q.float()[start:end]
                q_sb_pooled = q_sb.mean(dim=0).reshape(
                    heads_per_group, self._num_kv_heads, d_pairs, 2
                )
                q_sb_list.append(q_sb_pooled)
                q_sb_mids.append(total_kv_len + start + (end - start) / 2.0)

            q_stack = torch.stack(q_sb_list, dim=0)
            q0_mq = q_stack[..., 0]
            q1_mq = q_stack[..., 1]

            A_mq = torch.einsum("qghp,shp->qghsp", q0_mq, k0) + torch.einsum(
                "qghp,shp->qghsp", q1_mq, k1
            )
            B_mq = torch.einsum("qghp,shp->qghsp", q1_mq, k0) - torch.einsum(
                "qghp,shp->qghsp", q0_mq, k1
            )
            M_mq = torch.sqrt(A_mq**2 + B_mq**2)
            phi_mq = torch.atan2(B_mq, A_mq)

            sb_deltas_mq = []
            for q_sb_mid in q_sb_mids:
                for b_idx in range(num_available):
                    num_sb = sb_counts[b_idx]
                    for s_idx in range(num_sb):
                        k_sb_mid = (
                            b_idx * block_size
                            + s_idx * SUB_BLOCK_SIZE
                            + SUB_BLOCK_SIZE / 2.0
                        )
                        delta = q_sb_mid - k_sb_mid
                        sb_deltas_mq.append(delta)
            deltas_mq = torch.tensor(sb_deltas_mq, device=device, dtype=torch.float32)
            deltas_mq = deltas_mq.view(num_q_sub_blocks, 1, 1, total_sb, 1)
            thetas_exp_mq = thetas.view(1, 1, 1, 1, d_pairs)

            exact_dot_mq = M_mq * torch.cos(deltas_mq * thetas_exp_mq + phi_mq)
            scores_mq = exact_dot_mq.sum(dim=-1)
            max_scores_mq = scores_mq.max(dim=1).values.max(dim=1).values

            # Per-block multi-Q union
            union_mask_mq = torch.zeros(total_sb, dtype=torch.bool, device=device)
            for i, count in enumerate(sb_counts):
                block_id = available_blocks[i]
                offset_b = sum(sb_counts[:i])
                block_union = torch.zeros(count, dtype=torch.bool, device=device)
                for q_idx in range(num_q_sub_blocks):
                    q_sb_scores_for_block = max_scores_mq[
                        q_idx, offset_b : offset_b + count
                    ]
                    num_keep_q = max(1, count * 7 // 10)
                    _, top_sb_idx = torch.topk(q_sb_scores_for_block, num_keep_q)
                    block_union[top_sb_idx] = True
                per_block_mq_union[block_id] = (
                    block_union.sum().item() / count * 100.0
                )
                union_mask_mq[offset_b : offset_b + count] = block_union
            union_density = union_mask_mq.sum().item() / total_sb * 100.0

            # Accumulate to global Q union
            if layer_id not in self._global_q_union_masks:
                self._global_q_union_masks[layer_id] = {}
            global_union = self._global_q_union_masks[layer_id]
            offset_q = 0
            for i, count in enumerate(sb_counts):
                block_id = available_blocks[i]
                block_mask = union_mask_mq[offset_q : offset_q + count]
                if block_id in global_union:
                    old_mask = global_union[block_id]
                    if old_mask.shape[0] < count:
                        extended = torch.zeros(
                            count, dtype=torch.bool, device=old_mask.device
                        )
                        extended[: old_mask.shape[0]] = old_mask
                        old_mask = extended
                    global_union[block_id] = old_mask | block_mask.cpu()
                else:
                    global_union[block_id] = block_mask.cpu()
                offset_q += count

        global_density = None
        if layer_id in self._global_q_union_masks:
            global_union = self._global_q_union_masks[layer_id]
            global_total_sb = sum(m.shape[0] for m in global_union.values())
            global_selected = sum(m.sum().item() for m in global_union.values())
            global_density = (
                global_selected / global_total_sb * 100.0 if global_total_sb else 0.0
            )

        # ---- Sub-block masks for compute-phase sparse attention ----
        self._sub_block_masks[layer_id] = {}
        offset = 0
        total_sb_selected = 0
        total_sb = 0
        for i, count in enumerate(sb_counts):
            block_id = available_blocks[i]
            sb_scores_block = max_scores[offset : offset + count]
            num_keep = max(2, count * 7 // 10)
            _, top_sb_idx = torch.topk(sb_scores_block, num_keep)
            mask = torch.zeros(count, dtype=torch.bool, device=device)
            mask[top_sb_idx] = True
            self._sub_block_masks[layer_id][block_id] = mask
            total_sb_selected += num_keep
            total_sb += count
            offset += count

        if ctx.layer_id % 10 == 0:
            self._stats_selected_blocks += len(selected)
            block_density = (
                len(selected) / num_available * 100.0 if num_available else 0.0
            )
            sb_density = (
                total_sb_selected / total_sb * 100.0 if total_sb else 0.0
            )
            if union_density is not None:
                gden = f" global_union={global_density:.1f}%" if global_density is not None else ""
                if ctx.query_chunk_idx == 5 and per_block_mq_union:
                    pb_str = ", ".join(
                        f"B{bid}:{pct:.0f}%"
                        for bid, pct in sorted(per_block_mq_union.items())
                    )
                    logger.info(
                        f"[QPool DMAS] L{ctx.layer_id} chunk={ctx.query_chunk_idx}, "
                        f"blocks={len(selected)}/{num_available} ({block_density:.0f}%), "
                        f"sub-blocks={total_sb_selected}/{total_sb} ({sb_density:.0f}%), "
                        f"multi-Q_union={union_density:.1f}% (Q_sbs={num_q_sub_blocks}){gden} | per-block: {pb_str}"
                    )
                else:
                    logger.info(
                        f"[QPool DMAS] L{ctx.layer_id} chunk={ctx.query_chunk_idx}, "
                        f"blocks={len(selected)}/{num_available} ({block_density:.0f}%), "
                        f"sub-blocks={total_sb_selected}/{total_sb} ({sb_density:.0f}%), "
                        f"multi-Q_union={union_density:.1f}% (Q_sbs={num_q_sub_blocks}){gden}"
                    )
            else:
                logger.info(
                    f"[QPool DMAS] L{ctx.layer_id} chunk={ctx.query_chunk_idx}, "
                    f"blocks={len(selected)}/{num_available} ({block_density:.0f}%), "
                    f"sub-blocks={total_sb_selected}/{total_sb} ({sb_density:.0f}%)"
                )
        return selected

    def reset(self) -> None:
        """Reset policy state for new sequence."""
        if hasattr(self, "_k_summary_cache") and self._k_summary_cache:
            for layer_cache in self._k_summary_cache.values():
                layer_cache.clear()
        self._global_q_union_masks = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats_total_blocks = 0
        self._stats_num_chunks = 0
        self._stats_selected_blocks = 0

    def get_density_stats(self) -> dict:
        density = 1.0
        if self._stats_total_blocks > 0:
            density = self._stats_selected_blocks / self._stats_total_blocks
        return {
            "total_available_blocks": self._stats_total_blocks,
            "total_selected_blocks": self._stats_selected_blocks,
            "num_chunks": self._stats_num_chunks,
            "overall_density": density,
        }

    def print_density_stats(self) -> None:
        stats = self.get_density_stats()
        logger.info(
            f"[QPool DMAS Policy] Density Stats: chunks={stats['num_chunks']}, "
            f"selected={stats['total_selected_blocks']}/"
            f"{stats['total_available_blocks']} "
            f"density={stats['overall_density']*100.0:.1f}%"
        )

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
        from flash_attn import flash_attn_varlen_func
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        del layer_id
        _, positions, rotary_emb = self._get_rope_context()

        if block_tables is None:
            q_rot, k_rot = self._apply_rope_qk(q, k, positions, rotary_emb)
            return flash_attn_varlen_func(
                q_rot,
                k_rot,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
            )

        block_size = k.shape[1]
        outputs = []
        q_offset = 0

        q_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        k_lengths = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()

        for seq_idx, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
            q_len = int(q_len)
            k_len = int(k_len)
            q_seq = q[q_offset : q_offset + q_len]
            q_pos = positions[q_offset : q_offset + q_len]
            q_rot_batched = self._apply_rope(q_seq, q_pos, rotary_emb).unsqueeze(0)

            prefix_len = k_len - q_len
            o_acc = None
            lse_acc = None

            if prefix_len > 0:
                num_prefix_blocks = (prefix_len + block_size - 1) // block_size
                prefix_block_ids = (
                    block_tables[seq_idx, :num_prefix_blocks]
                    .to(dtype=torch.long)
                    .tolist()
                )
                for block_idx, block_id in enumerate(prefix_block_ids):
                    block_start = block_idx * block_size
                    block_end = min(block_start + block_size, prefix_len)
                    num_block_tokens = block_end - block_start
                    hist_k = k[block_id, :num_block_tokens]
                    hist_v = v[block_id, :num_block_tokens]
                    hist_pos = torch.arange(
                        block_start,
                        block_end,
                        device=q.device,
                        dtype=q_pos.dtype,
                    )
                    hist_k_rot = self._apply_rope(
                        hist_k, hist_pos, rotary_emb
                    ).unsqueeze(0)
                    hist_o, hist_lse = flash_attn_with_lse(
                        q_rot_batched,
                        hist_k_rot,
                        hist_v.unsqueeze(0),
                        softmax_scale=softmax_scale,
                        causal=False,
                    )
                    if o_acc is None:
                        o_acc, lse_acc = hist_o, hist_lse
                    else:
                        o_acc, lse_acc = merge_attention_outputs(
                            o_acc, lse_acc, hist_o, hist_lse
                        )

            current_k, current_v = self._materialize_kv_range(
                k, v, block_tables[seq_idx], prefix_len, k_len
            )
            current_k_rot = self._apply_rope(
                current_k, q_pos, rotary_emb
            ).unsqueeze(0)
            current_o, current_lse = flash_attn_with_lse(
                q_rot_batched,
                current_k_rot,
                current_v.unsqueeze(0),
                softmax_scale=softmax_scale,
                causal=True,
            )
            if o_acc is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(
                    o_acc, lse_acc, current_o, current_lse
                )

            outputs.append(final_o.squeeze(0))
            q_offset += q_len

        return torch.cat(outputs, dim=0)

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
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
        )

        del layer_id
        if block_tables is None:
            raise RuntimeError(
                "QPoolPolicy GPU-only decode requires block_tables for raw KV materialization"
            )

        _, positions, rotary_emb = self._get_rope_context()
        outputs = []

        for seq_idx, seqlen in enumerate(cache_seqlens.tolist()):
            seqlen = int(seqlen)
            q_seq = q[seq_idx : seq_idx + 1]
            q_pos = positions[seq_idx : seq_idx + 1]
            q_rot_batched = self._apply_rope(q_seq, q_pos, rotary_emb).unsqueeze(0)
            hist_k, hist_v = self._materialize_kv_range(
                k_cache,
                v_cache,
                block_tables[seq_idx],
                0,
                seqlen,
            )
            hist_positions = torch.arange(
                seqlen,
                device=q.device,
                dtype=q_pos.dtype,
            )
            hist_k_rot = self._apply_rope(
                hist_k,
                hist_positions,
                rotary_emb,
            ).unsqueeze(0)
            decode_o, _ = flash_attn_with_lse(
                q_rot_batched,
                hist_k_rot,
                hist_v.unsqueeze(0),
                softmax_scale=softmax_scale,
                causal=False,
            )
            outputs.append(decode_o)

        return torch.cat(outputs, dim=0)

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
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        del k, v, kvcache_manager, current_chunk_idx, seq
        _, positions, rotary_emb = self._get_rope_context()

        compute_stream = offload_engine.compute_stream
        compute_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(compute_stream):
            q_rot_batched = self._apply_rope(q, positions, rotary_emb).unsqueeze(0)

        o_acc = None
        lse_acc = None
        cpu_block_table = selected_blocks

        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_blocks = len(cpu_block_table)

            if len(load_slots) == 1:
                slot = load_slots[0]
                for block_idx in range(num_blocks):
                    cpu_block_id = cpu_block_table[block_idx]
                    offload_engine.load_to_slot_layer(
                        slot, layer_id, cpu_block_id, chunk_idx=cpu_block_id
                    )
                    offload_engine.wait_slot_layer(slot)

                    with torch.cuda.stream(compute_stream):
                        prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
                        prev_k, prev_v, prev_pos = self._apply_sub_block_mask(
                            prev_k, prev_v, cpu_block_id, layer_id, block_idx,
                            offload_engine.block_size, q.device, positions.dtype,
                        )
                        prev_k_rot = self._apply_rope(
                            prev_k.squeeze(0), prev_pos, rotary_emb
                        ).unsqueeze(0)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_rot_batched,
                            prev_k_rot,
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
                        prev_k, prev_v, prev_pos = self._apply_sub_block_mask(
                            prev_k, prev_v, cpu_block_id, layer_id, block_idx,
                            offload_engine.block_size, q.device, positions.dtype,
                        )
                        prev_k_rot = self._apply_rope(
                            prev_k.squeeze(0), prev_pos, rotary_emb
                        ).unsqueeze(0)
                        prev_o, prev_lse = flash_attn_with_lse(
                            q_rot_batched,
                            prev_k_rot,
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

        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            k_curr_rot = self._apply_rope(
                k_curr.squeeze(0), positions[:num_tokens], rotary_emb
            ).unsqueeze(0)
            current_o, current_lse = flash_attn_with_lse(
                q_rot_batched,
                k_curr_rot,
                v_curr,
                softmax_scale=softmax_scale,
                causal=True,
            )

        with torch.cuda.stream(compute_stream):
            if o_acc is None:
                final_o = current_o
            else:
                final_o, _ = merge_attention_outputs(
                    o_acc, lse_acc, current_o, current_lse
                )

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
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
        )

        _, positions, rotary_emb = self._get_rope_context()
        compute_stream = offload_engine.compute_stream
        compute_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(compute_stream):
            q_rot_batched = self._apply_rope(q, positions, rotary_emb).unsqueeze(1)
        torch.cuda.default_stream().wait_stream(compute_stream)

        cpu_block_table = selected_blocks
        if layer_id == 0:
            logger.debug(
                f"QPool decode: selected_blocks={len(selected_blocks)}, "
                f"seq.block_table={list(seq.block_table)}"
            )
        if not cpu_block_table:
            raise RuntimeError(
                "Chunked decode attention failed: no prefilled CPU blocks available"
            )

        block_size = kvcache_manager.block_size
        all_prefilled_blocks = kvcache_manager.get_prefilled_cpu_blocks(seq)
        total_prefill_tokens = kvcache_manager.get_prefill_len(seq)
        last_block_valid_tokens = total_prefill_tokens % block_size
        if last_block_valid_tokens == 0 and total_prefill_tokens > 0:
            last_block_valid_tokens = block_size

        last_prefilled_block = (
            all_prefilled_blocks[-1] if all_prefilled_blocks else None
        )
        selected_contains_last = (
            cpu_block_table and cpu_block_table[-1] == last_prefilled_block
        )
        effective_last_block_tokens = (
            last_block_valid_tokens if selected_contains_last else block_size
        )
        hist_k, hist_v = self._materialize_prefilled_cpu_history(
            cpu_block_table=cpu_block_table,
            offload_engine=offload_engine,
            layer_id=layer_id,
            block_size=block_size,
            last_block_valid_tokens=effective_last_block_tokens,
        )

        seq_len = len(seq)
        decode_pos_in_block = (seq_len - 1) % block_size
        decode_start_pos = kvcache_manager.get_decode_start_pos(seq)
        decode_start_pos_in_block = decode_start_pos % block_size
        num_accumulated = decode_pos_in_block - decode_start_pos_in_block + 1

        decode_k_dense = hist_k.new_empty((0, hist_k.shape[1], hist_k.shape[2]))
        decode_v_dense = hist_v.new_empty((0, hist_v.shape[1], hist_v.shape[2]))
        if num_accumulated > 0:
            if getattr(offload_engine, "is_head_first", False):
                decode_k_dense = offload_engine.decode_k_buffer[
                    layer_id, :, decode_start_pos_in_block : decode_pos_in_block + 1
                ].transpose(0, 1).contiguous()
                decode_v_dense = offload_engine.decode_v_buffer[
                    layer_id, :, decode_start_pos_in_block : decode_pos_in_block + 1
                ].transpose(0, 1).contiguous()
            else:
                decode_k_dense = offload_engine.decode_k_buffer[
                    layer_id, decode_start_pos_in_block : decode_pos_in_block + 1
                ].contiguous()
                decode_v_dense = offload_engine.decode_v_buffer[
                    layer_id, decode_start_pos_in_block : decode_pos_in_block + 1
                ].contiguous()

        full_k = torch.cat([hist_k, decode_k_dense], dim=0)
        full_v = torch.cat([hist_v, decode_v_dense], dim=0)
        if full_k.numel() == 0:
            raise RuntimeError("Chunked decode attention failed: no KV available")

        full_positions = torch.arange(
            full_k.shape[0],
            device=q.device,
            dtype=positions.dtype,
        )
        full_k_rot = self._apply_rope(
            full_k,
            full_positions,
            rotary_emb,
        ).unsqueeze(0)
        o_acc, _ = flash_attn_with_lse(
            q_rot_batched,
            full_k_rot,
            full_v.unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=False,
        )

        torch.cuda.default_stream().wait_stream(compute_stream)
        return o_acc

    def __repr__(self) -> str:
        return "QPoolPolicy()"
