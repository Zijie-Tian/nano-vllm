"""
TriAttention policy.

TriAttention is a standalone policy that keeps Q/K in pre-RoPE form across the
Attention layer, KV cache, and CPU offload buffers. RoPE is applied lazily
inside this policy immediately before attention kernels are launched.
"""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import torch

from nanovllm.layers.rotary_embedding import (
    apply_rotary_emb,
    apply_rotary_emb_interleaved,
)
from nanovllm.utils.context import get_context

from .policy import SparsePolicy, PolicyContext
from .triattention_decode_selector import (
    TriAttentionDecodeConfig,
    TriAttentionDecodeSelector,
)

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class TriAttentionPolicy(SparsePolicy):
    """
    Full-attention policy with lazy RoPE inside policy compute methods.

    Relative to FULL, the difference is semantic rather than sparsity:
    - model layers pass raw pre-RoPE Q/K into Attention
    - KV cache / CPU offload persist raw K
    - this policy applies RoPE right before flash attention kernels
    """

    supports_prefill = True
    supports_decode = True
    apply_rope_in_attention = True

    def __init__(
        self,
        stats_path: str | None = None,
        kv_budget: int = 2048,
        window_size: int = 128,
        score_aggregation: str = "mean",
        sparse_normalize_scores: bool = False,
        offset_max_length: int = 65536,
        score_chunk_max_tokens: int = 4096,
        protect_prefill: bool = False,
        include_prefill_in_budget: bool = True,
    ):
        self._stats_total_blocks = 0
        self._stats_num_chunks = 0
        env_config = TriAttentionDecodeConfig.from_env()
        self._decode_selector = TriAttentionDecodeSelector(
            TriAttentionDecodeConfig(
                stats_path=(
                    Path(stats_path).expanduser()
                    if stats_path
                    else env_config.stats_path
                ),
                kv_budget=kv_budget if stats_path is not None else env_config.kv_budget,
                window_size=window_size if stats_path is not None else env_config.window_size,
                score_aggregation=(
                    score_aggregation
                    if stats_path is not None
                    else env_config.score_aggregation
                ),
                normalize_scores=(
                    sparse_normalize_scores
                    if stats_path is not None
                    else env_config.normalize_scores
                ),
                protect_prefill=(
                    protect_prefill
                    if stats_path is not None
                    else env_config.protect_prefill
                ),
                include_prefill_in_budget=(
                    include_prefill_in_budget
                    if stats_path is not None
                    else env_config.include_prefill_in_budget
                ),
                offset_max_length=(
                    offset_max_length
                    if stats_path is not None
                    else env_config.offset_max_length
                ),
                score_chunk_max_tokens=(
                    score_chunk_max_tokens
                    if stats_path is not None
                    else env_config.score_chunk_max_tokens
                ),
                disable_mlr=env_config.disable_mlr,
                disable_trig=env_config.disable_trig,
            )
        )

    def _get_rope_context(self):
        context = get_context()
        if context.positions is None or context.rotary_emb is None:
            raise RuntimeError(
                "TriAttention requires context.positions and context.rotary_emb"
            )
        return context, context.positions, context.rotary_emb

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

    def _rope_style(self, rotary_emb) -> str:
        return "interleaved" if hasattr(rotary_emb, "rotary_dim") else "half"

    def _rope_omega(self, rotary_emb, device: torch.device) -> torch.Tensor:
        if hasattr(rotary_emb, "inv_freq") and rotary_emb.inv_freq is not None:
            return rotary_emb.inv_freq.to(device=device, dtype=torch.float32)
        if hasattr(rotary_emb, "rotary_dim"):
            cache = rotary_emb.cos_sin_cache[1]
            cos = cache[..., 0].squeeze(0)
            sin = cache[..., 1].squeeze(0)
        else:
            cos_sin = rotary_emb.cos_sin_cache[1].squeeze(0)
            cos, sin = cos_sin.chunk(2, dim=-1)
        return torch.atan2(sin.float(), cos.float()).to(device=device, dtype=torch.float32)

    def _current_round_start(self, positions: torch.Tensor) -> int:
        if positions.numel() == 0:
            return 0
        return int(positions.reshape(-1)[-1].item())

    def _select_decode_keep_indices(
        self,
        *,
        key_states: torch.Tensor,
        layer_id: int,
        positions: torch.Tensor,
        rotary_emb,
        num_attention_heads: int,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        selector = self._decode_selector
        if not selector.enabled:
            return torch.arange(key_states.shape[0], device=key_states.device, dtype=torch.long)
        return selector.select_indices(
            key_states=key_states,
            layer_id=layer_id,
            round_start=self._current_round_start(positions),
            num_attention_heads=num_attention_heads,
            key_positions=positions,
            omega=self._rope_omega(rotary_emb, key_states.device),
            rope_style=self._rope_style(rotary_emb),
            prefix_length=prefix_length,
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
                (0, offload_engine.k_cache_gpu.shape[-2], offload_engine.k_cache_gpu.shape[-1])
            )
            return empty, empty.clone()

        slot = offload_engine.decode_load_slots[0]
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
            block_k, block_v = offload_engine.get_kv_for_slot(slot)
            valid_tokens = (
                last_block_valid_tokens
                if block_idx == len(cpu_block_table) - 1 and last_block_valid_tokens < block_size
                else block_k.shape[1]
            )
            hist_k_parts.append(block_k.squeeze(0)[:valid_tokens].clone())
            hist_v_parts.append(block_v.squeeze(0)[:valid_tokens].clone())
            offload_engine.record_slot_compute_done(slot)
        return torch.cat(hist_k_parts, dim=0), torch.cat(hist_v_parts, dim=0)

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        if ctx.layer_id == 0 and available_blocks:
            self._stats_total_blocks += len(available_blocks)
            self._stats_num_chunks += 1
            logger.debug(
                f"[TriAttention] chunk={ctx.query_chunk_idx}, "
                f"blocks={len(available_blocks)}, density=100.0%"
            )
        return available_blocks

    def reset_stats(self) -> None:
        self._stats_total_blocks = 0
        self._stats_num_chunks = 0

    def get_density_stats(self) -> dict:
        return {
            "total_available_blocks": self._stats_total_blocks,
            "total_selected_blocks": self._stats_total_blocks,
            "num_chunks": self._stats_num_chunks,
            "overall_density": 1.0,
        }

    def print_density_stats(self) -> None:
        stats = self.get_density_stats()
        logger.info(
            f"[TriAttention Policy] Density Stats: chunks={stats['num_chunks']}, "
            f"blocks={stats['total_available_blocks']}, density=100.0%"
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
                    block_tables[seq_idx, :num_prefix_blocks].to(dtype=torch.long).tolist()
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
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        if block_tables is None:
            raise RuntimeError(
                "TriAttention GPU-only decode requires block_tables for raw KV materialization"
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
            keep_indices = self._select_decode_keep_indices(
                key_states=hist_k,
                layer_id=layer_id,
                positions=hist_positions,
                rotary_emb=rotary_emb,
                num_attention_heads=q.shape[1],
            )
            selected_k = hist_k.index_select(0, keep_indices)
            selected_v = hist_v.index_select(0, keep_indices)
            selected_pos = hist_positions.index_select(0, keep_indices)
            selected_k_rot = self._apply_rope(
                selected_k,
                selected_pos,
                rotary_emb,
            ).unsqueeze(0)
            decode_o, _ = flash_attn_with_lse(
                q_rot_batched,
                selected_k_rot,
                selected_v.unsqueeze(0),
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

        _, positions, rotary_emb = self._get_rope_context()
        logger.debug(
            f"[DEBUG] TriAttention.compute_chunked_prefill called, "
            f"layer={layer_id}, chunk={current_chunk_idx}, num_tokens={num_tokens}, "
            f"selected_blocks={len(selected_blocks)}"
        )

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
                        prev_pos = self._block_positions(
                            block_idx,
                            prev_k.shape[1],
                            offload_engine.block_size,
                            q.device,
                            positions.dtype,
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
                        prev_pos = self._block_positions(
                            block_idx,
                            prev_k.shape[1],
                            offload_engine.block_size,
                            q.device,
                            positions.dtype,
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
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        context, positions, rotary_emb = self._get_rope_context()
        compute_stream = offload_engine.compute_stream
        compute_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(compute_stream):
            q_rot_batched = self._apply_rope(q, positions, rotary_emb).unsqueeze(1)
        torch.cuda.default_stream().wait_stream(compute_stream)

        cpu_block_table = selected_blocks
        if layer_id == 0:
            logger.debug(
                f"TriAttention decode: selected_blocks={len(selected_blocks)}, "
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
        keep_indices = self._select_decode_keep_indices(
            key_states=full_k,
            layer_id=layer_id,
            positions=full_positions,
            rotary_emb=rotary_emb,
            num_attention_heads=q.shape[1],
            prefix_length=total_prefill_tokens,
        )
        selected_k = full_k.index_select(0, keep_indices)
        selected_v = full_v.index_select(0, keep_indices)
        selected_pos = full_positions.index_select(0, keep_indices)
        selected_k_rot = self._apply_rope(
            selected_k,
            selected_pos,
            rotary_emb,
        ).unsqueeze(0)
        o_acc, _ = flash_attn_with_lse(
            q_rot_batched,
            selected_k_rot,
            selected_v.unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=False,
        )

        torch.cuda.default_stream().wait_stream(compute_stream)
        return o_acc

    def _decode_ring_buffer_pipeline(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: list,
        load_slots: list,
        offload_engine: "OffloadEngine",
        block_size: int,
        last_block_valid_tokens: int,
        layer_id: int,
        softmax_scale: float,
        rotary_emb,
        position_dtype: torch.dtype,
    ):
        from nanovllm.ops.chunked_attention import (
            flash_attn_with_lse_flashinfer as flash_attn_with_lse,
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )

        num_blocks = len(cpu_block_table)
        if num_blocks == 0 or not load_slots:
            return None, None

        o_acc, lse_acc = None, None
        num_slots = len(load_slots)
        compute_stream = offload_engine.compute_stream

        num_preload = min(num_slots, num_blocks)
        for i in range(num_preload):
            cpu_block_id = cpu_block_table[i]
            offload_engine.load_to_slot_layer(
                load_slots[i],
                layer_id,
                cpu_block_id,
                chunk_idx=cpu_block_id,
                is_prefill=False,
            )

        for block_idx in range(num_blocks):
            current_slot = load_slots[block_idx % num_slots]
            offload_engine.wait_slot_layer(current_slot)

            with torch.cuda.stream(compute_stream):
                prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)

                is_last_block = block_idx == num_blocks - 1
                valid_tokens = (
                    last_block_valid_tokens
                    if is_last_block and last_block_valid_tokens < block_size
                    else prev_k.shape[1]
                )
                if valid_tokens < prev_k.shape[1]:
                    prev_k = prev_k[:, :valid_tokens, :, :]
                    prev_v = prev_v[:, :valid_tokens, :, :]

                prev_pos = self._block_positions(
                    block_idx,
                    valid_tokens,
                    block_size,
                    q_batched.device,
                    position_dtype,
                )
                prev_k_rot = self._apply_rope(
                    prev_k.squeeze(0), prev_pos, rotary_emb
                ).unsqueeze(0)

                prev_o, prev_lse = flash_attn_with_lse(
                    q_batched,
                    prev_k_rot,
                    prev_v,
                    softmax_scale=softmax_scale,
                    causal=False,
                )

                offload_engine.record_slot_compute_done(current_slot)

            next_block_idx = block_idx + num_slots
            if next_block_idx < num_blocks:
                next_cpu_block_id = cpu_block_table[next_block_idx]
                offload_engine.load_to_slot_layer(
                    current_slot,
                    layer_id,
                    next_cpu_block_id,
                    chunk_idx=next_cpu_block_id,
                    is_prefill=False,
                )

            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = merge_attention_outputs(
                        o_acc, lse_acc, prev_o, prev_lse
                    )

        return o_acc, lse_acc

    def offload_prefill_chunk(
        self,
        offload_engine: "OffloadEngine",
        layer_id: int,
        cpu_block_id: int,
        num_tokens: int,
        **kwargs,
    ) -> None:
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
        return "TriAttentionPolicy()"
