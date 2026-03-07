"""
BLASST sparse attention policy for chunked prefill.

BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

Key insight: Use online softmax running max to dynamically skip unimportant blocks.
Skip condition: local_max - running_max < ln(λ)
Threshold formula: λ = a / L (inverse with sequence length)

Implementation notes:
- IO communication: Same as FullAttention (load all blocks)
- Compute decision: Skip merging sub-blocks based on BLASST condition
- Fine-grained: Both query and KV processed at configurable granularity (default 128)
- Running max: Extracted from LSE (log-sum-exp) of flash attention
"""

import os
import logging
import math
import torch
from typing import List, TYPE_CHECKING

from .policy import SparsePolicy, PolicyContext

if TYPE_CHECKING:
    from nanovllm.kvcache.offload_engine import OffloadEngine
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)

# Global debug toggle for dumping mask buffers
DEBUG_DUMP_BLASST_MASK = False
DUMP_BASE_DIR = "results/chunked_mask"


class BLASSTPolicy(SparsePolicy):
    """
    BLASST sparse attention policy with fine-grained granularity.
    """

    # BLASST supports both prefill and decode
    supports_prefill = True
    supports_decode = True

    def __init__(self, a: int = 16384, fixed_lambda: float = None, granularity: int = 128):
        self.a = a
        self.fixed_lambda = fixed_lambda
        self.granularity = granularity
        self._stats_num_chunks = 0
        self._stats_skipped_subblocks = 0
        self._stats_total_subblocks = 0

    def _get_lambda(self, seq_len: int) -> float:
        if self.fixed_lambda is not None:
            return self.fixed_lambda
        return self.a / max(seq_len, 1)

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx: PolicyContext,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        if ctx.layer_id == 0:
            self._stats_num_chunks += 1
            logger.debug(f"[BLASST] chunk={ctx.query_chunk_idx}, "
                        f"available={len(available_blocks)}, selected={len(available_blocks)}")
        return available_blocks

    def reset_stats(self) -> None:
        self._stats_num_chunks = 0
        self._stats_skipped_subblocks = 0
        self._stats_total_subblocks = 0

    def get_stats(self) -> dict:
        skip_rate = 0.0
        if self._stats_total_subblocks > 0:
            skip_rate = self._stats_skipped_subblocks / self._stats_total_subblocks
        return {
            "num_chunks": self._stats_num_chunks,
            "skipped_subblocks": self._stats_skipped_subblocks,
            "total_subblocks": self._stats_total_subblocks,
            "skip_rate": skip_rate,
        }

    def compute_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float,
        layer_id: int, block_tables=None,
    ) -> torch.Tensor:
        raise NotImplementedError("BLASST policy only supports chunked prefill mode.")

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
            merge_attention_outputs_flashinfer as merge_attention_outputs,
        )
        from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill

        total_seq_len = len(seq) if seq else num_tokens
        lambda_val = self._get_lambda(total_seq_len)
        ln_lambda = math.log(lambda_val)
        
        q_len = q.shape[0]
        num_heads = q.shape[1]
        compute_stream = offload_engine.compute_stream
        q_input = q.unsqueeze(0).transpose(1, 2).contiguous()

        historical_o = None
        historical_lse = None

        collect_density = (layer_id == 0)
        compute_density_sum = 0.0
        required_kv_density_sum = 0.0
        num_density_measurements = 0
        layer_masks = {}

        TRITON_BLOCK_M, TRITON_BLOCK_N = 128, 64
        grid_0 = (q_len + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
        grid_1 = num_heads
        num_kv_subblocks = kvcache_manager.block_size // TRITON_BLOCK_N

        def get_mask_buffer(is_causal=False, kv_len_override=None):
            num_sub = num_kv_subblocks if kv_len_override is None else (kv_len_override + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
            # Initialize to 1s (Compute all by default)
            mask = torch.ones((grid_0, grid_1, num_sub), device=q.device, dtype=torch.int8)
            
            if is_causal:
                # Apply Block-level Causal Mask: Set to 0 where KV strictly in future of Q
                for q_idx in range(grid_0):
                    q_end_pos = (q_idx + 1) * TRITON_BLOCK_M
                    for kv_idx in range(num_sub):
                        kv_start_pos = kv_idx * TRITON_BLOCK_N
                        if kv_start_pos >= q_end_pos:
                            mask[q_idx, :, kv_idx] = 0
            return mask

        # 1. Process CPU Offloaded Blocks (Historical)
        cpu_block_table = selected_blocks
        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_slots = len(load_slots)
            num_blocks = len(cpu_block_table)

            num_preload = min(num_slots, num_blocks)
            for i in range(num_preload):
                offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

            for block_idx in range(num_blocks):
                current_slot = load_slots[block_idx % num_slots]
                cpu_block_id = cpu_block_table[block_idx]
                offload_engine.wait_slot_layer(current_slot)

                with torch.cuda.stream(compute_stream):
                    prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                    k_input = prev_k.transpose(1, 2).contiguous()
                    v_input = prev_v.transpose(1, 2).contiguous()

                    # Historical blocks are fully visible
                    mask_buffer = get_mask_buffer(is_causal=False)
                    
                    out, lse = blasst_chunked_prefill(
                        q=q_input, k=k_input, v=v_input,
                        threshold_ln_lambda=ln_lambda,
                        lse_in=historical_lse,  # Pass historical LSE to preserve running max
                        mask_buffer=mask_buffer
                    )
                    
                    compute_stream.synchronize()
                    if collect_density:
                        compute_density_sum += mask_buffer.float().mean().item()
                        required_kv_mask = mask_buffer.any(dim=0)
                        required_kv_density_sum += required_kv_mask.float().mean().item()
                        num_density_measurements += 1
                    if DEBUG_DUMP_BLASST_MASK:
                        layer_masks[f"kvchunk_{cpu_block_id}"] = mask_buffer.cpu()

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
                    offload_engine.load_to_slot_layer(load_slots[next_block_idx % num_slots], 
                                                     layer_id, cpu_block_table[next_block_idx])

        # 2. Process Current Prefill Chunk (GPU Buffer, Causal)
        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            k_curr_input = k_curr.transpose(1, 2).contiguous()
            v_curr_input = v_curr.transpose(1, 2).contiguous()
            
            # Causal mask for the diagonal chunk
            curr_mask_buffer = get_mask_buffer(is_causal=True, kv_len_override=num_tokens)

            out_curr, lse_curr = blasst_chunked_prefill(
                q=q_input, k=k_curr_input, v=v_curr_input,
                threshold_ln_lambda=ln_lambda,
                lse_in=historical_lse,
                mask_buffer=curr_mask_buffer
            )
            
            compute_stream.synchronize()
            if collect_density:
                # To be completely accurate with density, we should probably ignore the causal 0s,
                # but for overall system IO/compute view, including them is fine.
                compute_density_sum += curr_mask_buffer.float().mean().item()
                required_kv_mask = curr_mask_buffer.any(dim=0)
                required_kv_density_sum += required_kv_mask.float().mean().item()
                num_density_measurements += 1
            if DEBUG_DUMP_BLASST_MASK:
                layer_masks["kvchunk_current"] = curr_mask_buffer.cpu()

            block_o = out_curr.transpose(1, 2).contiguous()
            block_lse = lse_curr

            if historical_o is None:
                final_o = block_o
            else:
                final_o, _ = merge_attention_outputs(historical_o, historical_lse, block_o, block_lse)

        # 3. Finalize
        if DEBUG_DUMP_BLASST_MASK and layer_masks:
            dump_dir = os.path.join(DUMP_BASE_DIR, f"q_chunk_{current_chunk_idx}")
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(layer_masks, os.path.join(dump_dir, f"layer_{layer_id}.pt"))

        if layer_id == 0 and num_density_measurements > 0:
            avg_comp = compute_density_sum / num_density_measurements
            avg_req = required_kv_density_sum / num_density_measurements
            logger.info(f"[BLASST] Chunk {current_chunk_idx} Stats: "
                       f"Compute Density={avg_comp*100:.2f}%, "
                       f"Required KV Density={avg_req*100:.2f}%")

        torch.cuda.default_stream().wait_stream(compute_stream)
        return final_o.squeeze(0)

    def compute_chunked_decode(
        self, q: torch.Tensor, layer_id: int, softmax_scale: float,
        offload_engine: "OffloadEngine", kvcache_manager: "KVCacheManager",
        seq: "Sequence", selected_blocks: List[int],
    ) -> torch.Tensor:
        from .full_policy import FullAttentionPolicy
        return FullAttentionPolicy().compute_chunked_decode(
            q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq, selected_blocks
        )

    def __repr__(self) -> str:
        if self.fixed_lambda is not None:
            return f"BLASSTPolicy(fixed_lambda={self.fixed_lambda}, granularity={self.granularity})"
        return f"BLASSTPolicy(a={self.a}, granularity={self.granularity})"
