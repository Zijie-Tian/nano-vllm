from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TriAttentionDecodeConfig:
    stats_path: Optional[Path]
    kv_budget: int = 2048
    window_size: int = 128
    score_aggregation: str = "mean"
    normalize_scores: bool = False
    protect_prefill: bool = False
    include_prefill_in_budget: bool = True
    offset_max_length: int = 65536
    score_chunk_max_tokens: int = 4096
    disable_mlr: bool = False
    disable_trig: bool = False

    @classmethod
    def from_env(cls) -> "TriAttentionDecodeConfig":
        stats_path_raw = os.environ.get("TRIATTENTION_STATS_PATH", "").strip()
        stats_path = Path(stats_path_raw).expanduser() if stats_path_raw else None
        return cls(
            stats_path=stats_path,
            kv_budget=int(os.environ.get("TRIATTENTION_KV_BUDGET", "2048")),
            window_size=int(os.environ.get("TRIATTENTION_WINDOW_SIZE", "128")),
            score_aggregation=os.environ.get("TRIATTENTION_SCORE_AGGREGATION", "mean").strip().lower() or "mean",
            normalize_scores=_env_flag("TRIATTENTION_NORMALIZE_SCORES", False),
            protect_prefill=_env_flag("TRIATTENTION_PROTECT_PREFILL", False),
            include_prefill_in_budget=_env_flag("TRIATTENTION_INCLUDE_PREFILL_IN_BUDGET", True),
            offset_max_length=int(os.environ.get("TRIATTENTION_OFFSET_MAX_LENGTH", "65536")),
            score_chunk_max_tokens=int(os.environ.get("TRIATTENTION_SCORE_CHUNK_MAX_TOKENS", "4096")),
            disable_mlr=_env_flag("TRIATTENTION_DISABLE_MLR", False),
            disable_trig=_env_flag("TRIATTENTION_DISABLE_TRIG", False),
        )


@dataclass
class HeadFrequencyStats:
    q_mean_complex: torch.Tensor
    q_abs_mean: torch.Tensor


class TriAttentionDecodeSelector:
    """Decode-time token selector adapted from TriAttention.

    This helper assumes keys are already stored in pre-RoPE form. That matches
    nano-vllm's local TriAttention cache semantics, so no RoPE inversion is
    needed before scoring.
    """

    def __init__(self, config: TriAttentionDecodeConfig):
        self.config = config
        self._loaded = False
        self._disabled_reason: Optional[str] = None
        self.sampled_heads: list[tuple[int, int]] = []
        self.head_stats: dict[tuple[int, int], HeadFrequencyStats] = {}
        self.expected_head_dim: Optional[int] = None
        self.expected_rope_style: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None and self.config.stats_path is not None and self.config.kv_budget > 0

    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def _ensure_loaded(self) -> None:
        if self._loaded or self._disabled_reason is not None:
            return
        if self.config.stats_path is None:
            self._disabled_reason = "stats_path_not_set"
            return
        if not self.config.stats_path.exists():
            self._disabled_reason = f"stats_path_not_found:{self.config.stats_path}"
            logger.warning("TriAttention decode selector disabled: %s", self._disabled_reason)
            return

        payload = torch.load(self.config.stats_path, map_location="cpu")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        stats_blob = None
        if isinstance(payload, dict):
            stats_blob = payload.get("stats") or payload.get("head_stats")
        if not isinstance(stats_blob, dict):
            self._disabled_reason = "unsupported_stats_format"
            logger.warning("TriAttention decode selector disabled: %s", self._disabled_reason)
            return

        sampled_heads = metadata.get("sampled_heads", [])
        self.sampled_heads = [tuple(map(int, item)) for item in sampled_heads]
        self.expected_head_dim = int(metadata["head_dim"]) if "head_dim" in metadata else None
        rope_style = metadata.get("rope_style")
        self.expected_rope_style = str(rope_style) if rope_style is not None else None

        for key, value in stats_blob.items():
            layer_head: Optional[tuple[int, int]] = None
            if isinstance(key, tuple) and len(key) == 2:
                layer_head = (int(key[0]), int(key[1]))
            elif isinstance(key, str) and key.startswith("layer") and "_head" in key:
                layer_part, head_part = key.split("_head", 1)
                layer_head = (int(layer_part.replace("layer", "")), int(head_part))
            if layer_head is None or not isinstance(value, dict):
                continue
            if "q_mean_real" not in value or "q_mean_imag" not in value or "q_abs_mean" not in value:
                continue
            self.head_stats[layer_head] = HeadFrequencyStats(
                q_mean_complex=torch.complex(
                    value["q_mean_real"].to(dtype=torch.float32),
                    value["q_mean_imag"].to(dtype=torch.float32),
                ),
                q_abs_mean=value["q_abs_mean"].to(dtype=torch.float32),
            )

        if not self.head_stats:
            self._disabled_reason = "empty_head_stats"
            logger.warning("TriAttention decode selector disabled: %s", self._disabled_reason)
            return

        if not self.sampled_heads:
            self.sampled_heads = sorted(self.head_stats.keys())
        self._loaded = True
        logger.info(
            "TriAttention decode selector loaded: stats=%s sampled_heads=%d budget=%d window=%d",
            self.config.stats_path,
            len(self.sampled_heads),
            self.config.kv_budget,
            self.config.window_size,
        )

    def select_indices(
        self,
        *,
        key_states: torch.Tensor,
        layer_id: int,
        round_start: int,
        num_attention_heads: int,
        key_positions: torch.Tensor,
        omega: torch.Tensor,
        rope_style: str,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        self._ensure_loaded()
        seq_len = int(key_states.shape[0])
        if seq_len == 0:
            return torch.empty(0, device=key_states.device, dtype=torch.long)
        if not self.enabled or seq_len <= self.config.kv_budget:
            return torch.arange(seq_len, device=key_states.device, dtype=torch.long)
        if self.expected_head_dim is not None and int(key_states.shape[-1]) != self.expected_head_dim:
            logger.warning(
                "TriAttention decode selector fallback: head_dim mismatch local=%d stats=%d",
                key_states.shape[-1],
                self.expected_head_dim,
            )
            return torch.arange(seq_len, device=key_states.device, dtype=torch.long)
        if self.expected_rope_style is not None and rope_style != self.expected_rope_style:
            logger.warning(
                "TriAttention decode selector fallback: rope_style mismatch local=%s stats=%s",
                rope_style,
                self.expected_rope_style,
            )
            return torch.arange(seq_len, device=key_states.device, dtype=torch.long)

        layer_heads = [(layer, head) for layer, head in self.sampled_heads if layer == layer_id]
        if not layer_heads:
            return torch.arange(seq_len, device=key_states.device, dtype=torch.long)

        num_kv_heads = int(key_states.shape[1])
        num_key_value_groups = max(1, num_attention_heads // max(1, num_kv_heads))
        key_positions = key_positions.to(device=key_states.device, dtype=torch.float32)
        head_scores = []

        for _, head in layer_heads:
            stats = self.head_stats.get((layer_id, head))
            if stats is None:
                continue
            kv_head = min(num_kv_heads - 1, head // num_key_value_groups)
            k_values = key_states[:, kv_head, :]
            amp, phi, extra = _compute_frequency_statistics_from_means(
                stats.q_mean_complex.to(device=key_states.device),
                stats.q_abs_mean.to(device=key_states.device),
                k_values,
                style=rope_style,
                disable_mlr=self.config.disable_mlr,
            )
            scores = _score_keys_for_round(
                key_indices=key_positions,
                round_start=round_start,
                amp=amp,
                phi=phi,
                omega=omega.to(device=key_states.device, dtype=torch.float32),
                extra=extra,
                aggregation=self.config.score_aggregation,
                disable_trig=self.config.disable_trig,
            )
            head_scores.append(scores)

        if not head_scores:
            return torch.arange(seq_len, device=key_states.device, dtype=torch.long)

        head_matrix = torch.stack(head_scores, dim=0)
        if self.config.normalize_scores and head_matrix.numel() > 0:
            mean = head_matrix.mean(dim=1, keepdim=True)
            std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            head_matrix = (head_matrix - mean) / std

        protected_mask = torch.zeros(seq_len, device=key_states.device, dtype=torch.bool)
        if self.config.window_size > 0:
            protected_mask[max(0, seq_len - self.config.window_size):] = True
        if self.config.protect_prefill and prefix_length > 0:
            protected_mask[: min(prefix_length, seq_len)] = True
        if protected_mask.any():
            head_matrix[:, protected_mask] = float("inf")

        combined = head_matrix.max(dim=0).values
        keep_count = min(self.config.kv_budget, seq_len)
        return _select_union_based(head_matrix, combined, keep_count)


def _to_complex_pairs(tensor: torch.Tensor, *, style: str) -> torch.Tensor:
    if tensor.size(-1) % 2 != 0:
        raise ValueError("head_dim must be even to form complex pairs")
    work = tensor.to(dtype=torch.float32)
    if style == "interleaved":
        real = work[..., ::2].contiguous()
        imag = work[..., 1::2].contiguous()
        return torch.complex(real, imag)
    half = work.shape[-1] // 2
    return torch.complex(work[..., :half].contiguous(), work[..., half:].contiguous())


def _compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    key_unrot: torch.Tensor,
    *,
    style: str,
    disable_mlr: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = _to_complex_pairs(key_unrot, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    if disable_mlr:
        extra = q_abs_mean.unsqueeze(0) * k_abs
    else:
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def _score_keys_for_round(
    *,
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    aggregation: str,
    disable_trig: bool,
) -> torch.Tensor:
    base_delta = float(round_start) - key_indices.to(dtype=torch.float32)
    phase = base_delta.unsqueeze(1) * omega.view(1, -1) + phi
    base_scores = (amp * torch.cos(phase)).sum(dim=1)
    additive = extra.sum(dim=1)
    combined = additive if disable_trig else (base_scores + additive)
    if aggregation == "max":
        return combined
    return combined


def _select_union_based(per_head_scores: torch.Tensor, combined: torch.Tensor, keep_count: int) -> torch.Tensor:
    candidate_count = int(combined.shape[0])
    if candidate_count <= keep_count:
        return torch.arange(candidate_count, device=combined.device, dtype=torch.long)
    union_mask = torch.zeros(candidate_count, device=combined.device, dtype=torch.bool)
    per_head_quota = min(keep_count, candidate_count)
    for head_scores in per_head_scores:
        top_idx = torch.topk(head_scores, k=per_head_quota, largest=True).indices
        union_mask[top_idx] = True
    union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
    if union_indices.numel() >= keep_count:
        subset_scores = combined.index_select(0, union_indices)
        top_subset = torch.topk(subset_scores, k=keep_count, largest=True).indices
        return torch.sort(union_indices.index_select(0, top_subset)).values
    remaining = keep_count - int(union_indices.numel())
    if remaining > 0:
        residual_scores = combined.clone()
        residual_scores[union_mask] = float("-inf")
        extra_indices = torch.topk(residual_scores, k=min(remaining, candidate_count - int(union_indices.numel())), largest=True).indices
        union_indices = torch.cat([union_indices, extra_indices], dim=0)
    return torch.sort(union_indices).values
