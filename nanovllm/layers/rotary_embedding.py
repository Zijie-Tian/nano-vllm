import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class Llama3RotaryEmbedding(nn.Module):
    """
    Llama 3 RoPE with special frequency scaling.

    Llama 3 uses a piecewise frequency adjustment:
    - High frequencies (short wavelengths): unchanged
    - Low frequencies (long wavelengths): scaled down by factor
    - Medium frequencies: smoothly interpolated
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # Compute base inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # Apply Llama3 scaling
        inv_freq = self._compute_llama3_inv_freq(
            inv_freq,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        )

        # Build cos/sin cache
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_llama3_inv_freq(
        self,
        inv_freq: torch.Tensor,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
    ) -> torch.Tensor:
        """
        Apply Llama3 frequency scaling.

        - wavelength > low_freq_wavelen: scale down by factor (long range, needs interpolation)
        - wavelength < high_freq_wavelen: keep unchanged (short range, high fidelity)
        - in between: smooth interpolation
        """
        old_context_len = original_max_position_embeddings

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq

        # Low frequency: scale down by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

        # Medium frequency: smooth interpolation
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama + smooth_factor * inv_freq
        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# Cache for RoPE instances (keyed by hashable parameters)
_rope_cache: dict[tuple, nn.Module] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    # Create hashable cache key
    if rope_scaling is None:
        cache_key = (head_size, rotary_dim, max_position, base, None)
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type == "llama3":
            cache_key = (
                head_size, rotary_dim, max_position, base, "llama3",
                rope_scaling["factor"],
                rope_scaling["low_freq_factor"],
                rope_scaling["high_freq_factor"],
                rope_scaling["original_max_position_embeddings"],
            )
        else:
            cache_key = (head_size, rotary_dim, max_position, base, rope_type)

    if cache_key in _rope_cache:
        return _rope_cache[cache_key]

    if rope_scaling is None:
        rope = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type == "llama3":
            rope = Llama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            )
        else:
            raise ValueError(f"Unsupported rope_type: {rope_type}")

    _rope_cache[cache_key] = rope
    return rope
