import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Non-interleaved RoPE (used by Llama, Qwen, etc.)"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved RoPE (used by GLM-4, etc.)

    Args:
        x: [seq_len, num_heads, head_dim]
        cos: [seq_len, 1, head_dim // 2]
        sin: [seq_len, 1, head_dim // 2]

    x is reshaped to [seq_len, num_heads, head_dim // 2, 2] where:
    - x[..., 0] are even positions
    - x[..., 1] are odd positions
    """
    rot_dim = x.shape[-1]
    # x_shaped: [seq_len, num_heads, rot_dim // 2, 2]
    x_shaped = x.float().reshape(*x.shape[:-1], rot_dim // 2, 2)
    # x_0, x_1: [seq_len, num_heads, rot_dim // 2]
    x_0 = x_shaped[..., 0]
    x_1 = x_shaped[..., 1]
    # cos/sin: [seq_len, 1, rot_dim // 2] - broadcasts to num_heads
    x_out = torch.stack([
        x_0 * cos - x_1 * sin,
        x_1 * cos + x_0 * sin,
    ], dim=-1)
    return x_out.flatten(-2).to(x.dtype)


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


class GLM4RotaryEmbedding(nn.Module):
    """
    GLM-4 RoPE with interleaved rotation and partial rotation.

    GLM-4 uses:
    - Interleaved rotation (pairs adjacent elements, not first/second half)
    - rope_ratio to scale base: base = 10000 * rope_ratio
    - Partial rotation: only rotates first rotary_dim elements, rest pass through
    - rotary_dim = head_dim // 2 (only half of head_dim is rotated)
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim  # GLM-4: rotary_dim = head_dim // 2
        # inv_freq shape: [rotary_dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rotary_dim // 2]
        cos = freqs.cos()
        sin = freqs.sin()
        # cache shape [max_pos, 1, rotary_dim // 2, 2]
        cache = torch.stack((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key.

        Args:
            positions: [seq_len]
            query: [seq_len, num_heads, head_dim]
            key: [seq_len, num_kv_heads, head_dim]

        Returns:
            Rotated query and key with same shapes as input.
        """
        cache = self.cos_sin_cache[positions]  # [seq_len, 1, rotary_dim // 2, 2]
        cos = cache[..., 0]  # [seq_len, 1, rotary_dim // 2]
        sin = cache[..., 1]  # [seq_len, 1, rotary_dim // 2]

        # Split into rotated and pass-through parts
        q_rot = query[..., :self.rotary_dim]
        q_pass = query[..., self.rotary_dim:]
        k_rot = key[..., :self.rotary_dim]
        k_pass = key[..., self.rotary_dim:]

        # Apply interleaved RoPE to rotated part
        q_rot = apply_rotary_emb_interleaved(q_rot, cos, sin)
        k_rot = apply_rotary_emb_interleaved(k_rot, cos, sin)

        # Concatenate rotated and pass-through parts
        query = torch.cat([q_rot, q_pass], dim=-1)
        key = torch.cat([k_rot, k_pass], dim=-1)

        return query, key


# Cache for RoPE instances (keyed by hashable parameters)
_rope_cache: dict[tuple, nn.Module] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
    is_interleaved: bool = False,
):
    # Create hashable cache key
    if rope_scaling is None:
        cache_key = (head_size, rotary_dim, max_position, base, None, is_interleaved)
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type == "llama3":
            cache_key = (
                head_size, rotary_dim, max_position, base, "llama3",
                rope_scaling["factor"],
                rope_scaling["low_freq_factor"],
                rope_scaling["high_freq_factor"],
                rope_scaling["original_max_position_embeddings"],
                is_interleaved,
            )
        else:
            cache_key = (head_size, rotary_dim, max_position, base, rope_type, is_interleaved)

    if cache_key in _rope_cache:
        return _rope_cache[cache_key]

    if rope_scaling is None:
        if is_interleaved:
            rope = GLM4RotaryEmbedding(head_size, rotary_dim, max_position, base)
        else:
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
