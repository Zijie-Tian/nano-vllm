"""
Custom Qwen3 implementation using only torch and transformers.
This file provides a clean reference implementation for understanding the model computation graph.

Computation Graph:
==================

Input: token_ids [batch, seq_len]
         │
         ▼
    ┌─────────────┐
    │  Embedding  │  embed_tokens: [vocab_size, hidden_size]
    └─────────────┘
         │
         ▼
    hidden_states [batch, seq_len, hidden_size]
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   Decoder Layer (x N)                   │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │              Self Attention Block                 │  │
    │  │                                                   │  │
    │  │  input_layernorm (RMSNorm)                        │  │
    │  │         │                                         │  │
    │  │         ▼                                         │  │
    │  │  ┌─────────────────────────────────────────────┐  │  │
    │  │  │            Qwen3Attention                   │  │  │
    │  │  │  Q = q_proj(x) → q_norm → reshape           │  │  │
    │  │  │  K = k_proj(x) → k_norm → reshape           │  │  │
    │  │  │  V = v_proj(x) → reshape                    │  │  │
    │  │  │         │                                   │  │  │
    │  │  │         ▼                                   │  │  │
    │  │  │  Q, K = apply_rotary_pos_emb(Q, K, cos, sin)│  │  │
    │  │  │         │                                   │  │  │
    │  │  │         ▼                                   │  │  │
    │  │  │  attn_output = attention(Q, K, V)           │  │  │
    │  │  │         │                                   │  │  │
    │  │  │         ▼                                   │  │  │
    │  │  │  output = o_proj(attn_output)               │  │  │
    │  │  └─────────────────────────────────────────────┘  │  │
    │  │         │                                         │  │
    │  │         ▼                                         │  │
    │  │  hidden_states = residual + attn_output           │  │
    │  └───────────────────────────────────────────────────┘  │
    │                        │                                │
    │                        ▼                                │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │                  MLP Block                        │  │
    │  │                                                   │  │
    │  │  post_attention_layernorm (RMSNorm)               │  │
    │  │         │                                         │  │
    │  │         ▼                                         │  │
    │  │  ┌─────────────────────────────────────────────┐  │  │
    │  │  │              Qwen3MLP                       │  │  │
    │  │  │  gate = gate_proj(x)                        │  │  │
    │  │  │  up = up_proj(x)                            │  │  │
    │  │  │  output = down_proj(silu(gate) * up)        │  │  │
    │  │  └─────────────────────────────────────────────┘  │  │
    │  │         │                                         │  │
    │  │         ▼                                         │  │
    │  │  hidden_states = residual + mlp_output            │  │
    │  └───────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────┐
    │    norm     │  final RMSNorm
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   lm_head   │  [hidden_size, vocab_size]
    └─────────────┘
         │
         ▼
    logits [batch, seq_len, vocab_size]
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3RMSNorm(nn.Module):
    """RMSNorm implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim] or similar
            position_ids: Position indices [batch, seq_len]

        Returns:
            cos, sin: [batch, seq_len, head_dim]
        """
        # inv_freq: [dim/2]
        # position_ids: [batch, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float()  # [1, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [batch, 1, seq_len]

        # freqs: [batch, dim/2, seq_len]
        freqs = inv_freq_expanded @ position_ids_expanded
        # freqs: [batch, seq_len, dim/2]
        freqs = freqs.transpose(1, 2)

        # Duplicate for full head_dim: [batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim]
        sin: [batch, seq_len, head_dim]

    Returns:
        q_embed, k_embed with same shapes as inputs
    """
    # Unsqueeze for broadcasting: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """
    Qwen3 Multi-Head Attention with Grouped Query Attention (GQA) support.

    Data Flow:
    ---------
    hidden_states [batch, seq_len, hidden_size]
            │
            ├──► q_proj ──► q_norm ──► reshape ──► Q [batch, num_heads, seq_len, head_dim]
            ├──► k_proj ──► k_norm ──► reshape ──► K [batch, num_kv_heads, seq_len, head_dim]
            └──► v_proj ──► reshape ──► V [batch, num_kv_heads, seq_len, head_dim]
                    │
                    ▼
            apply_rotary_pos_emb(Q, K)
                    │
                    ▼
            attention(Q, K, V) ──► attn_output [batch, num_heads, seq_len, head_dim]
                    │
                    ▼
            reshape ──► o_proj ──► output [batch, seq_len, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.layer_idx = layer_idx

        # Scaling factor
        self.scaling = head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # QK normalization (Qwen3 specific)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = Qwen3RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_qkv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, kv_seq_len] (causal mask)
            past_key_value: (k_cache, v_cache) from previous steps
            use_cache: Whether to return updated cache
            output_qkv: Whether to output Q, K, V tensors for debugging

        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: Updated cache (if use_cache=True)
            qkv_dict: {"q": Q, "k": K, "v": V} (if output_qkv=True)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # === QKV Projections ===
        q = self.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # === QK Normalization (Qwen3 specific) ===
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # === Rotary Position Embeddings ===
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # === KV Cache Update ===
        if past_key_value is not None:
            k_cache, v_cache = past_key_value
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_past_key_value = (k, v) if use_cache else None

        # === Grouped Query Attention (expand KV heads if needed) ===
        if self.num_kv_groups > 1:
            # Repeat KV for each query group
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # === Attention Computation (using SDPA for memory efficiency) ===
        # Use PyTorch's scaled_dot_product_attention which can use FlashAttention backend
        # is_causal only works when q_len == kv_len (prefill), not during decode
        q_len, kv_len = q.shape[2], k.shape[2]
        is_causal = (q_len == kv_len) and (q_len > 1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scaling,
        )  # [batch, num_heads, seq_len, head_dim]

        # === Output Projection ===
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)  # [batch, seq_len, hidden_size]
        output = self.o_proj(attn_output)

        # Optional QKV output for debugging
        qkv_dict = None
        if output_qkv:
            qkv_dict = {
                "q": q,  # [batch, num_heads, seq_len, head_dim] (post-RoPE)
                "k": k,  # [batch, num_heads, kv_seq_len, head_dim] (post-RoPE, expanded)
                "v": v,  # [batch, num_heads, kv_seq_len, head_dim] (expanded)
            }

        return output, new_past_key_value, qkv_dict


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP with SwiGLU activation.

    Data Flow:
    ---------
    hidden_states [batch, seq_len, hidden_size]
            │
            ├──► gate_proj ──► gate [batch, seq_len, intermediate_size]
            │
            └──► up_proj ──► up [batch, seq_len, intermediate_size]
                    │
                    ▼
            silu(gate) * up
                    │
                    ▼
            down_proj ──► output [batch, seq_len, hidden_size]
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class Qwen3DecoderLayer(nn.Module):
    """Single Qwen3 Decoder Layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention LayerNorm
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

        # Self-attention
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            layer_idx=layer_idx,
        )

        # Post-attention LayerNorm
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, bias=mlp_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_qkv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            attention_mask: Causal attention mask
            past_key_value: KV cache for this layer
            use_cache: Whether to return updated cache
            output_qkv: Whether to output Q, K, V for debugging

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: Updated cache
            qkv_dict: QKV tensors (if output_qkv=True)
        """
        # === Self Attention Block ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, new_past_key_value, qkv_dict = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_qkv=output_qkv,
        )

        hidden_states = residual + attn_output

        # === MLP Block ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_past_key_value, qkv_dict


class Qwen3Model(nn.Module):
    """Qwen3 Transformer Model (without LM head)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                mlp_bias=mlp_bias,
                layer_idx=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final LayerNorm
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_qkv_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[dict]]:
        """
        Args:
            input_ids: [batch, seq_len]
            position_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] or pre-computed 4D mask
            past_key_values: List of (k, v) tuples for each layer
            use_cache: Whether to return new cache
            output_qkv_layers: List of layer indices to output QKV for

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            new_past_key_values: Updated cache
            qkv_outputs: {layer_idx: qkv_dict}
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Attention mask (create causal mask if not provided)
        if attention_mask is None or attention_mask.dim() == 2:
            kv_seq_len = seq_len + (past_key_values[0][0].shape[2] if past_key_values else 0)
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=input_ids.device),
                diagonal=kv_seq_len - seq_len + 1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_seq_len]

        # Initialize cache list
        new_past_key_values = [] if use_cache else None
        qkv_outputs = {} if output_qkv_layers else None

        # Decoder layers
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            output_qkv = output_qkv_layers is not None and i in output_qkv_layers

            hidden_states, new_kv, qkv_dict = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                output_qkv=output_qkv,
            )

            if use_cache:
                new_past_key_values.append(new_kv)
            if qkv_dict is not None:
                qkv_outputs[i] = qkv_dict

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_past_key_values, qkv_outputs


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 Model with Language Modeling head."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        # Transformer model
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
        )

        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_qkv_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[dict]]:
        """
        Args:
            input_ids: [batch, seq_len]
            ... (same as Qwen3Model)

        Returns:
            logits: [batch, seq_len, vocab_size]
            past_key_values: Updated KV cache
            qkv_outputs: QKV tensors for specified layers
        """
        hidden_states, new_past_key_values, qkv_outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_qkv_layers=output_qkv_layers,
        )

        logits = self.lm_head(hidden_states)

        return logits, new_past_key_values, qkv_outputs

    @classmethod
    def from_pretrained(cls, model_path: str, dtype: torch.dtype = torch.float16) -> "Qwen3ForCausalLM":
        """
        Load weights from a pretrained Qwen3 model.

        Args:
            model_path: Path to model directory containing config.json and model weights
            dtype: Data type for model weights

        Returns:
            Initialized Qwen3ForCausalLM model
        """
        import json
        import os
        from safetensors.torch import load_file

        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Create model
        model = cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            head_dim=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
            max_position_embeddings=config.get("max_position_embeddings", 32768),
            rope_theta=config.get("rope_theta", 10000.0),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            attention_bias=config.get("attention_bias", False),
            mlp_bias=config.get("mlp_bias", False),
            tie_word_embeddings=config.get("tie_word_embeddings", True),
        )

        # Load weights
        weight_files = sorted([
            f for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        ])

        state_dict = {}
        for wf in weight_files:
            state_dict.update(load_file(os.path.join(model_path, wf)))

        # Load into model
        model.load_state_dict(state_dict, strict=False)

        # Tie lm_head weights to embed_tokens if configured
        if model.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight

        model = model.to(dtype)

        return model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        past_key_values = None
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if past_key_values is None:
                current_input = generated
            else:
                current_input = generated[:, -1:]

            logits, past_key_values, _ = self(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = logits[:, -1, :]
            if temperature > 0 and do_sample:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


def print_computation_graph():
    """Print the computation graph for reference."""
    print(__doc__)


if __name__ == "__main__":
    print_computation_graph()
