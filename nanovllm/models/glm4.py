"""GLM-4 model implementation for nano-vllm."""
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.registry import register_model


class GLM4Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 1048576,
        head_dim: int = 128,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,  # GLM-4 has QKV bias
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,  # GLM-4 has no output bias
        )
        # GLM-4 only rotates half of head_dim
        rotary_dim = self.head_dim // 2
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_interleaved=True,  # GLM-4 uses interleaved RoPE
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class GLM4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,  # GLM-4 has no MLP bias
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class GLM4DecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        # GLM-4 config field mapping
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'multi_query_group_num', num_heads)
        head_dim = getattr(config, 'kv_channels', hidden_size // num_heads)
        max_position = getattr(config, 'seq_length', 1048576)
        rope_ratio = getattr(config, 'rope_ratio', 1)
        rope_theta = 10000 * rope_ratio  # GLM-4 uses rope_ratio to scale base
        intermediate_size = getattr(config, 'ffn_hidden_size', getattr(config, 'intermediate_size', None))
        rms_norm_eps = getattr(config, 'layernorm_epsilon', 1e-5)

        self.self_attn = GLM4Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = GLM4MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GLM4Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        vocab_size = getattr(config, 'padded_vocab_size', config.vocab_size)
        num_layers = getattr(config, 'num_layers', config.num_hidden_layers)
        rms_norm_eps = getattr(config, 'layernorm_epsilon', 1e-5)

        self.embed_tokens = VocabParallelEmbedding(vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GLM4DecoderLayer(config) for _ in range(num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_model("ChatGLMModel", "ChatGLMForConditionalGeneration")
class ChatGLMForCausalLM(nn.Module):
    """
    GLM-4 model for causal language modeling.

    Weight mapping from HuggingFace to nanovllm:
    - transformer.embedding.word_embeddings → model.embed_tokens
    - transformer.encoder.layers.X.input_layernorm → model.layers.X.input_layernorm
    - transformer.encoder.layers.X.self_attention.query_key_value → model.layers.X.self_attn.qkv_proj (split q/k/v)
    - transformer.encoder.layers.X.self_attention.dense → model.layers.X.self_attn.o_proj
    - transformer.encoder.layers.X.post_attention_layernorm → model.layers.X.post_attention_layernorm
    - transformer.encoder.layers.X.mlp.dense_h_to_4h → model.layers.X.mlp.gate_up_proj (split gate/up)
    - transformer.encoder.layers.X.mlp.dense_4h_to_h → model.layers.X.mlp.down_proj
    - transformer.encoder.final_layernorm → model.norm
    - transformer.output_layer → lm_head
    """
    packed_modules_mapping = {
        # QKV is merged in GLM-4 as query_key_value
        "query_key_value": ("qkv_proj", None),  # Special handling needed
        # MLP gate and up are merged as dense_h_to_4h
        "dense_h_to_4h": ("gate_up_proj", None),  # Special handling needed
    }

    # Weight name mapping for loader
    hf_to_nanovllm_mapping = {
        "transformer.embedding.word_embeddings": "model.embed_tokens",
        "transformer.encoder.final_layernorm": "model.norm",
        "transformer.output_layer": "lm_head",
    }

    def __init__(self, config) -> None:
        super().__init__()
        vocab_size = getattr(config, 'padded_vocab_size', config.vocab_size)
        self.config = config
        self.model = GLM4Model(config)
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size)
        # GLM-4 does not tie embeddings
        # if getattr(config, 'tie_word_embeddings', False):
        #     self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
