import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# GLM-4 weight name mappings
GLM4_NAME_MAPPING = {
    "transformer.embedding.word_embeddings": "model.embed_tokens",
    "transformer.encoder.final_layernorm": "model.norm",
    "transformer.output_layer": "lm_head",
}

GLM4_LAYER_MAPPING = {
    "self_attention.query_key_value": "self_attn.qkv_proj",
    "self_attention.dense": "self_attn.o_proj",
    "mlp.dense_h_to_4h": "mlp.gate_up_proj",
    "mlp.dense_4h_to_h": "mlp.down_proj",
}


def convert_glm4_weight_name(weight_name: str) -> tuple[str, str | None]:
    """
    Convert GLM-4 weight name to nanovllm format.

    Returns:
        tuple: (converted_name, shard_id) where shard_id is used for packed modules
               Returns (None, None) for weights that should be skipped
    """
    # Skip rotary embedding weights (we use our own RoPE implementation)
    if "rotary_pos_emb" in weight_name:
        return None, None

    # Check direct mappings first
    for glm_name, nano_name in GLM4_NAME_MAPPING.items():
        if weight_name.startswith(glm_name):
            return weight_name.replace(glm_name, nano_name), None

    # Handle layer weights: transformer.encoder.layers.X.xxx
    layer_match = re.match(r"transformer\.encoder\.layers\.(\d+)\.(.+)", weight_name)
    if layer_match:
        layer_idx = layer_match.group(1)
        remainder = layer_match.group(2)

        # Handle packed modules (QKV and gate_up)
        for glm_subname, nano_subname in GLM4_LAYER_MAPPING.items():
            if remainder.startswith(glm_subname):
                suffix = remainder[len(glm_subname) :]  # .weight or .bias
                new_name = f"model.layers.{layer_idx}.{nano_subname}{suffix}"

                # Determine shard_id for packed modules
                if "qkv_proj" in nano_subname:
                    return new_name, "qkv"  # Special marker for GLM4 QKV
                elif "gate_up_proj" in nano_subname:
                    return new_name, "gate_up"  # Special marker for GLM4 gate_up
                else:
                    return new_name, None

        # Handle non-packed layer weights (layernorms)
        new_name = f"model.layers.{layer_idx}.{remainder}"
        return new_name, None

    # No mapping found, return original
    return weight_name, None


def load_glm4_qkv(param: nn.Parameter, loaded_weight: torch.Tensor, config):
    """Load GLM-4 merged QKV weights by splitting into q, k, v."""
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "multi_query_group_num", num_heads)
    head_dim = getattr(config, "kv_channels", config.hidden_size // num_heads)

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Split QKV: [q_size + kv_size + kv_size, hidden_size]
    q, k, v = loaded_weight.split([q_size, kv_size, kv_size], dim=0)

    # Load each part using the weight_loader
    weight_loader = getattr(param, "weight_loader")
    weight_loader(param, q, "q")
    weight_loader(param, k, "k")
    weight_loader(param, v, "v")


def load_glm4_gate_up(param: nn.Parameter, loaded_weight: torch.Tensor, config):
    """Load GLM-4 merged gate_up weights by splitting into gate, up."""
    ffn_hidden_size = getattr(
        config, "ffn_hidden_size", getattr(config, "intermediate_size", None)
    )

    # Split gate_up: [ffn_hidden_size * 2, hidden_size]
    gate, up = loaded_weight.split([ffn_hidden_size, ffn_hidden_size], dim=0)

    # Load each part using the weight_loader
    weight_loader = getattr(param, "weight_loader")
    weight_loader(param, gate, 0)  # gate_proj is shard 0
    weight_loader(param, up, 1)  # up_proj is shard 1


def is_glm4_model(model: nn.Module) -> bool:
    """Check if the model is a GLM-4 model."""
    return model.__class__.__name__ in ("ChatGLMForCausalLM",)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    is_glm4 = is_glm4_model(model)
    config = getattr(model, "config", None)

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)

                # GLM-4 specific handling
                if is_glm4:
                    param_name, shard_id = convert_glm4_weight_name(weight_name)

                    # Skip weights that don't need to be loaded
                    if param_name is None:
                        continue

                    if shard_id == "qkv":
                        param = model.get_parameter(param_name)
                        load_glm4_qkv(param, loaded_weight, config)
                        continue
                    elif shard_id == "gate_up":
                        param = model.get_parameter(param_name)
                        load_glm4_gate_up(param, loaded_weight, config)
                        continue
                    else:
                        # Regular weight, use converted name
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        continue

                # Original loading logic for other models
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
