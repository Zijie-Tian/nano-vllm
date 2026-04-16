#!/usr/bin/env python3
"""Calibrate frequency-domain statistics for COMPASS TriAttention.

Runs a single forward pass on plain text input, captures per-layer query states,
inverts RoPE, and writes the stats format consumed by `compass.src.TriAttention`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from compass.src.triattention_utils import (
    HeadFrequencyStats,
    invert_rope,
    rotate_half,
    save_head_frequency_stats,
    to_complex_pairs,
)


def _find_attention_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    backbone = getattr(model, "model", model)
    layer_list = getattr(backbone, "layers", None)
    if layer_list is None:
        raise RuntimeError("Cannot locate transformer layers. Expected model.model.layers.")
    layers = []
    for layer_module in layer_list:
        attn = getattr(layer_module, "self_attn", None)
        if attn is None:
            raise RuntimeError("Layer missing self_attn attribute.")
        layers.append(attn)
    return layers


def calibrate_triattention_stats(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    max_length: int = 32768,
    device: str = "cuda",
    attn_implementation: str = "flash_attention_2",
) -> None:
    device_obj = torch.device(device)
    dtype = torch.bfloat16

    print(f"Loading model: {model_name_or_path}", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if device_obj.type == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if device_obj.type != "cuda":
        model = model.to(device_obj)
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    rope_style = "half" if "llama" in getattr(config, "model_type", "") else "half"

    attn_layers = _find_attention_layers(model)
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "rotary_emb"):
        rotary = backbone.rotary_emb
    else:
        rotary = attn_layers[0].rotary_emb
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    print(f"Reading input: {input_path}", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = input_ids.to(device_obj)
    seq_len = input_ids.shape[1]
    print(f"Tokenized length: {seq_len}", file=sys.stderr)

    position_ids = torch.arange(seq_len, device=device_obj).unsqueeze(0)
    probe = torch.zeros(1, seq_len, head_dim, device=device_obj, dtype=dtype)
    cos_table, sin_table = rotary(probe, position_ids)

    captured_q: Dict[int, torch.Tensor] = {}

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            attn = module
            bsz, q_len, _ = hidden_states.shape
            q = attn.q_proj(hidden_states)
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            pos_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            p = torch.zeros(1, q_len, head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            cos, sin = rotary(p, pos_ids)
            q_rot = (q * cos.unsqueeze(1)) + (rotate_half(q, style=rope_style) * sin.unsqueeze(1))
            q_rot = q_rot * attn_scale
            captured_q[layer_idx] = q_rot.detach()
        return hook_fn

    handles = [attn.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True) for layer_idx, attn in enumerate(attn_layers)]

    print("Running forward pass...", file=sys.stderr)
    with torch.no_grad():
        model(input_ids)
    print("Forward pass complete.", file=sys.stderr)

    for handle in handles:
        handle.remove()

    print("Computing frequency statistics...", file=sys.stderr)
    sampled_heads: List[Tuple[int, int]] = []
    stats_map = {}

    for layer_idx in range(num_layers):
        q_rot = captured_q.get(layer_idx)
        if q_rot is None:
            continue

        cos = cos_table[:, :seq_len, :].unsqueeze(1)
        sin = sin_table[:, :seq_len, :].unsqueeze(1)
        q_base = invert_rope(q_rot, cos, sin, attn_scale, style=rope_style)

        for head_idx in range(num_heads):
            q_head = q_base[0, head_idx]
            q_complex = to_complex_pairs(q_head, style=rope_style)
            sampled_heads.append((layer_idx, head_idx))
            stats_map[(layer_idx, head_idx)] = HeadFrequencyStats(
                q_mean_complex=q_complex.mean(dim=0),
                q_abs_mean=q_complex.abs().mean(dim=0),
            )

        del captured_q[layer_idx]

    rope_scaling = getattr(config, "rope_scaling", {}) or {}
    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or getattr(config, "rope_type", "default") or "default"
    metadata = {
        "num_traces": 1,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": attn_implementation,
        "rope_style": rope_style,
        "rope_type": rope_type,
    }

    out = Path(output_path)
    save_head_frequency_stats(out, sampled_heads, stats_map, metadata)
    print(f"Saved stats to {out} ({len(sampled_heads)} heads)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate COMPASS TriAttention stats from plain text.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--input", required=True, help="Plain text file for calibration.")
    parser.add_argument("--output", required=True, help="Output .pt stats file.")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()
    calibrate_triattention_stats(
        model_name_or_path=args.model,
        input_path=args.input,
        output_path=args.output,
        max_length=args.max_length,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()
