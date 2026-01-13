import torch
from flash_attn import flash_attn_func

def Full_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
    attention_mask = None,
):
    # flash_attn_func expects shape: (batch, seqlen, nheads, headdim)
    # Input shape: (batch, nheads, seqlen, headdim)
    q = query_states.transpose(1, 2)  # (batch, seqlen, nheads, headdim)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    # flash_attn_func returns (batch, seqlen, nheads, headdim)
    attn_output = flash_attn_func(q, k, v, causal=causal)

    # Convert back to (batch, nheads, seqlen, headdim)
    return attn_output.transpose(1, 2)