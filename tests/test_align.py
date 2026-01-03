"""
Test attention I/O observation with CPU offload.
Uses hooks to observe attention layer inputs (Q, K, V) and outputs.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

from nanovllm import LLM, SamplingParams
from utils import generate_needle_prompt

# Config
MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
MAX_MODEL_LEN = 32 * 1024
NUM_GPU_BLOCKS = 4
INPUT_LEN = 32 * 1024
BLOCK_SIZE = 1024


def make_attention_io_hook(layer_id: int, hook_type: str = "pre"):
    """
    Create hooks to inspect attention inputs/outputs.

    Hook positions on decoder_layer.self_attn.attn:
    - PRE HOOK inputs:  (q, k, v, ...) - Q/K/V tensors AFTER projection, AFTER RoPE
    - POST HOOK output: attention_output tensor - shape [batch, seq_len, num_heads * head_dim]

    Alternative hook position on decoder_layer.self_attn:
    - PRE HOOK inputs:  (hidden_states, ...) - BEFORE Q/K/V projection
    - POST HOOK output: (attn_output, attn_weights, past_key_value)
    """
    def pre_hook(module, inputs):
        """
        Attention input hook - captures Q, K, V tensors.

        Position: decoder_layer.self_attn.attn (the Attention layer)
        inputs[0] = Q tensor: [batch, seq_len, num_heads, head_dim]
        inputs[1] = K tensor: [batch, seq_len, num_kv_heads, head_dim]
        inputs[2] = V tensor: [batch, seq_len, num_kv_heads, head_dim]
        """
        if len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            print(f"\n[Layer {layer_id}] ATTENTION INPUT (pre-hook on self_attn.attn):")
            print(f"  Q shape: {q.shape}, dtype: {q.dtype}, mean: {q.float().mean():.4f}")
            print(f"  K shape: {k.shape}, dtype: {k.dtype}, mean: {k.float().mean():.4f}")
            print(f"  V shape: {v.shape}, dtype: {v.dtype}, mean: {v.float().mean():.4f}")
        return None  # Don't modify inputs

    def post_hook(module, inputs, output):
        """
        Attention output hook - captures attention result.

        Position: decoder_layer.self_attn.attn (the Attention layer)
        output = attention_output tensor: [batch, seq_len, num_heads * head_dim]

        NOTE: This is the output AFTER attention computation but BEFORE output projection.
        """
        # output can be tensor or tuple depending on implementation
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        print(f"\n[Layer {layer_id}] ATTENTION OUTPUT (post-hook on self_attn.attn):")
        print(f"  Output shape: {attn_output.shape}, dtype: {attn_output.dtype}")
        print(f"  Output mean: {attn_output.float().mean():.4f}, std: {attn_output.float().std():.4f}")
        return None  # Don't modify output

    return pre_hook if hook_type == "pre" else post_hook


# Main
llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=MAX_MODEL_LEN,
    enable_cpu_offload=True,
    kvcache_block_size=BLOCK_SIZE,
    num_gpu_blocks=NUM_GPU_BLOCKS,
    dtype="float16",
)

# ============================================================
# Register I/O hooks to inspect attention inputs/outputs
# ============================================================
# Only enable for first 2 layers to avoid excessive output
io_hooks = []
for layer_idx, decoder_layer in enumerate(llm.model_runner.model.model.layers):
    if layer_idx >= 2:  # Only first 2 layers
        break

    # Position: decoder_layer.self_attn.attn (the Attention layer)
    # - PRE hook sees: Q, K, V tensors (AFTER projection, AFTER RoPE)
    # - POST hook sees: attention output (BEFORE output projection)
    io_hooks.append(decoder_layer.self_attn.attn.register_forward_pre_hook(
        make_attention_io_hook(layer_idx, "pre")
    ))
    io_hooks.append(decoder_layer.self_attn.attn.register_forward_hook(
        make_attention_io_hook(layer_idx, "post")
    ))

prompt, expected = generate_needle_prompt(
    tokenizer=llm.tokenizer,
    target_length=INPUT_LEN,
    needle_position=0.5,
    needle_value="7492",
    verbose=True,
)
outputs = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=16), use_tqdm=False)

for hook in io_hooks:
    hook.remove()

print("test_align: PASSED")
