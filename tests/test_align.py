"""
Test alignment between nanovllm and custom torch Qwen3 implementation.
Compares attention layer outputs and QKV tensors to verify correctness.

Usage:
    python test_align.py                    # Without CPU offload
    python test_align.py --enable-offload   # With CPU offload
    python test_align.py --input-len 4096   # Custom input length
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

import argparse
import torch
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from modeling_qwen3 import Qwen3ForCausalLM
from utils import generate_needle_prompt

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--enable-offload", action="store_true", help="Enable CPU offload")
parser.add_argument("--input-len", type=int, default=1024 * 12, help="Input sequence length")
parser.add_argument("--model-path", type=str, default="~/models/Qwen3-0.6B/", help="Model path")
parser.add_argument("--num-gpu-blocks", type=int, default=6, help="Number of GPU blocks (ring buffer slots)")
parser.add_argument("--block-size", type=int, default=1024, help="KV cache block size")
args = parser.parse_args()

# Config
MODEL_PATH = os.path.expanduser(args.model_path)
INPUT_LEN = args.input_len
ENABLE_OFFLOAD = args.enable_offload
NUM_GPU_BLOCKS = args.num_gpu_blocks
BLOCK_SIZE = args.block_size
DTYPE = torch.float16

print(f"Config: input_len={INPUT_LEN}, enable_offload={ENABLE_OFFLOAD}, num_gpu_blocks={NUM_GPU_BLOCKS}, block_size={BLOCK_SIZE}")

# Storage for captured tensors
nanovllm_outputs = {}
torch_outputs = {}
nanovllm_qkv = {}
nanovllm_proj_inputs = {}
torch_proj_inputs = {}


# ============================================================
# Hook functions for non-offload mode (overwrite)
# ============================================================
def make_nanovllm_hook(layer_id: int, storage: dict):
    def hook(module, inputs, output):
        attn_output = output[0] if isinstance(output, tuple) else output
        if attn_output.dim() == 2:
            attn_output = attn_output.unsqueeze(0)
        storage[layer_id] = attn_output.detach().clone()
    return hook


def make_nanovllm_qkv_hook(layer_id: int, storage: dict):
    def hook(module, inputs):
        q, k, v = inputs[0], inputs[1], inputs[2]
        storage[layer_id] = {
            "q": q.detach().clone(),
            "k": k.detach().clone(),
            "v": v.detach().clone(),
        }
    return hook


def make_proj_input_hook(layer_id: int, storage: dict):
    def hook(module, inputs):
        hidden = inputs[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        storage[layer_id] = hidden.detach().clone()
    return hook


# ============================================================
# Hook functions for offload mode (accumulate Q and I, overwrite O)
# ============================================================
def make_accumulating_q_hook(layer_id: int, storage: dict):
    """Accumulate Q from each chunk for offload mode."""
    def hook(module, inputs):
        q = inputs[0].detach().clone()
        if layer_id not in storage:
            storage[layer_id] = []
        storage[layer_id].append(q)
    return hook


def make_accumulating_input_hook(layer_id: int, storage: dict):
    """Accumulate input hidden states from each chunk for offload mode."""
    def hook(module, inputs):
        hidden = inputs[0].detach().clone()
        if layer_id not in storage:
            storage[layer_id] = []
        storage[layer_id].append(hidden)
    return hook


def make_overwrite_output_hook(layer_id: int, storage: dict):
    """Overwrite output (keep only last chunk) for offload mode."""
    def hook(module, inputs, output):
        attn_output = output[0] if isinstance(output, tuple) else output
        if attn_output.dim() == 2:
            attn_output = attn_output.unsqueeze(0)
        storage[layer_id] = attn_output.detach().clone()
    return hook


# ============================================================
# CPU KV cache access for offload mode
# ============================================================
def get_nanovllm_kv_from_cpu(llm, seq, num_layers):
    """Get complete K, V cache from CPU side after all chunks finish."""
    offload_engine = llm.model_runner.kvcache_manager.offload_engine
    kvcache_manager = llm.model_runner.kvcache_manager

    # CRITICAL: Synchronize all CUDA operations before reading CPU memory
    # The D2H copy runs on transfer_stream_main and may still be in progress
    torch.cuda.synchronize()

    cpu_block_ids = kvcache_manager.get_cpu_block_table(seq)

    kv_per_layer = {}
    for layer_id in range(num_layers):
        k_blocks = []
        v_blocks = []
        for cpu_block_id in cpu_block_ids:
            k_block, v_block = offload_engine.get_cpu_block(layer_id, cpu_block_id)
            k_blocks.append(k_block)
            v_blocks.append(v_block)

        # Concatenate all blocks: [total_tokens, kv_heads, head_dim]
        k_full = torch.cat(k_blocks, dim=0)[:seq.num_tokens]
        v_full = torch.cat(v_blocks, dim=0)[:seq.num_tokens]
        kv_per_layer[layer_id] = {"k": k_full, "v": v_full}

    return kv_per_layer


def make_torch_hook(layer_id: int, storage: dict):
    def hook(module, inputs, output):
        storage[layer_id] = output[0].detach().clone()
    return hook


def cosine_sim(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """Cosine similarity between flattened tensors (1.0 = identical)."""
    return torch.nn.functional.cosine_similarity(
        t1.flatten().float(), t2.flatten().float(), dim=0
    ).item()


def compute_qkv_sims(nano_qkv: dict, torch_qkv: dict, num_kv_groups: int):
    """Compute Q, K, V cosine similarities. Returns (q_sim, k_sim, v_sim)."""
    nano_q = nano_qkv["q"]
    torch_q = torch_qkv["q"].squeeze(0).transpose(0, 1)

    nano_k = nano_qkv["k"]
    torch_k = torch_qkv["k"].squeeze(0)[::num_kv_groups, :, :].transpose(0, 1)

    nano_v = nano_qkv["v"]
    torch_v = torch_qkv["v"].squeeze(0)[::num_kv_groups, :, :].transpose(0, 1)

    return cosine_sim(nano_q, torch_q), cosine_sim(nano_k, torch_k), cosine_sim(nano_v, torch_v)


# ============================================================
# Load models
# ============================================================
print("Loading nanovllm model...")
llm_kwargs = dict(
    enforce_eager=True,
    max_model_len=32768,
    gpu_memory_utilization=0.2,
    max_num_batched_tokens=32768,
    enable_cpu_offload=ENABLE_OFFLOAD,
    dtype="float16",
    kvcache_block_size=BLOCK_SIZE,
)
if ENABLE_OFFLOAD:
    llm_kwargs["num_gpu_blocks"] = NUM_GPU_BLOCKS
llm = LLM(MODEL_PATH, **llm_kwargs)

num_heads = llm.model_runner.model.model.layers[0].self_attn.num_heads
num_kv_heads = llm.model_runner.model.model.layers[0].self_attn.num_kv_heads
num_kv_groups = num_heads // num_kv_heads
num_layers = len(llm.model_runner.model.model.layers)

print("Loading torch model...")
torch_model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
torch_model = torch_model.to("cuda")
torch_model.eval()

# ============================================================
# Generate test input
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt, _ = generate_needle_prompt(tokenizer=tokenizer, target_length=INPUT_LEN, verbose=True)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
print(f"Input shape: {input_ids.shape}")

# ============================================================
# Register hooks
# ============================================================
nanovllm_hooks = []
nanovllm_q_accum = {}  # For offload mode: accumulated Q from all chunks
nanovllm_i_accum = {}  # For offload mode: accumulated I from all chunks

for layer_idx, layer in enumerate(llm.model_runner.model.model.layers):
    if ENABLE_OFFLOAD:
        # Offload mode: accumulate Q and I, overwrite O
        nanovllm_hooks.append(layer.self_attn.register_forward_hook(make_overwrite_output_hook(layer_idx, nanovllm_outputs)))
        nanovllm_hooks.append(layer.self_attn.attn.register_forward_pre_hook(make_accumulating_q_hook(layer_idx, nanovllm_q_accum)))
        nanovllm_hooks.append(layer.self_attn.qkv_proj.register_forward_pre_hook(make_accumulating_input_hook(layer_idx, nanovllm_i_accum)))
    else:
        # Non-offload mode: overwrite all
        nanovllm_hooks.append(layer.self_attn.register_forward_hook(make_nanovllm_hook(layer_idx, nanovllm_outputs)))
        nanovllm_hooks.append(layer.self_attn.attn.register_forward_pre_hook(make_nanovllm_qkv_hook(layer_idx, nanovllm_qkv)))
        nanovllm_hooks.append(layer.self_attn.qkv_proj.register_forward_pre_hook(make_proj_input_hook(layer_idx, nanovllm_proj_inputs)))

torch_hooks = []
for layer_idx, layer in enumerate(torch_model.model.layers):
    torch_hooks.append(layer.self_attn.register_forward_hook(make_torch_hook(layer_idx, torch_outputs)))
    torch_hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_proj_input_hook(layer_idx, torch_proj_inputs)))

# ============================================================
# Run inference
# ============================================================
print("Running nanovllm inference...")

if ENABLE_OFFLOAD:
    # Manual execution to capture KV cache before deallocation
    # Use max_tokens=2 so sequence doesn't finish immediately after prefill
    llm.add_request(input_ids[0].tolist(), SamplingParams(temperature=0.01, max_tokens=2))

    # Run prefill step (this calls run_chunked_offload_prefill internally)
    output, num_tokens = llm.step()
    print(f"[Offload] Prefill done: {num_tokens} tokens")

    # Now seq is in running queue, KV cache is in CPU
    seq = llm.scheduler.running[0]
    print(f"[Offload] Sequence: {seq}")

    # Get KV cache from CPU BEFORE decode step deallocates it
    nanovllm_kv_cpu = get_nanovllm_kv_from_cpu(llm, seq, num_layers)
    print(f"[Offload] Retrieved KV cache from CPU for {seq.num_tokens} tokens")

    # IMPORTANT: Save outputs NOW before decode step overwrites them
    # nanovllm_outputs contains prefill attention outputs at this point
    nanovllm_outputs_prefill = {k: v.clone() for k, v in nanovllm_outputs.items()}

    # Complete remaining steps (decode)
    while not llm.is_finished():
        llm.step()

    # Use prefill outputs for comparison
    nanovllm_outputs = nanovllm_outputs_prefill
else:
    nanovllm_result = llm.generate([input_ids[0].tolist()], SamplingParams(temperature=0.01, max_tokens=1), use_tqdm=False)

print("Running torch inference...")
with torch.no_grad():
    torch_logits, _, torch_qkv_outputs = torch_model(input_ids, output_qkv_layers=list(range(num_layers)))

# ============================================================
# Compare using cosine similarity (1.0 = perfect alignment)
# ============================================================
print("\n" + "=" * 70)
print(f"{'Layer':<8} {'I':>10} {'Q':>10} {'K':>10} {'V':>10} {'O':>10}")
print("=" * 70)

all_passed = True
threshold = 0.999  # Cosine similarity threshold

for layer_idx in range(num_layers):
    if ENABLE_OFFLOAD:
        # ============================================================
        # Offload mode: use accumulated Q/I and CPU-side K/V
        # Only compare prompt tokens (INPUT_LEN), exclude generated tokens
        # ============================================================
        # I: concatenate accumulated chunks, trim to prompt length
        i_chunks = nanovllm_i_accum[layer_idx]
        nano_in = torch.cat(i_chunks, dim=0)[:INPUT_LEN]
        if nano_in.dim() == 2:
            nano_in = nano_in.unsqueeze(0)
        torch_in = torch_proj_inputs[layer_idx]
        if nano_in.shape != torch_in.shape and nano_in.numel() == torch_in.numel():
            torch_in = torch_in.view(nano_in.shape)
        i_sim = cosine_sim(nano_in, torch_in)

        # Q: concatenate accumulated chunks, trim to prompt length
        q_chunks = nanovllm_q_accum[layer_idx]
        nano_q = torch.cat(q_chunks, dim=0)[:INPUT_LEN]
        torch_q = torch_qkv_outputs[layer_idx]["q"].squeeze(0).transpose(0, 1)
        q_sim = cosine_sim(nano_q, torch_q)

        # K, V: from CPU cache, trim to prompt length and move to GPU
        nano_k = nanovllm_kv_cpu[layer_idx]["k"][:INPUT_LEN].cuda()
        torch_k = torch_qkv_outputs[layer_idx]["k"].squeeze(0)[::num_kv_groups, :, :].transpose(0, 1)
        k_sim = cosine_sim(nano_k, torch_k)

        nano_v = nanovllm_kv_cpu[layer_idx]["v"][:INPUT_LEN].cuda()
        torch_v = torch_qkv_outputs[layer_idx]["v"].squeeze(0)[::num_kv_groups, :, :].transpose(0, 1)
        v_sim = cosine_sim(nano_v, torch_v)

        # O: compare attention outputs directly
        # For single-chunk case (input_len <= block_size), shapes should match
        # For multi-chunk case, nano_out is the last chunk only
        nano_out = nanovllm_outputs[layer_idx]
        torch_out = torch_outputs[layer_idx]

        if nano_out.numel() == torch_out.numel():
            # Single chunk or shapes match - compare directly
            o_sim = cosine_sim(nano_out, torch_out)
        else:
            # Multi-chunk case: compare last chunk with corresponding torch slice
            last_chunk_len = nano_out.shape[1] if nano_out.dim() == 3 else nano_out.shape[0]
            torch_out_slice = torch_out[:, -last_chunk_len:, :] if torch_out.dim() == 3 else torch_out[-last_chunk_len:, :]
            o_sim = cosine_sim(nano_out, torch_out_slice)
    else:
        # ============================================================
        # Non-offload mode: original logic
        # ============================================================
        # Input similarity
        nano_in = nanovllm_proj_inputs[layer_idx]
        torch_in = torch_proj_inputs[layer_idx]

        if nano_in.shape != torch_in.shape and nano_in.numel() == torch_in.numel():
            torch_in = torch_in.view(nano_in.shape)

        i_sim = cosine_sim(nano_in, torch_in)

        # QKV similarities
        q_sim, k_sim, v_sim = compute_qkv_sims(nanovllm_qkv[layer_idx], torch_qkv_outputs[layer_idx], num_kv_groups)

        # O similarity
        nano_out = nanovllm_outputs[layer_idx]
        torch_out = torch_outputs[layer_idx]
        if nano_out.shape != torch_out.shape and nano_out.numel() == torch_out.numel():
            torch_out = torch_out.view(nano_out.shape)
        o_sim = cosine_sim(nano_out, torch_out)

    # Check pass/fail
    passed = all(s >= threshold for s in [i_sim, q_sim, k_sim, v_sim, o_sim])
    all_passed = all_passed and passed
    status = "" if passed else " *"

    print(f"Layer {layer_idx:2d}{status:<3} {i_sim:>10.6f} {q_sim:>10.6f} {k_sim:>10.6f} {v_sim:>10.6f} {o_sim:>10.6f}")

# ============================================================
# Cleanup and result
# ============================================================
for hook in nanovllm_hooks + torch_hooks:
    hook.remove()

print("=" * 70)
mode_str = " [offload]" if ENABLE_OFFLOAD else ""
if all_passed:
    print(f"test_align{mode_str}: PASSED (cosine_sim >= 0.999)")
else:
    print(f"test_align{mode_str}: FAILED (* = cosine_sim < 0.999)")
