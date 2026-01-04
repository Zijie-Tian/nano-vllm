"""
Test alignment between nanovllm and custom torch Qwen3 implementation.
Compares attention layer outputs and QKV tensors to verify correctness.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

import torch
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from modeling_qwen3 import Qwen3ForCausalLM
from utils import generate_needle_prompt

# Config
MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
INPUT_LEN = 64
DTYPE = torch.float16

# Storage for captured tensors
nanovllm_outputs = {}
torch_outputs = {}
nanovllm_qkv = {}
nanovllm_proj_inputs = {}  # Input to qkv_proj
torch_proj_inputs = {}      # Input to q_proj


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
    """Capture input to projection layer (hidden_states after layernorm)."""
    def hook(module, inputs):
        # inputs[0] is hidden_states
        hidden = inputs[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        storage[layer_id] = hidden.detach().clone()
    return hook


def make_torch_hook(layer_id: int, storage: dict):
    def hook(module, inputs, output):
        storage[layer_id] = output[0].detach().clone()
    return hook


def max_diff(t1: torch.Tensor, t2: torch.Tensor) -> float:
    return (t1.float() - t2.float()).abs().max().item()


def compute_qkv_diffs(nano_qkv: dict, torch_qkv: dict, num_kv_groups: int):
    """Compute Q, K, V max diffs. Returns (q_diff, k_diff, v_diff)."""
    nano_q = nano_qkv["q"]
    torch_q = torch_qkv["q"].squeeze(0).transpose(0, 1)
    q_diff = max_diff(nano_q, torch_q)

    nano_k = nano_qkv["k"]
    torch_k = torch_qkv["k"].squeeze(0)[::num_kv_groups, :, :].transpose(0, 1)
    k_diff = max_diff(nano_k, torch_k)

    nano_v = nano_qkv["v"]
    torch_v = torch_qkv["v"].squeeze(0)[::num_kv_groups, :, :].transpose(0, 1)
    v_diff = max_diff(nano_v, torch_v)

    return q_diff, k_diff, v_diff


# ============================================================
# Load models
# ============================================================
print("Loading nanovllm model...")
llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    enable_cpu_offload=False,
    dtype="float16",
)

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
for layer_idx, layer in enumerate(llm.model_runner.model.model.layers):
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
nanovllm_result = llm.generate([input_ids[0].tolist()], SamplingParams(temperature=0.01, max_tokens=1), use_tqdm=False)

print("Running torch inference...")
with torch.no_grad():
    torch_logits, _, torch_qkv_outputs = torch_model(input_ids, output_qkv_layers=list(range(num_layers)))

# ============================================================
# Compare QKVO per layer (one line each)
# ============================================================
print("\n" + "=" * 82)
print(f"{'Layer':<8} {'I':>10} {'Q':>10} {'K':>10} {'V':>10} {'O':>10}")
print("=" * 82)

all_passed = True
atol = 0.1

for layer_idx in range(num_layers):
    # Input diff (to qkv_proj / q_proj)
    nano_in = nanovllm_proj_inputs[layer_idx]
    torch_in = torch_proj_inputs[layer_idx]
    if nano_in.shape != torch_in.shape and nano_in.numel() == torch_in.numel():
        torch_in = torch_in.view(nano_in.shape)
    i_diff = max_diff(nano_in, torch_in)

    # QKV diffs
    q_diff, k_diff, v_diff = compute_qkv_diffs(nanovllm_qkv[layer_idx], torch_qkv_outputs[layer_idx], num_kv_groups)

    # O diff
    nano_out = nanovllm_outputs[layer_idx]
    torch_out = torch_outputs[layer_idx]
    if nano_out.shape != torch_out.shape and nano_out.numel() == torch_out.numel():
        torch_out = torch_out.view(nano_out.shape)
    o_diff = max_diff(nano_out, torch_out)

    # Check pass/fail
    passed = all(d < atol for d in [i_diff, q_diff, k_diff, v_diff, o_diff])
    all_passed = all_passed and passed
    status = "" if passed else " *"

    print(f"Layer {layer_idx:2d}{status:<3} {i_diff:>10.6f} {q_diff:>10.6f} {k_diff:>10.6f} {v_diff:>10.6f} {o_diff:>10.6f}")

# ============================================================
# Cleanup and result
# ============================================================
for hook in nanovllm_hooks + torch_hooks:
    hook.remove()

print("=" * 82)
if all_passed:
    print("test_align: PASSED")
else:
    print("test_align: FAILED (* = max_diff >= 0.1)")
