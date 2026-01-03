"""
Test alignment between nanovllm and custom torch Qwen3 implementation.
Compares attention layer outputs to verify correctness.
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
INPUT_LEN = 512  # Use shorter length for alignment test
DTYPE = torch.float16

# Storage for captured tensors
nanovllm_outputs = {}
torch_outputs = {}


def make_nanovllm_hook(layer_id: int, storage: dict):
    """Capture nanovllm self_attn outputs (after o_proj)."""
    def hook(module, inputs, output):
        # Qwen3Attention output is a tuple (attn_output, None)
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        # nanovllm shape: [num_tokens, hidden_size] -> add batch dim
        if attn_output.dim() == 2:
            attn_output = attn_output.unsqueeze(0)
        storage[layer_id] = attn_output.detach().clone()
    return hook


def make_torch_hook(layer_id: int, storage: dict):
    """Capture torch model self_attn outputs (after o_proj)."""
    def hook(module, inputs, output):
        # Qwen3Attention output is (attn_output, past_kv, qkv_dict)
        attn_output, _, _ = output
        storage[layer_id] = attn_output.detach().clone()
    return hook


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-2):
    """Compare two tensors and print statistics."""
    # Handle shape differences
    if t1.shape != t2.shape:
        print(f"[{name}] Shape mismatch: {t1.shape} vs {t2.shape}")
        # Try to reshape for comparison if possible
        if t1.numel() == t2.numel():
            t2 = t2.view(t1.shape)
        else:
            return False

    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"

    print(f"[{name}] {status}")
    print(f"  Shape: {list(t1.shape)}")
    print(f"  t1 mean: {t1.float().mean():.6f}, std: {t1.float().std():.6f}")
    print(f"  t2 mean: {t2.float().mean():.6f}, std: {t2.float().std():.6f}")
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    return passed


# ============================================================
# Load nanovllm model
# ============================================================
print("=" * 60)
print("Loading nanovllm model...")
print("=" * 60)

llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    enable_cpu_offload=False,  # Disable offload for alignment test
    dtype="float16",
)

# ============================================================
# Load torch model
# ============================================================
print("\n" + "=" * 60)
print("Loading custom torch model...")
print("=" * 60)

torch_model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
torch_model = torch_model.to("cuda")
torch_model.eval()

# ============================================================
# Generate test input
# ============================================================
print("\n" + "=" * 60)
print("Generating test input...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt, _ = generate_needle_prompt(
    tokenizer=tokenizer,
    target_length=INPUT_LEN,
    verbose=True,
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
print(f"Input shape: {input_ids.shape}")

# ============================================================
# Register hooks on both models
# ============================================================
print("\n" + "=" * 60)
print("Registering hooks...")
print("=" * 60)

# Hook on nanovllm (self_attn is Qwen3Attention, captures output after o_proj)
nanovllm_hooks = []
for layer_idx, layer in enumerate(llm.model_runner.model.model.layers):
    if layer_idx >= 2:  # Only first 2 layers
        break
    nanovllm_hooks.append(
        layer.self_attn.register_forward_hook(
            make_nanovllm_hook(layer_idx, nanovllm_outputs)
        )
    )
    print(f"  Registered nanovllm hook on layer {layer_idx} self_attn")

# Hook on torch model (self_attn is Qwen3Attention, captures output after o_proj)
torch_hooks = []
for layer_idx, layer in enumerate(torch_model.model.layers):
    if layer_idx >= 2:  # Only first 2 layers
        break
    torch_hooks.append(
        layer.self_attn.register_forward_hook(
            make_torch_hook(layer_idx, torch_outputs)
        )
    )
    print(f"  Registered torch hook on layer {layer_idx} self_attn")

# ============================================================
# Run nanovllm inference
# ============================================================
print("\n" + "=" * 60)
print("Running nanovllm inference...")
print("=" * 60)

# Use prompt_token_ids to ensure same input
prompt_token_ids = input_ids[0].tolist()
nanovllm_result = llm.generate(
    [prompt_token_ids],
    SamplingParams(temperature=0.01, max_tokens=1),  # Near-greedy for determinism
    use_tqdm=False,
)

# ============================================================
# Run torch inference
# ============================================================
print("\n" + "=" * 60)
print("Running torch inference...")
print("=" * 60)

with torch.no_grad():
    torch_logits, _, _ = torch_model(input_ids)

# ============================================================
# Compare outputs
# ============================================================
print("\n" + "=" * 60)
print("Comparing attention outputs...")
print("=" * 60)

all_passed = True
for layer_idx in sorted(nanovllm_outputs.keys()):
    if layer_idx not in torch_outputs:
        print(f"[Layer {layer_idx}] Missing torch output")
        all_passed = False
        continue

    nano_out = nanovllm_outputs[layer_idx]
    torch_out = torch_outputs[layer_idx]

    print(f"\n--- Layer {layer_idx} ---")
    passed = compare_tensors(f"Layer {layer_idx} attn_output", nano_out, torch_out, atol=0.1)
    all_passed = all_passed and passed

# ============================================================
# Cleanup
# ============================================================
for hook in nanovllm_hooks:
    hook.remove()
for hook in torch_hooks:
    hook.remove()

# ============================================================
# Result
# ============================================================
print("\n" + "=" * 60)
if all_passed:
    print("test_align: PASSED - nanovllm and torch outputs aligned!")
else:
    print("test_align: FAILED - outputs differ!")
print("=" * 60)
