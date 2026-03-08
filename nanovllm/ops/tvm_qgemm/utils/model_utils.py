"""Model utilities for kernel shape extraction and quantization configuration."""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

import os
import logging
import json
import configparser
import numpy as np

logger = logging.getLogger("model_utils")

# Preset kernel configurations for various models
_PRESET_KERNELS = {
    "llama-2-7b-4bit": [
        # bits, M, K, N, m_groups
        # [4, 12288, 4096, 1, -1],  # unused for llama.cpp
        [4, 4096, 4096, 1, -1],
        [4, 11008, 4096, 1, -1],
        [4, 4096, 11008, 1, -1],
    ],
    "llama-2-7b-2bit": [
        # [2, 12288, 4096, 1, -1],  # unused for llama.cpp
        [2, 4096, 4096, 1, -1],
        [2, 11008, 4096, 1, -1],
        [2, 4096, 11008, 1, -1],
    ],
    "llama-2-13b-2bit": [
        [2, 5120, 5120, 1, -1],
        [2, 13824, 5120, 1, -1],
        [2, 5120, 13824, 1, -1],
    ],
    "llama-2-13b-4bit": [
        [4, 5120, 5120, 1, -1],
        [4, 13824, 5120, 1, -1],
        [4, 5120, 13824, 1, -1],
    ],
    "llama-3-8b-2bit": [
        [2, 4096, 4096, 1, -1],
        [2, 14336, 4096, 1, -1],
        [2, 4096, 14336, 1, -1],
        [2, 1024, 4096, 1, -1],
    ],
    "llama-3-8b-4bit": [
        [4, 4096, 4096, 1, -1],
        [4, 14336, 4096, 1, -1],
        [4, 4096, 14336, 1, -1],
        [4, 1024, 4096, 1, -1],
    ],
    "hf-bitnet-3b": [
        [2, 3200, 8640, 1, 1],
        [2, 8640, 3200, 1, 1],
        [2, 3200, 3200, 1, 1],
    ],
    "hf-bitnet-large-intn": [    # 700M
        [2, 1536, 4096, 1, 1],
        [2, 4096, 1536, 1, 1],
        [2, 1536, 1536, 1, 1],
    ],
    "hf-bitnet-large-tq": [    # 700M
        [2, 1536, 4096, 1, -1],
        [2, 4096, 1536, 1, -1],
        [2, 1536, 1536, 1, -1],
    ],
    "ms-bitnet-3b": [
        [2, 3200, 800, 1, 1],
        [2, 3200, 3200, 1, 1],
        [2, 3200, 10240, 1, 1],
        [2, 10240, 3200, 1, 1],
        [2, 800, 3200, 1, 1],
    ],
    "phi-3-mini-2bit": [
        [2, 3072, 3072, 1, -1],
        [2, 9216, 3072, 1, -1],
        [2, 3072, 8192, 1, -1],
        [2, 16384, 3072, 1, -1],
    ],
    "phi-3-mini-4bit": [
        [4, 3072, 3072, 1, -1],
        [4, 9216, 3072, 1, -1],
        [4, 3072, 8192, 1, -1],
        [4, 16384, 3072, 1, -1],
    ],
    "trilm-3.9b": [
        [2, 3072, 3072, 1, -1],
        [2, 3072, 9216, 1, -1],
        [2, 9216, 3072, 1, -1],
        [2, 768, 3072, 1, -1],
    ],
    "kvcache-2bit": [
        # K cache decode: M=seq_len, K=head_dim(128), N=1, m_groups=-1
        [2, 128, 128, 1, -1],   # seq_len=128 (shared with V cache due to same shape)
        [2, 512, 128, 1, -1],   # seq_len=512
        [2, 1024, 128, 1, -1],  # seq_len=1024
        [2, 4096, 128, 1, -1],  # seq_len=4096
        [2, 8192, 128, 1, -1],  # seq_len=8192
        # V cache decode: M=head_dim(128), K=seq_len, N=1, m_groups=-1
        # Note: [2, 128, 128, 1, -1] is same as K cache seq_len=128, so skipped
        [2, 128, 512, 1, -1],   # seq_len=512
        [2, 128, 1024, 1, -1],  # seq_len=1024
        [2, 128, 4096, 1, -1],  # seq_len=4096
        [2, 128, 8192, 1, -1],  # seq_len=8192
    ],
    "kvcache-4bit": [
        # K cache decode: M=seq_len, K=head_dim(128), N=1, m_groups=-1
        [4, 128, 128, 1, -1],   # seq_len=128 (shared with V cache due to same shape)
        [4, 512, 128, 1, -1],   # seq_len=512
        [4, 1024, 128, 1, -1],  # seq_len=1024
        [4, 4096, 128, 1, -1],  # seq_len=4096
        [4, 8192, 128, 1, -1],  # seq_len=8192
        # V cache decode: M=head_dim(128), K=seq_len, N=1, m_groups=-1
        # Note: [4, 128, 128, 1, -1] is same as K cache seq_len=128, so skipped
        [4, 128, 512, 1, -1],   # seq_len=512
        [4, 128, 1024, 1, -1],  # seq_len=1024
        [4, 128, 4096, 1, -1],  # seq_len=4096
        [4, 128, 8192, 1, -1],  # seq_len=8192
    ],
    "test": [
        # Add customized kernels here for testing
        [4, 256, 768, 1, -1],
        [4, 512, 1024, 1, -1],
    ],
    "gptq-auto": [],  # Will be auto-detected
}


def get_preset_models() -> List[str]:
    """Get list of preset model names."""
    return list(_PRESET_KERNELS.keys())


def extract_kernel_shapes(model_name: str, model_dir: Optional[str] = None) -> List[Tuple[int, int, int, int, int]]:
    """
    Extract kernel shapes for a given model.
    
    Args:
        model_name: Name of the preset model or 'gptq-auto'
        model_dir: Directory containing model files (for auto-detection)
    
    Returns:
        List of (bits, M, K, N, m_groups) tuples
    """
    if model_name in _PRESET_KERNELS:
        if model_name == "gptq-auto" and model_dir:
            # Auto-detect from model files
            return auto_detect_kernel_shapes(model_dir)
        return _PRESET_KERNELS[model_name]
    else:
        logger.error(f"Unknown model: {model_name}")
        return []


def get_quantization_config(model_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get quantization configuration from model directory.
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        Dictionary with quantization configuration or None
    """
    model_path = Path(model_dir)
    
    # Look for quantization config files
    config_files = ["quantize_config.json", "config.json", "quant_config.json"]
    
    for config_file in config_files:
        config_path = model_path / config_file
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    
                    # GPTQ format
                    if "bits" in config:
                        return {
                            "quant_method": "gptq",
                            "bits": config.get("bits", 4),
                            "group_size": config.get("group_size", 128),
                            "sym": config.get("sym", True),
                            "desc_act": config.get("desc_act", False),
                        }
                    
                    # GPTQ v2 format
                    if "quantization_config" in config:
                        quant_config = config["quantization_config"]
                        if quant_config.get("quant_method") == "gptq":
                            return {
                                "quant_method": "gptq_v2",
                                "bits": quant_config.get("bits", 4),
                                "group_size": quant_config.get("group_size", 128),
                                "sym": quant_config.get("sym", True),
                                "desc_act": quant_config.get("desc_act", False),
                            }
                    
                    # BitNet format
                    if config.get("quantization_config", {}).get("quant_method") == "bitnet":
                        return {
                            "quant_method": "bitnet",
                            "bits": 2,
                            "group_size": -1,  # Unified scale
                            "sym": True,
                        }
                    
            except Exception as e:
                logger.warning(f"Failed to parse config file {config_path}: {e}")
    
    return None


def auto_detect_kernel_shapes(model_dir: str) -> List[Tuple[int, int, int, int, int]]:
    """
    Auto-detect kernel shapes from model weights.
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        List of (bits, M, K, N, m_groups) tuples
    """
    model_path = Path(model_dir)
    kernel_shapes = []
    
    # Get quantization config
    quant_config = get_quantization_config(model_dir)
    if not quant_config:
        logger.error(f"Cannot detect quantization config from {model_dir}")
        return []
    
    bits = quant_config.get("bits", 4)
    
    # Try to load model config to get dimensions
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                model_config = json.load(f)
                
                # Extract dimensions based on model architecture
                arch = model_config.get("model_type", "").lower()
                
                if "llama" in arch:
                    hidden_size = model_config.get("hidden_size", 4096)
                    intermediate_size = model_config.get("intermediate_size", 11008)
                    
                    # Standard Llama shapes
                    kernel_shapes = [
                        (bits, hidden_size, hidden_size, 1, -1),
                        (bits, intermediate_size, hidden_size, 1, -1),
                        (bits, hidden_size, intermediate_size, 1, -1),
                    ]
                    
                    # Add head dimension if available
                    if "num_attention_heads" in model_config:
                        num_heads = model_config["num_attention_heads"]
                        head_dim = hidden_size // num_heads
                        if head_dim * num_heads != hidden_size:
                            kernel_shapes.append((bits, head_dim * num_heads, hidden_size, 1, -1))
                
                elif "phi" in arch:
                    hidden_size = model_config.get("hidden_size", 3072)
                    intermediate_size = model_config.get("intermediate_size", 8192)
                    
                    kernel_shapes = [
                        (bits, hidden_size, hidden_size, 1, -1),
                        (bits, intermediate_size, hidden_size, 1, -1),
                        (bits, hidden_size, intermediate_size, 1, -1),
                    ]
                
                elif "bitnet" in arch:
                    hidden_size = model_config.get("hidden_size", 3200)
                    intermediate_size = model_config.get("intermediate_size", 8640)
                    
                    # BitNet uses unified scale (m_groups = 1)
                    kernel_shapes = [
                        (bits, hidden_size, hidden_size, 1, 1),
                        (bits, intermediate_size, hidden_size, 1, 1),
                        (bits, hidden_size, intermediate_size, 1, 1),
                    ]
                
                else:
                    logger.warning(f"Unknown architecture: {arch}")
                    
        except Exception as e:
            logger.error(f"Failed to parse model config: {e}")
    
    # Try to detect from weight files if config parsing failed
    if not kernel_shapes:
        kernel_shapes = detect_from_weight_files(model_path, bits)
    
    return kernel_shapes


def detect_from_weight_files(model_path: Path, bits: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Detect kernel shapes from weight file names and sizes.
    
    Args:
        model_path: Path to model directory
        bits: Quantization bits
    
    Returns:
        List of (bits, M, K, N, m_groups) tuples
    """
    kernel_shapes = set()
    
    # Look for safetensors or pytorch bin files
    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    
    if not weight_files:
        logger.warning(f"No weight files found in {model_path}")
        return []
    
    # This is a simplified detection - in practice would need to load and inspect weights
    # For now, return empty list
    logger.info("Weight file inspection not fully implemented - please specify model preset")
    
    return list(kernel_shapes)


def preprocess_weights(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 4,
    g: int = 4,
    bm: int = 512,
    kfactor: int = 16,
    simd_n_in: int = 16,
    simd_n_out: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offline preprocess the weights before inference.
    
    Parameters
    ----------
    w : np.ndarray
        Quantized weights of shape (M, K) and type "uint8".
        Add a bias of 2^(bits-1) to the original int1/2/3/4 values to convert it to uint values.
        E.g., add a bias of 2 to int2: -2, -1, 0, 1 -> 0, 1, 2, 3
    scales: np.ndarray
        Quantization scales of shape (M, K // group_size) or (m_groups,) and type float32/16.
    zeros: np.ndarray
        Same shape and type with scales.
        If None, the actual zero points will be 2^(bits-1) * scales;
        if not None, the actual zero points will be zeros + 2^(bits-1) * scales.
        E.g., before passing the zeros from BitDistiller/GPTQ, you need to modify it as following:
        `zeros = (zeros - (2 ** (bits - 1))) * scales`
    bits: int
        Number of bits for each quantized element
    g: int
        Group size of LUT
    bm: int
        Tuned tiling size of M
    kfactor: int
        Tuned tiling size of K
    simd_n_in: int
        Number of SIMD lanes for input (16 for ARM NEON/AVX2 with uint8)
    simd_n_out: int
        Number of SIMD lanes for output (8 for ARM NEON with float16, AVX2 with float32/int32)
    
    Returns
    -------
    w: np.ndarray
        Permuted weights with shape (M // bm, K // g, bm // ngroups_per_elem)
    scales: np.ndarray
        Permuted scales with adjusted shape for TVM API
    """
    assert(w.dtype == "uint8")
    
    M, K = w.shape
    M = M * bits  # Total number of weight values after bit expansion
    ngroups_per_elem = 8 // g  # Number of groups that can be packed into one uint8
    
    # Step 1 - Extract individual bits from quantized weights
    # Convert each weight element into its bit representation
    # Example: if bits=4 and w[i,j]=11 (binary: 1011), extract [1,1,0,1]
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    
    # Step 2 - Reorganize bits for group processing
    # Transpose and reshape to group consecutive elements together
    w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
    
    # Step 3 - Pack groups into single elements for LUT indexing
    # Packing multiple bits allows using them as direct LUT indices
    # Example: if g=4, pack 4 bits [b0,b1,b2,b3] into value b0 + 2*b1 + 4*b2 + 8*b3
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    
    # Step 4 - Reshape for SIMD processing
    # Reorganize data layout for efficient SIMD instruction execution
    w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
    mgroup = ngroups_per_elem * simd_n_in
    w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
    
    # Step 5 - Final tiling for optimized memory access
    # Create hierarchical tiling structure for cache optimization
    w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
    w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
    w = w.reshape(M // bm, K // g // kfactor, bm // mgroup, kfactor, simd_n_in)
    
    # Final reshape to match TVM API requirements
    w = w.reshape(M // bm, K // g, bm // ngroups_per_elem)
    
    # Process scales for group-wise quantization
    if scales.size >= M // bits:
        group_size = K // scales.shape[1]
        scales = scales.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
        scales = scales.reshape(M // bm, K // group_size, bm // bits // simd_n_out, simd_n_out)
        if zeros is not None:
            zeros = zeros.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
            zeros = zeros.reshape(M // bm, K // group_size, bm // bits // simd_n_out, simd_n_out)
            scales = np.stack([scales, zeros], axis=-2)
        # Input size of current TVM API
        scales = scales.reshape(M // bm, K // group_size, -1)
    else:
        # Per-tensor quantization
        if zeros is not None:
            scales = np.concatenate([scales, zeros])
    
    return w, scales


def preprocess_kvcache(
    kv_cache: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    is_key: bool = True,
    bits: int = 4,
    g: int = 4,
    bm: int = 512,
    kfactor: int = 16,
    simd_n_in: int = 16,
    simd_n_out: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess KV cache for LUT-based attention.

    ============================================================================
                              Overview
    ============================================================================

    This function packs quantized KV cache into a format optimized for LUT-based
    matrix multiplication. Each (batch, head) is processed independently.

    ============================================================================
                         K vs V Dimension Mapping
    ============================================================================

    Why K and V have different layouts?

    In attention: O = softmax(Q @ K^T) @ V

    For LUT-based GeMV, we need:
      - "Activation" builds LUT (the matrix that multiplies)
      - "Weight" is quantized and packed (the matrix being multiplied)

    For Q @ K^T:
      - Q builds LUT, K is the "weight"
      - K shape: [seq_len, head_dim], reduction along head_dim
      - So K is packed as: M=seq_len, K=head_dim

    For P @ V (where P = softmax(Q @ K^T)):
      - P builds LUT, V is the "weight"
      - P shape: [seq_q, seq_kv], V shape: [seq_kv, head_dim]
      - Reduction along seq_kv, but we want to keep head_dim as output
      - So we transpose V and pack as: M=head_dim, K=seq_kv

    ============================================================================
                              K Cache Pack
    ============================================================================

    Input:  kv_cache [batch, n_head, chunk_size, head_dim]
            scales   [batch, n_head, chunk_size, 1]  (per-token)

    Per-head data flow:
    +---------------------+                  +---------------------+
    | K_head              |                  | K_packed            |
    | [chunk_size,        |  preprocess_     | [M*bits//bm,        |
    |  head_dim]          | ------------->   |  K//g,              |
    |                     |   weights        |  bm//ngroups]       |
    | M=chunk_size        |                  |                     |
    | K=head_dim          |                  |                     |
    +---------------------+                  +---------------------+

    Constraints:
      - bm constrains chunk_size:  chunk_size * bits % bm == 0
      - kfactor constrains head_dim: head_dim % (g * kfactor) == 0

    Example (chunk_size=64, head_dim=128, bits=4, g=4, bm=256, kfactor=16):
      - M = chunk_size = 64
      - K = head_dim = 128
      - bm constraint: 64 * 4 = 256, 256 % 256 = 0  OK
      - kfactor constraint: 128 % (4 * 16) = 128 % 64 = 0  OK
      - Output: [256//256, 128//4, 256//2] = [1, 32, 128]

    ============================================================================
                              V Cache Pack
    ============================================================================

    Input:  kv_cache [batch, n_head, chunk_size, head_dim]
            scales   [batch, n_head, n_groups, head_dim]  (per-channel)

    Per-head data flow (note the transpose!):
    +---------------------+     transpose    +---------------------+
    | V_head              |                  | V_head_T            |
    | [chunk_size,        | ------------->   | [head_dim,          |
    |  head_dim]          |                  |  chunk_size]        |
    +---------------------+                  +---------------------+
                                                      |
                                                      | preprocess_weights
                                                      v
                                             +---------------------+
                                             | V_packed            |
                                             | [M*bits//bm,        |
                                             |  K//g,              |
                                             |  bm//ngroups]       |
                                             |                     |
                                             | M=head_dim          |
                                             | K=chunk_size        |
                                             +---------------------+

    Constraints:
      - bm constrains head_dim:    head_dim * bits % bm == 0
      - kfactor constrains chunk_size: chunk_size % (g * kfactor) == 0

    Example (chunk_size=64, head_dim=128, bits=4, g=4, bm=512, kfactor=16):
      - M = head_dim = 128 (after transpose)
      - K = chunk_size = 64 (after transpose)
      - bm constraint: 128 * 4 = 512, 512 % 512 = 0  OK
      - kfactor constraint: 64 % (4 * 16) = 64 % 64 = 0  OK
      - Output: [512//512, 64//4, 512//2] = [1, 16, 256]

    ============================================================================
                           Constraint Summary
    ============================================================================

    +-------+------------------+------------------+---------------------------+
    | Cache | bm constrains    | kfactor constrains | Typical values          |
    +-------+------------------+------------------+---------------------------+
    |   K   | chunk_size       | head_dim         | bm=256, kfactor=16       |
    |   V   | head_dim         | chunk_size       | bm=512, kfactor=16       |
    +-------+------------------+------------------+---------------------------+

    IMPORTANT: kfactor must satisfy BOTH:
      - head_dim % (g * kfactor) == 0   (from K cache)
      - chunk_size % (g * kfactor) == 0 (from V cache)

    With g=4, kfactor=16: both head_dim and chunk_size must be multiples of 64.

    ============================================================================
                              Parameters
    ============================================================================

    Parameters
    ----------
    kv_cache : np.ndarray
        Quantized KV cache [batch, n_head, chunk_size, head_dim], dtype uint8.
        Values should have bias 2^(bits-1) added (e.g., int4 [-8,7] -> uint4 [0,15]).
    scales : np.ndarray
        Quantization scales:
        - K cache: [batch, n_head, chunk_size, 1] (per-token)
        - V cache: [batch, n_head, n_groups, head_dim] (per-channel within chunk)
    zeros : np.ndarray, optional
        Zero points with same shape as scales (for asymmetric quantization).
    is_key : bool
        True for K cache, False for V cache.
    bits : int
        Quantization bits (2, 3, 4, etc.).
    g : int
        LUT group size. Each group of g elements shares one LUT entry.
    bm : int
        Tile size for M dimension. Must satisfy M * bits % bm == 0.
    kfactor : int
        Tile size for K dimension. Must satisfy K % (g * kfactor) == 0.
    simd_n_in : int
        SIMD input lanes (16 for NEON/AVX2 with uint8).
    simd_n_out : int
        SIMD output lanes (8 for NEON fp16 / AVX2 fp32).

    Returns
    -------
    packed_kv : np.ndarray
        Packed KV cache [batch, n_head, M*bits//bm, K//g, bm//(8//g)].
    packed_scales : np.ndarray
        Packed scales [batch, n_head, ...].
    """
    assert kv_cache.dtype == np.uint8, f"Expected uint8, got {kv_cache.dtype}"

    batch, n_head, chunk_size, head_dim = kv_cache.shape

    # -------------------------------------------------------------------------
    # Validate scales shape and determine M, K dimensions
    # -------------------------------------------------------------------------
    assert scales.shape[0] == batch and scales.shape[1] == n_head

    if is_key:
        # K cache: scales [batch, n_head, chunk_size, 1] (per-token)
        assert scales.shape[2] == chunk_size and scales.shape[3] == 1, \
            f"K cache scales shape mismatch: expected [*, *, {chunk_size}, 1], got {scales.shape}"
        M, K = chunk_size, head_dim
    else:
        # V cache: scales [batch, n_head, n_groups, head_dim] (per-channel)
        assert scales.shape[3] == head_dim, \
            f"V cache scales shape mismatch: expected [*, *, *, {head_dim}], got {scales.shape}"
        M, K = head_dim, chunk_size  # Note: transposed!

    # -------------------------------------------------------------------------
    # Validate constraints
    # -------------------------------------------------------------------------
    M_expanded = M * bits
    if M_expanded % bm != 0:
        raise ValueError(
            f"{'K' if is_key else 'V'} cache: M*bits={M_expanded} must be divisible by bm={bm}.\n"
            f"  M={'chunk_size' if is_key else 'head_dim'}={M}, bits={bits}"
        )

    if K % (g * kfactor) != 0:
        raise ValueError(
            f"{'K' if is_key else 'V'} cache: K={K} must be divisible by g*kfactor={g*kfactor}.\n"
            f"  K={'head_dim' if is_key else 'chunk_size'}={K}, g={g}, kfactor={kfactor}"
        )

    # -------------------------------------------------------------------------
    # Pack each (batch, head) independently
    # -------------------------------------------------------------------------
    packed_list = []
    packed_scales_list = []

    for b in range(batch):
        batch_packed = []
        batch_scales = []

        for h in range(n_head):
            # Extract per-head data: [chunk_size, head_dim]
            kv_head = kv_cache[b, h, :, :]

            if is_key:
                # ----- K cache: no transpose -----
                # kv_head: [chunk_size, head_dim] -> M=chunk_size, K=head_dim
                # scales: [chunk_size] -> [chunk_size, 1] for preprocess_weights
                scales_head = scales[b, h, :, 0][:, np.newaxis]  # [M, 1]
                zeros_head = zeros[b, h, :, 0][:, np.newaxis] if zeros is not None else None
            else:
                # ----- V cache: transpose! -----
                # kv_head: [chunk_size, head_dim] -> transpose -> [head_dim, chunk_size]
                # This makes M=head_dim, K=chunk_size
                kv_head = kv_head.T

                # scales: [n_groups, head_dim] -> transpose -> [head_dim, n_groups]
                # This matches preprocess_weights expectation: [M, K//group_size]
                scales_head = scales[b, h, :, :].T  # [head_dim, n_groups]
                zeros_head = zeros[b, h, :, :].T if zeros is not None else None

            # Call preprocess_weights
            try:
                packed_w, packed_s = preprocess_weights(
                    kv_head, scales_head, zeros_head,
                    bits=bits, g=g, bm=bm, kfactor=kfactor,
                    simd_n_in=simd_n_in, simd_n_out=simd_n_out
                )
                batch_packed.append(packed_w)
                batch_scales.append(packed_s)

            except Exception as e:
                cache_type = 'K' if is_key else 'V'
                raise RuntimeError(
                    f"Failed to pack {cache_type} cache (batch={b}, head={h}):\n"
                    f"  kv_head.shape={kv_head.shape} (M={M}, K={K})\n"
                    f"  scales_head.shape={scales_head.shape}\n"
                    f"  bm={bm}, kfactor={kfactor}, g={g}, bits={bits}\n"
                    f"  Error: {e}"
                ) from e

        packed_list.append(np.stack(batch_packed, axis=0))
        packed_scales_list.append(np.stack(batch_scales, axis=0))

    packed_kv = np.stack(packed_list, axis=0)  # [batch, n_head, ...]
    packed_scales = np.stack(packed_scales_list, axis=0)

    return packed_kv, packed_scales