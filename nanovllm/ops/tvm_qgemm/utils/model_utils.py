"""Model utilities for kernel shape extraction and quantization configuration."""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

import logging
import json
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
    "hf-bitnet-large-intn": [  # 700M
        [2, 1536, 4096, 1, 1],
        [2, 4096, 1536, 1, 1],
        [2, 1536, 1536, 1, 1],
    ],
    "hf-bitnet-large-tq": [  # 700M
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
        [2, 128, 128, 1, -1],  # seq_len=128 (shared with V cache due to same shape)
        [2, 512, 128, 1, -1],  # seq_len=512
        [2, 1024, 128, 1, -1],  # seq_len=1024
        [2, 4096, 128, 1, -1],  # seq_len=4096
        [2, 8192, 128, 1, -1],  # seq_len=8192
        # V cache decode: M=head_dim(128), K=seq_len, N=1, m_groups=-1
        # Note: [2, 128, 128, 1, -1] is same as K cache seq_len=128, so skipped
        [2, 128, 512, 1, -1],  # seq_len=512
        [2, 128, 1024, 1, -1],  # seq_len=1024
        [2, 128, 4096, 1, -1],  # seq_len=4096
        [2, 128, 8192, 1, -1],  # seq_len=8192
    ],
    "kvcache-4bit": [
        # K cache decode: M=seq_len, K=head_dim(128), N=1, m_groups=-1
        [4, 128, 128, 1, -1],  # seq_len=128 (shared with V cache due to same shape)
        [4, 512, 128, 1, -1],  # seq_len=512
        [4, 1024, 128, 1, -1],  # seq_len=1024
        [4, 4096, 128, 1, -1],  # seq_len=4096
        [4, 8192, 128, 1, -1],  # seq_len=8192
        # V cache decode: M=head_dim(128), K=seq_len, N=1, m_groups=-1
        # Note: [4, 128, 128, 1, -1] is same as K cache seq_len=128, so skipped
        [4, 128, 512, 1, -1],  # seq_len=512
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


def extract_kernel_shapes(
    model_name: str, model_dir: Optional[str] = None
) -> List[Tuple[int, int, int, int, int]]:
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
                    if (
                        config.get("quantization_config", {}).get("quant_method")
                        == "bitnet"
                    ):
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
                            kernel_shapes.append(
                                (bits, head_dim * num_heads, hidden_size, 1, -1)
                            )

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


def detect_from_weight_files(
    model_path: Path, bits: int
) -> List[Tuple[int, int, int, int, int]]:
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
    weight_files = list(model_path.glob("*.safetensors")) + list(
        model_path.glob("*.bin")
    )

    if not weight_files:
        logger.warning(f"No weight files found in {model_path}")
        return []

    # This is a simplified detection - in practice would need to load and inspect weights
    # For now, return empty list
    logger.info(
        "Weight file inspection not fully implemented - please specify model preset"
    )

    return list(kernel_shapes)

