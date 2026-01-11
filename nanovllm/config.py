import os
from dataclasses import dataclass
from enum import Enum, auto
from transformers import AutoConfig
import torch


class SparsePolicyType(Enum):
    """Sparse attention policy types."""
    FULL = auto()   # No sparse attention (load all blocks)
    QUEST = auto()  # Query-aware Top-K block selection (decode only)
    MINFERENCE = auto()  # MInference vertical + slash sparse prefill (GPU-only)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 1024
    num_kvcache_blocks: int = -1
    dtype: str | None = None  # "float16", "bfloat16", or None (use model default)

    # CPU Offload configuration
    enable_cpu_offload: bool = False
    offload_policy: str = "lru"  # "lru", "fifo", or full class path
    num_transfer_streams: int = 4  # Number of CUDA streams for async transfers
    num_gpu_blocks: int = -1  # User-specified GPU blocks count, -1 = auto (use max available)
    num_kv_buffers: int = 4  # Ring buffer size for layer-wise offload (decode H2D pipeline)

    # Computed fields for offload (set in __post_init__ or by ModelRunner)
    num_gpu_kvcache_blocks: int = -1
    num_cpu_kvcache_blocks: int = -1

    # Sparse attention configuration
    # Quest: decode-only sparse attention with Top-K block selection
    # FULL: no sparse attention (load all blocks)
    # MINFERENCE: MInference vertical + slash sparse prefill (GPU-only)
    sparse_policy: SparsePolicyType = SparsePolicyType.FULL
    sparse_topk_blocks: int = 8  # Top-K blocks for Quest
    sparse_threshold_blocks: int = 4  # Apply sparse only when blocks > threshold

    # MInference configuration (used when sparse_policy == MINFERENCE)
    minference_adaptive_budget: float = 0.3  # Budget as fraction of seq_len (None to use fixed sizes)
    minference_vertical_size: int = 1000  # Fixed vertical size (if adaptive_budget is None)
    minference_slash_size: int = 6096  # Fixed slash size (if adaptive_budget is None)
    minference_num_sink_tokens: int = 30  # Sink tokens to always keep
    minference_num_recent_diags: int = 100  # Recent diagonals to always keep

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        # CPU offload mode only supports single sequence (layer-wise processing)
        if self.enable_cpu_offload and self.max_num_seqs != 1:
            import logging
            logging.warning(
                f"CPU offload mode only supports single sequence. "
                f"Overriding max_num_seqs from {self.max_num_seqs} to 1."
            )
            self.max_num_seqs = 1

        # Override torch_dtype if user specified
        if self.dtype is not None:
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if self.dtype not in dtype_map:
                raise ValueError(f"Invalid dtype: {self.dtype}. Choose from: {list(dtype_map.keys())}")
            self.hf_config.torch_dtype = dtype_map[self.dtype]
