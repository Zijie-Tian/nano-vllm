import os
from dataclasses import dataclass
from enum import Enum, auto
from transformers import AutoConfig
import torch


class SparsePolicyType(Enum):
    """Sparse attention policy types."""

    # fmt: off
    FULL = auto()       # No sparse attention (load all blocks)
    TRIATTENTION = auto()  # Full attention semantics, but apply RoPE inside Attention
    QUEST = auto()      # Query-aware Top-K block selection (decode only)
    XATTN_BSA = auto()  # XAttention Block Sparse Attention (prefill only, chunked)
    COMPASS = auto()    # COMPASS sparse attention (prefill only, chunked)
    BLASST = auto()     # BLASST sparse attention (prefill only, chunked)
    # fmt: on


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
    eos: int | list[int] = -1  # Single EOS token or list of EOS tokens (e.g., GLM-4)
    kvcache_block_size: int = 1024
    num_kvcache_blocks: int = -1
    dtype: str | None = None  # "float16", "bfloat16", or None (use model default)

    # CPU Offload configuration
    enable_cpu_offload: bool = False
    offload_policy: str = "lru"  # "lru", "fifo", or full class path

    num_gpu_blocks: int = (
        -1
    )  # User-specified GPU blocks count, -1 = auto (use max available)

    # Computed fields for offload (set in __post_init__ or by ModelRunner)
    num_gpu_kvcache_blocks: int = -1
    num_cpu_kvcache_blocks: int = -1

    # Sparse attention configuration
    # FULL: no sparse attention (load all blocks)
    # QUEST: decode-only sparse attention with Top-K block selection
    # XATTN_BSA: prefill-only block sparse attention with chunk-level selection
    sparse_policy: SparsePolicyType = SparsePolicyType.FULL
    sparse_topk_blocks: int = 8  # Top-K blocks for Quest
    sparse_threshold_blocks: int = 4  # Apply sparse only when blocks > threshold

    # XAttention BSA specific parameters
    sparse_block_size: int = 128  # Block size for BSA (tokens per block)
    sparse_samples_per_chunk: int = 128  # Samples per chunk for estimation
    sparse_threshold: float = 0.95  # Cumulative attention threshold (tau in XAttention)
    sparse_use_triton: bool = True  # Use Triton kernels for estimation
    sparse_stride: int = 8  # Stride for Q/K downsampling
    sparse_chunk_size: int = 16384  # Triton kernel chunk size for estimation

    # COMPASS specific parameters
    lambda_threshold: float = 0.001  # Threshold for COMPASS block selection
    compass_top_p: float = 0.9  # Top-p threshold for CPU sub-block selection
    compass_theta: float = 0.6  # Self-cosine threshold (SpargeAttention): force-select blocks with diverse tokens

    # BLASST specific parameters
    blasst_a: int = 16384  # Inverse formula numerator (λ = a / L). Default 16384 gives λ=0.5 at 32K.
    blasst_fixed_lambda: float | None = (
        None  # Fixed threshold instead of inverse formula (overrides a)
    )
    blasst_granularity: int = 128  # Token granularity for skip decisions (default 128)

    # TriAttention sparse decode parameters
    triattention_stats_path: str | None = None
    triattention_kv_budget: int = 2048
    triattention_window_size: int = 128
    triattention_score_aggregation: str = "mean"
    triattention_sparse_normalize_scores: bool = True
    triattention_offset_max_length: int = 65536
    triattention_score_chunk_max_tokens: int = 4096
    triattention_protect_prefill: bool = False
    triattention_include_prefill_in_budget: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.triattention_kv_budget > 0
        assert self.triattention_window_size >= 0
        assert self.triattention_offset_max_length >= 1
        assert self.triattention_score_chunk_max_tokens >= 1
        if self.triattention_score_aggregation not in {"mean", "max"}:
            raise ValueError(
                "triattention_score_aggregation must be 'mean' or 'max'"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        # Get max position embeddings (GLM-4 uses seq_length instead of max_position_embeddings)
        max_pos = getattr(
            self.hf_config,
            "max_position_embeddings",
            getattr(self.hf_config, "seq_length", 4096),
        )
        self.max_model_len = min(self.max_model_len, max_pos)
        assert self.max_num_batched_tokens >= self.max_model_len

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
                raise ValueError(
                    f"Invalid dtype: {self.dtype}. Choose from: {list(dtype_map.keys())}"
                )
            self.hf_config.torch_dtype = dtype_map[self.dtype]
