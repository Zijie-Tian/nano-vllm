"""
Hybrid sparse attention policy.

Allows using different policies for prefill vs decode phases.
This is useful because optimal sparsity patterns often differ:
- Prefill: fixed patterns work well (e.g., VerticalSlash)
- Decode: query-aware selection helps (e.g., Quest)
"""

from typing import List
import torch
from .policy import SparsePolicy, PolicyContext


class HybridPolicy(SparsePolicy):
    """
    Hybrid policy that uses different policies for prefill and decode.

    Example usage:
    ```python
    from nanovllm.kvcache.sparse import (
        HybridPolicy, VerticalSlashPolicy, QuestPolicy,
        VerticalSlashConfig, QuestConfig, BlockMetadataManager
    )

    # Prefill: use fast fixed pattern
    prefill_policy = VerticalSlashPolicy(VerticalSlashConfig(
        num_sink_blocks=1,
        local_window_blocks=3,
    ))

    # Decode: use query-aware selection
    metadata = BlockMetadataManager(num_blocks, num_layers, num_heads, head_dim)
    decode_policy = QuestPolicy(QuestConfig(topk_blocks=8), metadata)

    # Combine
    policy = HybridPolicy(prefill_policy, decode_policy)
    ```
    """

    def __init__(
        self,
        prefill_policy: SparsePolicy,
        decode_policy: SparsePolicy,
    ):
        """
        Initialize hybrid policy.

        Args:
            prefill_policy: Policy to use during prefill phase
            decode_policy: Policy to use during decode phase
        """
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy

    def select_blocks(
        self,
        available_blocks: List[int],
        ctx: PolicyContext,
    ) -> List[int]:
        """Delegate to appropriate policy based on phase."""
        if ctx.is_prefill:
            return self.prefill_policy.select_blocks(available_blocks, ctx)
        else:
            return self.decode_policy.select_blocks(available_blocks, ctx)

    def on_block_offloaded(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """Forward to both policies (both may need metadata updates)."""
        self.prefill_policy.on_block_offloaded(
            cpu_block_id, layer_id, k_cache, num_valid_tokens
        )
        self.decode_policy.on_block_offloaded(
            cpu_block_id, layer_id, k_cache, num_valid_tokens
        )

    def reset(self) -> None:
        """Reset both policies."""
        self.prefill_policy.reset()
        self.decode_policy.reset()

    def __repr__(self) -> str:
        return (
            f"HybridPolicy(\n"
            f"  prefill={self.prefill_policy},\n"
            f"  decode={self.decode_policy}\n"
            f")"
        )
