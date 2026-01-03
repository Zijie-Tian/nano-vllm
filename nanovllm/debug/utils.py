"""Utility functions for breakpoint alignment debugging."""

import torch
from nanovllm.utils.context import set_context, reset_context


def setup_prefill_context(seq_len: int, device: torch.device):
    """
    Set up nanovllm context for prefill alignment testing.

    Args:
        seq_len: Sequence length
        device: Target device
    """
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    slot_mapping = torch.full((seq_len,), -1, dtype=torch.int32, device=device)

    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=slot_mapping,
        is_chunked_prefill=False,
    )


def setup_decode_context(context_len: int, device: torch.device):
    """
    Set up nanovllm context for decode alignment testing.

    Args:
        context_len: Context length (number of previous tokens)
        device: Target device
    """
    context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)
    slot_mapping = torch.tensor([-1], dtype=torch.int32, device=device)
    block_tables = torch.zeros((1, 1), dtype=torch.int32, device=device)

    set_context(
        is_prefill=False,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def cleanup_context():
    """Reset nanovllm context after alignment testing."""
    reset_context()
