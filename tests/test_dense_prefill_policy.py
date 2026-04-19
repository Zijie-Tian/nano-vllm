import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import torch

from nanovllm.config import SparsePolicyType
from nanovllm.kvcache.sparse import create_sparse_policy
from nanovllm.kvcache.sparse.full_policy import FullAttentionPolicy
from nanovllm.kvcache.sparse.postrope import PostRoPEPolicy
from nanovllm.kvcache.sparse.prerope import PreRoPEPolicy
from nanovllm.kvcache.sparse.policy import PolicyContext
from nanovllm.kvcache.sparse.triattention import TriAttentionPolicy


class _DummyStream:
    def wait_stream(self, _other) -> None:
        return None


class _TrackingStreamContext:
    def __init__(self):
        self.active = False

    @contextlib.contextmanager
    def stream(self, _stream):
        prev = self.active
        self.active = True
        try:
            yield
        finally:
            self.active = prev


class _FakeOffloadEngine:
    def __init__(self, historical_blocks, current_block, *, num_ring_slots: int, block_size: int):
        self._historical_blocks = historical_blocks
        self._current_block = current_block
        self.num_ring_slots = num_ring_slots
        self.block_size = block_size
        self.compute_stream = _DummyStream()
        self.loaded = {}
        self.load_history = []
        self.compute_done = []

    def load_to_slot_layer(self, slot, layer_id, cpu_block_id, chunk_idx, is_prefill=True):
        del layer_id, chunk_idx, is_prefill
        self.loaded[slot] = self._historical_blocks[cpu_block_id]
        self.load_history.append((slot, cpu_block_id))

    def wait_slot_layer(self, slot):
        del slot

    def get_kv_for_slot(self, slot):
        return self.loaded[slot]

    def record_slot_compute_done(self, slot):
        self.compute_done.append(slot)

    def get_prefill_buffer_slice(self, layer_id, num_tokens):
        del layer_id, num_tokens
        return self._current_block


class _HistoryMaterializeOffloadEngine:
    def __init__(self, block):
        self.block = block
        self.decode_load_slots = [0]
        self.compute_stream = object()
        self.k_cache_gpu = torch.zeros(1, 2, 1, 2)

    def load_to_slot_layer(self, slot, layer_id, cpu_block_id, chunk_idx, is_prefill=False):
        del slot, layer_id, cpu_block_id, chunk_idx, is_prefill

    def wait_slot_layer(self, slot):
        del slot

    def get_kv_for_slot(self, slot):
        del slot
        if not self.tracker.active:
            raise AssertionError("get_kv_for_slot must run inside compute_stream")
        return self.block

    def record_slot_compute_done(self, slot):
        del slot


def _fake_flash_attn_with_lse(q, k, v, softmax_scale, causal):
    del softmax_scale
    bias = 1.0 if causal else 0.0
    out = q + k.sum(dim=1, keepdim=True) + v.sum(dim=1, keepdim=True) + bias
    lse = torch.full((1,), bias, dtype=q.dtype)
    return out, lse


def _fake_merge_attention_outputs(o1, l1, o2, l2):
    return o1 + o2, l1 + l2


class DensePrefillPolicyTests(unittest.TestCase):
    def test_full_policy_chunked_prefill_pipeline(self):
        q = torch.tensor(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ]
        )
        historical_blocks = {
            0: (
                torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]]]]),
                torch.tensor([[[[0.0, 1.0]], [[0.0, 2.0]]]]),
            ),
            1: (
                torch.tensor([[[[3.0, 0.0]], [[4.0, 0.0]]]]),
                torch.tensor([[[[0.0, 3.0]], [[0.0, 4.0]]]]),
            ),
            2: (
                torch.tensor([[[[5.0, 0.0]], [[6.0, 0.0]]]]),
                torch.tensor([[[[0.0, 5.0]], [[0.0, 6.0]]]]),
            ),
        }
        current_block = (
            torch.tensor([[[[7.0, 0.0]], [[8.0, 0.0]]]]),
            torch.tensor([[[[0.0, 7.0]], [[0.0, 8.0]]]]),
        )
        offload_engine = _FakeOffloadEngine(
            historical_blocks,
            current_block,
            num_ring_slots=2,
            block_size=2,
        )

        fake_chunked_attention = types.SimpleNamespace(
            flash_attn_with_lse_flashinfer=_fake_flash_attn_with_lse,
            merge_attention_outputs_flashinfer=_fake_merge_attention_outputs,
        )

        policy = FullAttentionPolicy()
        with patch.dict(
            sys.modules,
            {"nanovllm.ops.chunked_attention": fake_chunked_attention},
        ):
            with patch("torch.cuda.stream", lambda stream: contextlib.nullcontext()):
                with patch("torch.cuda.default_stream", return_value=_DummyStream()):
                    actual = policy.compute_chunked_prefill(
                        q,
                        q,
                        q,
                        layer_id=0,
                        softmax_scale=1.0,
                        offload_engine=offload_engine,
                        kvcache_manager=None,
                        current_chunk_idx=3,
                        seq=None,
                        num_tokens=2,
                        selected_blocks=[0, 1, 2],
                    )

        q_batched = q.unsqueeze(0)
        expected_parts = [
            _fake_flash_attn_with_lse(q_batched, *historical_blocks[idx], 1.0, False)[0]
            for idx in [0, 1, 2]
        ]
        expected_parts.append(
            _fake_flash_attn_with_lse(q_batched, *current_block, 1.0, True)[0]
        )
        expected = sum(expected_parts).squeeze(0)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(offload_engine.load_history, [(0, 0), (1, 1), (0, 2)])
        self.assertEqual(offload_engine.compute_done, [0, 1, 0])

    def test_full_and_triattention_share_dense_stats_helpers(self):
        ctx = PolicyContext(
            query_chunk_idx=1,
            num_query_chunks=2,
            layer_id=0,
            query=torch.zeros(1, 1, 1),
            is_prefill=True,
            block_size=2,
            total_kv_len=4,
        )
        available_blocks = [11, 12]

        full_policy = FullAttentionPolicy()
        triattention_policy = TriAttentionPolicy()

        self.assertEqual(
            full_policy.select_blocks(
                available_blocks,
                offload_engine=None,
                ctx=ctx,
                q=torch.zeros(1),
                k=torch.zeros(1),
            ),
            available_blocks,
        )
        self.assertEqual(
            triattention_policy.select_blocks(
                available_blocks,
                offload_engine=None,
                ctx=ctx,
                q=torch.zeros(1),
                k=torch.zeros(1),
            ),
            available_blocks,
        )
        self.assertEqual(full_policy.get_density_stats()["num_chunks"], 1)
        self.assertEqual(triattention_policy.get_density_stats()["num_chunks"], 1)
        self.assertEqual(full_policy.get_density_stats()["overall_density"], 1.0)
        self.assertEqual(triattention_policy.get_density_stats()["overall_density"], 1.0)

    def test_prerope_and_postrope_factory_flags(self):
        ctx = PolicyContext(
            query_chunk_idx=0,
            num_query_chunks=1,
            layer_id=0,
            query=torch.zeros(1, 1, 1),
            is_prefill=True,
            block_size=2,
            total_kv_len=2,
        )
        available_blocks = [3, 4]

        prerope_policy = create_sparse_policy(SparsePolicyType.PREROPE)
        postrope_policy = create_sparse_policy(SparsePolicyType.POSTROPE)

        self.assertIsInstance(prerope_policy, PreRoPEPolicy)
        self.assertIsInstance(postrope_policy, PostRoPEPolicy)
        self.assertTrue(prerope_policy.apply_rope_in_attention)
        self.assertFalse(postrope_policy.apply_rope_in_attention)
        self.assertEqual(
            prerope_policy.select_blocks(
                available_blocks,
                offload_engine=None,
                ctx=ctx,
                q=torch.zeros(1),
                k=torch.zeros(1),
            ),
            available_blocks,
        )
        self.assertEqual(
            postrope_policy.select_blocks(
                available_blocks,
                offload_engine=None,
                ctx=ctx,
                q=torch.zeros(1),
                k=torch.zeros(1),
            ),
            available_blocks,
        )
        self.assertEqual(repr(prerope_policy), "PreRoPEPolicy()")
        self.assertEqual(repr(postrope_policy), "PostRoPEPolicy()")

    def test_prerope_history_materialization_reads_slots_on_compute_stream(self):
        tracker = _TrackingStreamContext()
        block = (
            torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),
            torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]]]]),
        )
        offload_engine = _HistoryMaterializeOffloadEngine(block)
        offload_engine.tracker = tracker
        policy = PreRoPEPolicy()

        with patch("torch.cuda.stream", tracker.stream):
            with patch("torch.cuda.default_stream", return_value=_DummyStream()):
                hist_k, hist_v = policy._materialize_prefilled_cpu_history(
                    cpu_block_table=[0],
                    offload_engine=offload_engine,
                    layer_id=0,
                    block_size=2,
                    last_block_valid_tokens=2,
                )

        self.assertTrue(
            torch.equal(hist_k, torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])),
        )
        self.assertTrue(
            torch.equal(hist_v, torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])),
        )


if __name__ == "__main__":
    unittest.main()
