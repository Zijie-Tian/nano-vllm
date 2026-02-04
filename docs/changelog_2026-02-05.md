# Changelog 2026-02-05

## Bug Fixes

### XAttention Offload GQA Buffer OOM Fix

**Issue**: `docs/issue_xattn_offload_gqa_buffer_oom.md`

**Problem**: 在 XAttention BSA + CPU Offload 模式下，`alloc_policy_metadata()` 分配了只有 GPU-only 模式才需要的 GQA expansion buffers (`_k_expanded`, `_v_expanded`)，导致 24GB GPU (RTX 3090) 上 OOM。

**Root Cause**:
- GQA buffer 大小: `2 × num_heads × max_seq_len × head_dim × dtype_size`
- 对于 1M max_seq_len: 2 × 32 × 1048576 × 128 × 2 = **16 GB**
- Offload 模式的 `compute_chunked_prefill()` 不需要这些 buffer

**Fix** (commit `11a867f`):
1. `nanovllm/kvcache/sparse/policy.py`: 基类添加 `enable_cpu_offload` 参数
2. `nanovllm/kvcache/sparse/xattn_bsa.py`: offload 模式跳过 GQA buffer 分配
3. `nanovllm/engine/model_runner.py`: 传入 `enable_cpu_offload` 参数

**Memory Savings**:
| max_model_len | 修复前 | 修复后 |
|---------------|--------|--------|
| 72K | +1.1 GB | 0 GB |
| 1M | +16 GB | 0 GB |

**Verification**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_64k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 72000 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```
- 日志显示: `[XAttn] Offload mode: skipping GQA expansion buffers`
- 测试结果: 100% 准确率

---

## Code Cleanup

### Tests Directory Cleanup

**Commits**: `a709551`, `2b61c5a`, `d35dd76`

删除了 16 个冗余/过时的测试文件，保留核心测试：

**保留的文件** (4 个):
| 文件 | 用途 |
|------|------|
| `test_ruler.py` | 核心 RULER benchmark (13 tasks, 100 samples) |
| `test_xattn_estimate_alignment.py` | XAttn kernel 一致性验证 |
| `utils.py` | 共享工具函数 |
| `__init__.py` | 包标记 |

**删除的文件** (16 个, -4306 行):

| 类别 | 文件 | 删除原因 |
|------|------|----------|
| XAttn 测试 | `test_xattn_bsa.py` | 功能被 test_ruler 覆盖 |
| | `test_xattn_chunked.py` | 与 estimate_chunked 重复 |
| | `test_xattn_estimate_chunked.py` | chunked prefill 验证 |
| | `test_xattn_kernels.py` | Triton kernel 单元测试 |
| | `test_xattn_kv_chunking_batch.py` | batch 验证 |
| Needle 测试 | `test_needle.py` | 被 test_ruler NIAH 任务覆盖 |
| | `test_needle_ref.py` | HF 参考实现 |
| CUDA Graph | `test_chunk_attention_graph.py` | 被 graph_reuse 取代 |
| | `test_chunk_attention_graph_reuse.py` | 实验性功能 |
| | `test_cudagraph_memory.py` | 内存分析工具 |
| 其他 | `test_gpuonly_density_alignment.py` | GPU-only 密度测试 |
| | `test_hierarchical_estimate.py` | 分层估计测试 |
| | `test_quest_policy.py` | Quest 策略测试 |
| | `test_sequential.py` | 状态隔离测试 |
| | `bench_estimate_block_size.py` | 性能 benchmark |
| | `modeling_qwen3.py` | Qwen3 参考模型 |

**Note**: 所有删除的文件可从 git 历史恢复：
```bash
git checkout <commit-hash>^ -- tests/<filename>
```

---

## Summary

| 类型 | 数量 | 影响 |
|------|------|------|
| Bug Fix | 1 | 节省 16GB 显存 (1M seq) |
| 文件删除 | 16 | -4306 行代码 |
| 新增文档 | 1 | 本文件 |
