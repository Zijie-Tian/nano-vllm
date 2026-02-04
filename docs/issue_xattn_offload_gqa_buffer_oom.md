# Issue: XAttention Offload Mode GQA Buffer OOM

## 问题描述

在使用 XAttention BSA (Block Sparse Attention) + CPU Offload 模式运行 GLM-4-9B 等大模型时，出现 CUDA OOM 错误。

### 错误信息

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB.
GPU 0 has a total capacity of 23.57 GiB of which 4.19 GiB is free.
```

### 复现环境

| 项目 | 值 |
|------|-----|
| 模型 | GLM-4-9B-Chat-1M |
| GPU | RTX 3090 (24GB) |
| Context Length | 32K |
| sparse_policy | XATTN_BSA |
| enable_cpu_offload | true |
| max_model_len | 1048576 (1M) |

### 错误位置

```
File "nanovllm/kvcache/sparse/xattn_bsa.py", line 246, in alloc_policy_metadata
    self._k_expanded = torch.empty(shape, dtype=dtype, device=device)
```

---

## 问题分析

### 内存分配分析

`alloc_policy_metadata()` 在 KV cache 初始化时分配以下 buffer：

| Buffer | 用途 | 大小 (GLM-4, 1M seq) |
|--------|------|----------------------|
| `_prefill_mask_buffer` | BSA mask | ~32 MB |
| `_m_partial_buffer` | KV chunking m stats | ~32 MB |
| `_l_partial_buffer` | KV chunking l stats | ~32 MB |
| `_block_sums_buffer` | Block sums | ~64 MB |
| **`_k_expanded`** | GQA K 扩展 | **~8 GB** |
| **`_v_expanded`** | GQA V 扩展 | **~8 GB** |

### GQA Buffer 计算

```python
shape = (1, num_heads, max_seq_len, head_dim)
     = (1, 32, 1048576, 128)

size = 1 × 32 × 1048576 × 128 × 2 bytes (fp16)
     = 8,589,934,592 bytes
     = 8 GB per buffer
```

### 根本原因

1. **设计意图冲突**：`_k_expanded` 和 `_v_expanded` 的文档注释明确说是 "for GPU-only mode"
2. **条件检查不完整**：代码只检查了 `num_heads == num_kv_heads` 来跳过分配，没有检查 offload 模式
3. **Offload 模式不需要这些 buffer**：`compute_chunked_prefill()` 使用不同的计算路径，不依赖预分配的 GQA buffer

### 相关代码

```python
# xattn_bsa.py:238-247
# Only allocate GQA expansion buffers if GQA (num_heads != num_kv_heads)
if num_heads == num_kv_heads:
    logger.info(f"[XAttn] No GQA expansion needed (num_heads == num_kv_heads = {num_heads})")
    return  # <-- 只检查了 GQA，没检查 offload 模式

# Shape: [1, num_heads, max_seq_len, head_dim] for xattn_estimate format
shape = (1, num_heads, max_seq_len, head_dim)
self._k_expanded = torch.empty(shape, dtype=dtype, device=device)  # <-- OOM here
self._v_expanded = torch.empty(shape, dtype=dtype, device=device)
```

---

## 解决思路

### 方案 1: 在 Offload 模式下跳过 GQA Buffer 分配 (推荐)

在 `alloc_policy_metadata()` 中添加 offload 模式检查：

```python
def alloc_policy_metadata(
    self,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    enable_cpu_offload: bool = False,  # <-- 新增参数
) -> None:
    # ... 分配 mask buffer 和 KV chunking buffers (offload 模式需要)

    # Skip GQA buffers in offload mode
    # Chunked prefill uses compute_chunked_prefill() which doesn't need these
    if enable_cpu_offload:
        logger.info("[XAttn] Offload mode: skipping GQA expansion buffers")
        return

    # GPU-only mode: pre-allocate GQA buffers for compute_prefill()
    if num_heads == num_kv_heads:
        logger.info(f"[XAttn] No GQA expansion needed")
        return

    shape = (1, num_heads, max_seq_len, head_dim)
    self._k_expanded = torch.empty(shape, dtype=dtype, device=device)
    self._v_expanded = torch.empty(shape, dtype=dtype, device=device)
```

**需要修改的文件**：
1. `nanovllm/kvcache/sparse/xattn_bsa.py` - `alloc_policy_metadata()` 方法
2. `nanovllm/engine/model_runner.py` - 调用 `alloc_policy_metadata()` 时传入 `enable_cpu_offload`

### 方案 2: 延迟分配 (Lazy Allocation)

只在 `compute_prefill()` 首次调用时分配 GQA buffer，offload 模式走 `compute_chunked_prefill()` 不会触发分配。

```python
def compute_prefill(self, ...):
    # Lazy allocation on first use
    if self._k_expanded is None and num_heads != num_kv_heads:
        self._allocate_gqa_buffers(...)
    ...
```

### 方案 3: 基于 chunk_size 限制 buffer 大小

不预分配 max_seq_len 大小，而是只分配 chunk_size 大小：

```python
# 原来: max_seq_len (1M tokens) -> 8 GB
# 修改后: chunk_size (16K tokens) -> ~130 MB
buffer_len = self.chunk_size if enable_cpu_offload else max_seq_len
shape = (1, num_heads, buffer_len, head_dim)
```

---

## 验证方法

修复后运行以下命令验证：

```bash
cd /home/zijie/Code/COMPASS
GPULIST=0 ./scripts/run_ruler.sh glm4-9b-xattn-nanovllm synthetic xattn --task niah_single_1
```

预期结果：
- 不再出现 8GB allocation 的 OOM 错误
- 模型正常加载并完成推理

---

## 相关文档

- `docs/xattn_bsa_policy_design.md` - XAttention BSA Policy 设计文档
- `docs/gpu_only_xattn_guide.md` - GPU-Only XAttention 指南

## 优先级

**High** - 阻塞 9B+ 模型在 24GB 显存 GPU 上使用 XAttention + Offload 模式

---

## 修复状态

**✅ 已修复** (2026-02-05)

### 修复内容

采用方案 1，在 offload 模式下跳过 GQA buffer 分配：

1. `nanovllm/kvcache/sparse/policy.py`: 基类添加 `enable_cpu_offload` 参数
2. `nanovllm/kvcache/sparse/xattn_bsa.py`: 实现 offload 模式检查，跳过 GQA buffer
3. `nanovllm/engine/model_runner.py`: 传入 `enable_cpu_offload` 参数

### 验证结果

```bash
# 64K offload 测试
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

- ✅ 日志显示: `[XAttn] Offload mode: skipping GQA expansion buffers`
- ✅ 测试通过: 100% 准确率
- ✅ 内存节省: ~16 GB (for 1M max_seq_len)

### 内存对比

| 配置 | 修复前 | 修复后 |
|------|--------|--------|
| max_model_len=72K | +1.1 GB | 0 GB |
| max_model_len=1M | +16 GB | 0 GB |
