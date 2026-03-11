# BLASST Bug Fix Report: m_global/LSE 解耦 & 逐元素因果遮罩

**日期**: 2026-03-12  
**修改文件**:
- [`nanovllm/ops/blasst_chunked_prefill.py`](../nanovllm/ops/blasst_chunked_prefill.py) — Triton 内核
- [`nanovllm/kvcache/sparse/blasst.py`](../nanovllm/kvcache/sparse/blasst.py) — 调用方

---

## 1. 背景

BLASST (Blocked Attention Sparsity via Softmax Thresholding) 通过在 FlashAttention 的 Online Softmax 过程中零额外开销地融合稀疏判定，实现动态注意力剪枝。其核心跳过条件为：

$$\text{skip if } \tilde{m}_i^{(j)} - m_i^{(j-1)} < \ln(\lambda)$$

其中 $\tilde{m}_i^{(j)}$ 是当前 sub-block 的局部最大 QK 分数，$m_i^{(j-1)}$ 是历史所有 sub-block 的全局运行最大值。

在 RULER `niah_multikey_2` 测试中，BLASST 产生了 0/5 的准确率（FullAttention 基线为 4/5），暴露了两个底层 Bug。

---

## 2. Bug A: `m_global` 从 LSE 加载（而非纯 Running Max）

### 问题描述

在原始实现中，Triton 内核的 `m_global` 输入是从上一个 KV chunk 的 **LSE**（Log-Sum-Exp）加载的：

```python
# 原始代码 (blasst_chunked_prefill.py)
m_global = tl.load(lse_in_ptrs, ...)  # lse_in = m + ln(l)
```

但 LSE 的定义为 $\text{LSE} = m + \ln(l)$，其中 $l = \sum_j \exp(s_j - m)$ 是归一化因子。由于 $l \geq 1$（至少有一个 token 参与 softmax），$\ln(l) \geq 0$，因此 LSE **始终大于等于**纯最大值 $m$。

### 影响

`m_global` 被 $\ln(l)$ 膨胀（实测平均约 3.2 nats），导致跳过条件 $\tilde{m}_i^{(j)} - m_i^{(j-1)} < \ln(\lambda)$ 更容易满足，造成过度激进的剪枝。

### 修复

在 Triton 内核中新增独立的 `MglobalIn` / `MglobalOut` 张量，与 LSE 完全解耦：

```diff
 # Triton 内核参数
-    LseIn,
+    MglobalIn,
+    MglobalOut,

 # 加载 m_global（纯 running max，非 LSE）
-    if HAS_LSE_IN:
-        m_global = tl.load(lse_in_ptrs, ...)
+    if HAS_MGLOBAL_IN:
+        m_global = tl.load(mgin_ptrs, ...)

 # 内核末尾写出 m_global
+    tl.store(mgout_ptrs, m_global, ...)
```

调用方 `blasst.py` 维护独立的 `historical_m_global`：

```diff
+    historical_m_global = None

     out, lse, m_global_out = blasst_chunked_prefill(
-        ..., lse_in=historical_lse, ...
+        ..., m_global_in=historical_m_global, ...
     )

+    if historical_m_global is None:
+        historical_m_global = m_global_out
+    else:
+        historical_m_global = torch.maximum(historical_m_global, m_global_out)
```

---

## 3. Bug B: 内核缺少逐元素因果遮罩

### 问题描述

BLASST 内核在处理当前 prefill chunk（需要因果注意力）时，仅通过外部传入的 **block 级别** `mask_buffer` 实现因果遮罩——将整个 BLOCK_N (64 tokens) 的 sub-block 标记为 0 或 1。但在因果边界处的 sub-block 内部，部分 token 应该被遮蔽（future tokens），部分应该可见，block 级别遮罩无法区分。

### 影响

因果边界处的 sub-block 中，future token 的 QK 分数未被设为 $-\inf$，直接参与了 softmax 和 value 加权，产生严重错误。与 FlashAttention 的因果输出比较：

| 模式 | max_diff |
|------|----------|
| 非因果 | 0.000244 ✅ |
| 因果（修复前） | **3.923828** ❌ |
| 因果（修复后） | 0.000488 ✅ |

### 修复

在 Triton 内核中新增 `IS_CAUSAL` 和 `KV_OFFSET` 参数：

```python
# 1. Block 级别提前退出（优化，非必须）
if IS_CAUSAL:
    kv_block_start = KV_OFFSET + start_n
    q_block_max = KV_OFFSET + pid_m * BLOCK_M + BLOCK_M - 1
    if kv_block_start > q_block_max:
        do_compute = 0

# 2. 逐元素因果遮罩（核心修复）
if IS_CAUSAL:
    kv_global_pos = KV_OFFSET + start_n + offs_n
    causal_mask = q_global_pos[:, None] >= kv_global_pos[None, :]
    valid_mask = valid_mask & causal_mask
```

调用方 `blasst.py` 中，当前 chunk 改用内核原生因果遮罩：

```diff
 out_curr, lse_curr, _ = blasst_chunked_prefill(
     ...,
-    mask_buffer=get_mask_buffer(is_causal=True, ...),
+    mask_buffer=get_mask_buffer(is_causal=False, ...),
+    is_causal=True,
+    kv_offset=len(selected_blocks) * kvcache_manager.block_size,
 )
```

---

## 4. m_global 更新顺序

BLASST 算法中 `m_global` 的更新顺序为：**先检查跳过条件，后更新 `m_global`**。

```python
# 正确顺序
diff = m_local - m_global          # 使用旧的 m_global 做判定
max_diff = tl.max(diff, axis=0)
m_global = tl.maximum(m_global, m_local)  # 更新 m_global（无条件）

if max_diff < threshold_ln_lambda:
    do_compute = 0  # 跳过
```

如果顺序反过来（先更新后检查），则 `diff = m_local - max(m_global, m_local) ≤ 0` 恒成立，导致当 $\ln(\lambda) \geq 0$ 时所有 sub-block 均被跳过。

---

## 5. 验证结果

### 单元测试 (GPU 4)

| 测试 | 结果 |
|------|------|
| BLASST causal vs FlashAttention causal | max_diff=0.000488 ✅ |
| BLASST non-causal vs FlashAttention | max_diff=0.000244 ✅ |
| 完整 pipeline (historical + causal) | max_diff=0.000061 ✅ |
| m_global 语义验证 (≠ LSE) | mean(lse - m_global)=3.22 ✅ |

### RULER `niah_multikey_2` (5 samples, GPU 4)

| 配置 | 准确率 | Compute Density |
|------|--------|-----------------|
| FullAttention (基线) | 4/5 (80%) | 100% |
| BLASST λ=0.001 | **4/5 (80%)** ✅ | 50-88% |
| BLASST λ=0.5 (当前默认值) | 0/5 (0%) | 12-16% |

> **注意**: λ=0.001 时 BLASST 精确匹配 FullAttention 基线（sample 2 在两者中均失败）。默认 `fixed_lambda=0.5` 对 NIAH 检索任务过于激进，需进一步调优 λ 参数。

---

## 6. 相关文档

- [`docs/sparse_attention_blasst.md`](sparse_attention_blasst.md) — BLASST 算法概述
- [`docs/blasst_performance_analysis.md`](blasst_performance_analysis.md) — BLASST 性能分析
- [`docs/sparse_policy_architecture.md`](sparse_policy_architecture.md) — SparsePolicy 架构
