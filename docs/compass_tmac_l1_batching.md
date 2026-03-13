# COMPASS TMAC L1 批量化优化：深入分析

本文档记录了 COMPASS 稀疏策略中 TMAC qGEMM 的多线程并行化、L1 批量化优化以及端到端性能分析的完整过程。

## 1. 背景

COMPASS 策略在 prefill 阶段使用 CPU 端 TMAC qGEMM 预测每个 K sub-block 的重要性分数，以决定哪些 KV cache blocks 需要加载到 GPU。每层的 `select_blocks` 方法执行以下循环：

```
for q_grp in range(32):        # 32 Q-groups (4096 / 128)
    for kv_h in range(8):      # 8 KV heads
        1. _preprocess_q_per_head(q_group, kv_h)   → QLUT 生成
        2. _estimate_head_batched(layer, kv_h, ...) → qGEMM 估算
```

共 256 次内层调用/层，这是性能瓶颈的核心。

## 2. TVM qGEMM 并行维度分析

### 2.1 并行维度选择逻辑

`QGeMMLUTBitsCodegen._schedule` 中的核心逻辑：

```python
if self.num_threads > 1:
    N = int(C.shape[0])
    if N // self.bn >= self.num_threads:   # bn=8
        sch[C].parallel(no)   # N 维度并行（Query heads）
    else:
        sch[C].parallel(mo)   # M 维度并行（KV sub-blocks）
```

- **N 维度并行**：沿 Query heads 方向并行，数据局部性好（共享 A 矩阵），但受限于 N 值大小
- **M 维度并行**：沿 KV sub-blocks/tokens 方向并行，适合长序列场景

### 2.2 COMPASS 场景适配

| 配置 | N 值 | N//bn | 判定 | 并行维度 |
|------|------|-------|------|---------|
| L1 (per-head) | 1 | 0 | < 32 | **M 维度** ✓ |
| L2 (batch heads) | 8 | 1 | < 32 | **M 维度** |

COMPASS 的 L1 配置（N=1）自动走 M 维度并行，适合长 KV 序列。只需设置 `num_threads=32` 即可启用。

### 2.3 Preprocessor 并行

`QGeMMLUTBitsPreprocessorCodegen._schedule` 的多线程**被显式禁用**（第 696-700 行注释掉的代码），因为 C++ 层面通信开销过大。Preprocessor 始终单线程执行。

## 3. 32 线程 Microbench 结果

使用 `bench_compass_batched.py`，编译 `num_threads=32`：

| Blocks | KV Tokens | Micro (原始) | L1 (32t) | **L1 加速** |
|--------|-----------|-------------|----------|------------|
| 1 | 4K | 0.110s | 0.020s | **5.6×** |
| 4 | 16K | 0.500s | 0.016s | **31.1×** |
| 7 | 28K | 0.815s | 0.019s | **42.0×** |
| 16 | 64K | 1.735s | 0.023s | **74.6×** |
| 32 | 128K | 3.378s | 0.030s | **114.2×** |

### 32K 全序列估算（32 layers, chunks 1-7）

| 方法 | 总耗时 | vs BLASST |
|------|--------|-----------|
| Micro (原始) | 112.9s | 16.8× slower |
| **L1 batch (32t)** | **3.4s** | **0.5× (更快)** |
| BLASST (GPU Triton) | 6.7s | 基线 |

> **关键发现**：纯 TVM qGEMM 计算在 L1+32 线程下已比 BLASST GPU 内核更快（3.4s < 6.7s）。

## 4. 端到端性能分析

### 4.1 为什么 32 线程在端到端中帮助有限

将 `num_threads` 从 1 改为 32 后：

| 配置 | 端到端每层时间 | 改善 |
|------|-------------|------|
| 1 thread (原始) | ~5.5s | — |
| 32 threads | ~5.5s | **无改善** |

原因：TVM qGEMM 计算只占每层时间的 ~0.4%，剩余 99.6% 是 Python 封装开销。

### 4.2 时间 Breakdown（1 block, 1 layer）

```
实际系统: 4.7s/层
Microbench: 0.020s/层 (L1 batch)
差距: 235×
```

| 操作 | 调用次数/层 | 单次耗时 | 总计 | 占比 |
|------|-----------|---------|------|------|
| `_preprocess_q_per_head` | 256 | ~15ms | **~3.8s** | **81%** |
| `_estimate_head_batched` (无 pre-compute) | 256 | ~5ms | ~1.3s | 28% |
| `_prepare_head_data` (优化后) | 8 | ~20ms | ~0.16s | 3% |
| `func_batched()` + score extract | 256 | ~2ms | ~0.5s | 11% |

### 4.3 _preprocess_q_per_head 的内部开销

每次调用执行以下操作：

```python
# 1. PyTorch → float16 转换
q_fp16 = q_group.to(torch.float16)

# 2. pack_scales_tmac: 打包 Q 数据为 TMAC 格式
packed_q = pack_scales_tmac(q_tmac, ...)

# 3. PyTorch → NumPy → TVM 数据拷贝
Scales_np = packed_q.numpy()
qlut_np = np.zeros(...)
lut_s_np = np.zeros(...)
lut_b_np = np.zeros(...)

# 4. TVM NDArray 分配
qlut_tvm = tvm.nd.array(qlut_np, dev)
lut_s_tvm = tvm.nd.array(lut_s_np, dev)
lut_b_tvm = tvm.nd.array(lut_b_np, dev)

# 5. TVM preprocessor 内核执行（单线程）
func_pp(q_tvm, Scales_tvm, qlut_tvm, lut_s_tvm, lut_b_tvm)
```

其中步骤 2-4（数据封装）占 ~80% 时间，步骤 5（TVM 内核）很快。

## 5. Pre-compute Per-Head 优化

### 5.1 优化内容

新增 `_prepare_head_data()` 方法，在 Q-group 循环**之前**为每个 KV head 预计算：
- `A_tvm`：packed K cache（TVM NDArray）
- `Scales_tvm`：packed scales/zeros（TVM NDArray）
- `C_tvm`：输出缓冲区（TVM NDArray）

这将 `_estimate_head_batched` 内部的 `pack_scales_tmac` + `numpy` + `tvm.nd.array` 从 **256 次/层**减少到 **8 次/层**。

### 5.2 端到端效果

| 配置 | 总 prefill (32K) | 每层平均 | vs 基线 |
|------|-----------------|---------|--------|
| 基线 (1t, 无优化) | ~1320s | ~5.5s | — |
| **32t + pre-compute** | **1160s** | **~4.7s** | **~12%** |

Timer Breakdown（优化后）：

```
select_blocks:          1151.860s  (99.3%)
compute_chunked_prefill:   7.557s  ( 0.7%)
offload_prefill_chunk:     0.524s  ( 0.0%)
TOTAL:                  1159.942s
```

准确性：✅ NIAH score=1.00

### 5.3 为什么只有 12% 改善

从 5.5s → 4.7s，减少了约 0.8s/层。这正好对应消除了 248 次冗余的 `pack_scales_tmac` + `tvm.nd.array` 调用（约 3-4ms × 248 ≈ 0.75-1.0s）。

剩余 4.7s 中，**~3.8s 被 `_preprocess_q_per_head` 占据**（256 次 QLUT 生成）。

## 6. 下一步优化方向

### 6.1 优化 _preprocess_q_per_head（最高优先级）

当前瓶颈占 ~81%。可能的优化路径：

1. **预分配 TVM NDArray 缓冲区**：避免每次调用创建新的 `tvm.nd.array`，改为 `.copyfrom()` 到预分配缓冲区
2. **使用 DLPack 零拷贝**：通过 `tvm.nd.from_dlpack()` 直接从 PyTorch tensor 创建 TVM NDArray，避免中间 NumPy 拷贝
3. **减少 Q-group 粒度**：从 32 组减到 8 或 4 组（降低精度但 8× 或 4× 减少调用次数）
4. **缓存 pack_scales_tmac 结果**：如果 Q 数据在层间有模式相似性，可以复用

### 6.2 跳过浅层估算

前几层（layer 0-3）通常选择全部 blocks（100% selection），TMAC 估算是浪费。可直接返回所有 available blocks。

### 6.3 Preprocessor 多线程化

当前 TVM preprocessor 的多线程被禁用。如果能解决 C++ 侧的通信开销，启用多线程可以加速 QLUT 生成。

### 6.4 理论下限估算

假设 `_preprocess_q_per_head` 通过 DLPack + 预分配优化到 ~3ms/次：
- 256 × 3ms = 0.77s
- 加上 qGEMM + score extract: ~0.5s
- 理论下限: **~1.3s/层** → 32K 总计 ~370s (~6 分钟)

如果进一步减少 Q-group 到 8 组（64 calls/层）：
- 64 × 3ms + 0.5s = ~0.7s/层 → 32K 总计 ~200s (~3.3 分钟)

## 7. 代码变更记录

| 文件 | 变更 |
|------|------|
| `compass.py` L187,207,213 | `num_threads` 1→32（qGEMM 编译） |
| `compass.py` L380-464 | 新增 `_prepare_head_data()` 方法 |
| `compass.py` L289-299 | `_estimate_head_batched` 增加 `head_data` 参数 |
| `compass.py` L558-570 | `select_blocks` 预计算 head_data_cache |
| `bench_compass_batched.py` | 扩展到 128K tokens，32 线程编译 |

---

**Author**: Zijie Tian / Gemini CLI  
**Date**: 2026-03-14  
**Status**: Pre-compute per-head 已实现，_preprocess_q_per_head 优化待做
