# GPU-Only XAttention 指南

本文档介绍 GPU-only 模式下 XAttention BSA 的实现、内存优化和性能分析。

## 概述

GPU-only 模式下，所有 KV cache 存储在 GPU 上，无需 CPU offload。XAttention 通过稀疏注意力加速 prefill 阶段。

### 执行路径对比

| 模式 | Prefill 方法 | Decode 方法 | KV 存储 |
|------|-------------|-------------|---------|
| GPU-only Full | `compute_prefill()` | `compute_decode()` | GPU |
| GPU-only XAttn | `compute_prefill()` | `compute_decode()` | GPU |
| CPU Offload | `compute_chunked_prefill()` | `compute_chunked_decode()` | CPU + GPU |

## 架构设计

### SparsePolicy 接口

```python
class SparsePolicy:
    # GPU-only 方法
    def compute_prefill(self, q, k, v, ...) -> Tensor
    def compute_decode(self, q, k_cache, v_cache, ...) -> Tensor

    # CPU Offload 方法
    def compute_chunked_prefill(self, q, k, v, ...) -> Tensor
    def compute_chunked_decode(self, q, ...) -> Tensor

    # 初始化方法
    def initialize(self, num_layers, ...) -> None           # CPU offload metadata
    def alloc_policy_metadata(self, num_heads, ...) -> None  # GPU-only buffers
```

### XAttentionBSAPolicy 实现

```
GPU-only Prefill 流程:
┌─────────────────────────────────────────────────────────────┐
│  1. GQA 扩展 (使用预分配 buffer)                              │
│     K: [seq, kv_heads, dim] → K_exp: [1, heads, seq, dim]   │
│                                                              │
│  2. XAttention 估计                                          │
│     flat_group_gemm_fuse_reshape_kernel (Q@K^T)             │
│     softmax_fuse_block_sum_kernel (block 重要性)             │
│     → sparse mask                                            │
│                                                              │
│  3. BSA 稀疏注意力                                           │
│     flash_fwd_block_kernel (只计算选中的 blocks)             │
│     → output                                                 │
└─────────────────────────────────────────────────────────────┘
```

## 内存预分配

### 问题背景

XAttention 的 `compute_prefill()` 需要 GQA 扩展：

```python
# 之前: 动态分配 (~2GB for 64K)
K_exp = K.repeat_interleave(num_groups, dim=1)  # 分配 1
k_bsa = k.repeat_interleave(num_groups, dim=1)  # 分配 2 (重复!)
```

每次 prefill 都动态分配，导致：
- 内存碎片
- 分配延迟
- 可能 OOM

### 解决方案: alloc_policy_metadata()

在框架初始化时预分配 buffer：

```python
class XAttentionBSAPolicy(SparsePolicy):
    def alloc_policy_metadata(self, num_heads, num_kv_heads, head_dim,
                               max_seq_len, dtype, device):
        # 预分配 GQA 扩展 buffer
        shape = (1, num_heads, max_seq_len, head_dim)
        self._k_expanded = torch.empty(shape, dtype=dtype, device=device)
        self._v_expanded = torch.empty(shape, dtype=dtype, device=device)

    def compute_prefill(self, q, k, v, ...):
        seq_len = k.shape[0]
        # 使用预分配 buffer 的 slice
        K_exp = self._k_expanded[:, :, :seq_len, :]
        # 原地 GQA 扩展
        K_exp.view(...).copy_(K.unsqueeze(2).expand(...))
        # 复用同一 buffer 给 BSA
        k_bsa = K_exp.squeeze(0).transpose(0, 1)
```

### 内存使用

| 序列长度 | 预分配大小 | 说明 |
|---------|-----------|------|
| 32K | 512 MB | `2 * 32 * 32768 * 128 * 2 bytes` |
| 64K | 1024 MB | `2 * 32 * 65536 * 128 * 2 bytes` |

优化效果：
- 之前: ~2GB 动态分配 (xattn_estimate + BSA 各一次)
- 之后: ~1GB 预分配 (复用同一 buffer)

### 框架集成

```python
# model_runner.py - allocate_kv_cache()
def allocate_kv_cache(self):
    # ... KV cache 分配 ...

    # GPU-only 模式: 预分配 policy buffers
    if not config.enable_cpu_offload:
        self.kvcache_manager.sparse_policy.alloc_policy_metadata(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=config.max_model_len,
            dtype=dtype,
            device=torch.device("cuda"),
        )
```

## 性能分析

### 32K Prefill 性能

| Policy | Throughput | 相对提升 |
|--------|------------|----------|
| Baseline | 4880 tok/s | - |
| Full | 4892 tok/s | +0.2% |
| **XAttention** | **5602 tok/s** | **+15%** |

### 64K Prefill 性能

| Policy | Throughput | 相对提升 |
|--------|------------|----------|
| Baseline | 3386 tok/s | - |
| Full | 3355 tok/s | -0.9% |
| **XAttention** | **4775 tok/s** | **+41%** |

### Kernel 时间分解 (32K)

**XAttention:**
```
FFN GEMM:           3219 ms  (54%)
BSA Attention:      1231 ms  (21%)
XAttn Estimation:    415 ms  (7%)
Other:              1020 ms  (18%)
─────────────────────────────
Total:              5885 ms
```

**Full:**
```
FFN GEMM:           3244 ms  (48%)
Dense Attention:    2861 ms  (43%)
Other:               595 ms  (9%)
─────────────────────────────
Total:              6700 ms
```

### 加速来源

```
Dense Attention:    2861 ms
BSA Attention:      1231 ms  (节省 1630 ms, -57%)
XAttn Estimation:    415 ms  (额外开销)
─────────────────────────────
净节省:             1215 ms  (42% attention 时间)
```

## CUDA Graph 限制

### 为什么 Prefill 不能用 CUDA Graph

CUDA Graph 要求所有操作在 capture 时确定：

| 必须固定 | Prefill 的情况 |
|---------|---------------|
| Tensor 形状 | seq_len 可变 (1 ~ max_model_len) |
| Kernel grid | 依赖 seq_len |
| 内存地址 | 中间 tensor 大小变化 |

```python
# 不同请求的 seq_len 不同
request_1: prefill(seq_len=1024)   # grid=(8, 32, 1)
request_2: prefill(seq_len=32768)  # grid=(256, 32, 1)
```

### Decode 可以用 CUDA Graph

```python
# Decode 每次只处理 1 token
q: [batch_size, 1, heads, dim]  # 形状固定
```

nanovllm 为每个 batch_size 预先 capture 一个 graph：

```python
def capture_cudagraph(self):
    for batch_size in [1, 2, 4, 8, ...]:
        with torch.cuda.graph(g):
            self.run_model(dummy_input, is_prefill=False)
        self.graphs[batch_size] = g
```

### Nsys Profile 结果

```
XAttention 32K Prefill:
  Total kernels: 41,904
  Non-graph: 41,904 (100%)
  Graph: 0

Full 32K Prefill:
  Total kernels: 35,308
  Non-graph: 35,308 (100%)
  Graph: 0
```

**两者都是 100% NON-GRAPH**，这是 prefill 的本质特性。

## Profiling 工具

### 使用 profile.sh

```bash
# XAttention 32K
bash scripts/profile.sh --max-len 32768 --policy xattn

# Full 32K
bash scripts/profile.sh --max-len 32768 --policy full

# 64K (需要降低 gpu-util)
bash scripts/profile.sh --max-len 65536 --policy xattn --gpu-util 0.7
```

### 分析 nsys 结果

```bash
# 查看 kernel 统计
nsys stats --report cuda_gpu_kern_sum results/nsys/<file>.nsys-rep

# 用 sqlite 查询详细数据
sqlite3 results/nsys/<file>.sqlite "
SELECT
    (SELECT value FROM StringIds WHERE id = shortName) as kernel,
    COUNT(*) as count,
    SUM(end-start)/1e6 as total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY shortName
ORDER BY total_ms DESC
LIMIT 10
"
```

## 使用指南

### 启用 XAttention GPU-only

```python
from nanovllm import LLM
from nanovllm.config import SparsePolicyType

llm = LLM(
    model_path,
    max_model_len=32768,
    sparse_policy=SparsePolicyType.XATTN_BSA,
    gpu_memory_utilization=0.9,  # 64K 时可能需要降低
)
```

### 命令行测试

```bash
# bench.py
python bench.py --max-len 32768 --policy xattn

# 64K 需要降低 gpu-util
python bench.py --max-len 65536 --policy xattn --gpu-util 0.7
```

### 最佳实践

1. **32K 及以下**: 使用默认 `gpu_memory_utilization=0.9`
2. **64K**: 降低到 `gpu_memory_utilization=0.7`
3. **Decode**: XAttention 自动 fallback 到 FullAttentionPolicy
4. **Paged KV Cache**: 当 `block_tables` 存在时自动 fallback 到 flash_attn

## 相关文档

- [Sparse Policy 架构](sparse_policy_architecture.md)
- [XAttention 算法详解](xattention_algorithm_guide.md)
- [BSA 接口文档](block_sparse_attn_interface.md)
