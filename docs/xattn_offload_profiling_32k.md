# XAttention Offload Profiling - 32K Context

Nsys profiling 分析 XAttention vs Full Attention 在 Offload 模式下的性能。

**测试日期**: 2026-02-05
**测试模型**: Llama-3.1-8B-Instruct
**Context**: 32K tokens
**GPU**: A100-80GB (GPU 0)

---

## 测试配置

| 参数 | Full | XAttention |
|------|------|------------|
| Policy | FULL | XATTN_BSA |
| Block size | 4096 | 4096 |
| GPU blocks | 4 | 4 |
| Threshold | - | 0.95 |
| Density | 100% | ~50% |

---

## XAttention 各阶段时间统计

### NVTX Markers Summary

| 阶段 | 总时间(ms) | 调用次数 | 平均时间(ms) | 说明 |
|------|------------|----------|--------------|------|
| xattn_find_blocks | 1155.1 | 256 | 4.51 | 块选择 (threshold-based) |
| xattn_estimate_pass1 | 588.3 | 256 | 2.30 | 第一轮: partial stats |
| xattn_compute_historical | 512.0 | 224 | 2.29 | 历史 KV attention |
| xattn_estimate_pass2 | 501.6 | 256 | 1.96 | 第二轮: block sums |
| xattn_estimate_merge | 197.9 | 256 | 0.77 | 合并 softmax stats |
| xattn_compute_merge | 93.8 | 256 | 0.37 | 计算结果合并 |
| xattn_compute_current | 59.2 | 256 | 0.23 | 当前 chunk attention |

### 时间分配

```
Total XAttention overhead: 3108 ms

Estimate 阶段: 1288 ms (41.4%)
  - pass1: 588 ms
  - pass2: 502 ms
  - merge: 198 ms

Find blocks: 1155 ms (37.2%)

Compute 阶段: 665 ms (21.4%)
  - historical: 512 ms
  - merge: 94 ms
  - current: 59 ms
```

---

## Chunk7 (最后一个 chunk) 对比

### Per-Layer 时间

| Policy | Layer 0 | Layer 1 | ... | Layer 31 | Avg |
|--------|---------|---------|-----|----------|-----|
| Full | 36.5 ms | 33.6 ms | ... | 32.7 ms | ~35 ms |
| XAttn | 39.7 ms | 39.3 ms | ... | 38.5 ms | ~38 ms |

### 分析

Chunk7 是序列的最后 ~4K tokens (3813 tokens)，此时：
- K 长度: 32485 tokens
- Density: 42.08%

**结论**: XAttention 在 Chunk7 比 Full 慢约 8%，原因：
1. Estimate 开销无法被稀疏计算收益抵消
2. 42% density 仍然较高，稀疏收益有限

---

## Full Attention Chunk7 详细数据

```
Layer  Time(ms)
L0     36.5
L1     44.3
L2     43.7
L3     38.7
L4     34.2
L5     45.2
...
L31    32.7
Avg    ~35
```

---

## XAttention Chunk7 详细数据

```
Layer  Time(ms)
L0     39.7
L1     39.3
L2     37.1
L3     39.1
L4     38.7
L5     39.4
...
L31    38.5
Avg    ~38
```

---

## 性能瓶颈分析

### 1. xattn_find_blocks 开销过高

- 平均 4.51 ms per call
- 占总时间 37.2%
- 原因: threshold-based 块选择涉及排序和累积求和

### 2. 两轮 estimate 开销

- Pass1 + Pass2 共 1090 ms
- 需要遍历所有 KV chunks 两次
- 可优化方向: 单轮 estimate

### 3. Compute 阶段相对高效

- 只占 21.4%
- 说明 BSA 稀疏计算本身效率不错

---

## 优化建议

### 短期

1. **减少 find_blocks 开销**
   - 使用 top-k 而不是 threshold
   - 预分配 mask buffer 避免动态分配

2. **合并 estimate 两轮**
   - 在单轮中同时计算 stats 和 block sums

### 中期

1. **estimate 阶段使用更小的 block_size**
   - 当前 block_size=4096 对 estimate 不友好
   - 参考 `docs/estimate_block_size_performance.md`

2. **Pipeline estimate 和 H2D**
   - 将 estimate 与下一个 chunk 的 H2D 重叠

### 长期

1. **预测式块选择**
   - 基于历史 pattern 预测下一个 chunk 的重要 blocks
   - 减少 estimate 开销

---

## 相关文件

- `results/nsys/full_offload_32k_blk4096_20260205_023257.nsys-rep`
- `results/nsys/xattn_offload_32k_blk4096_20260205_023435.nsys-rep`

---

## 命令

### Profile Full
```bash
bash scripts/profile_offload.sh --policy full --ctx-len 32k --gpu 0 --model ~/models/Llama-3.1-8B-Instruct
```

### Profile XAttention
```bash
bash scripts/profile_offload.sh --policy xattn --ctx-len 32k --gpu 0 --model ~/models/Llama-3.1-8B-Instruct
```

### 分析 NVTX
```bash
nsys stats --report nvtx_pushpop_sum <file>.nsys-rep
```
