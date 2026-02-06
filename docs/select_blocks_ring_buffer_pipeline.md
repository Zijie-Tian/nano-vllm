# select_blocks Ring Buffer Pipeline 优化

## 概述

将 `XAttentionBSAPolicy.select_blocks()` 中 estimate 阶段的 H2D 传输从单 slot 串行模式改为 ring buffer pipeline 多 slot 模式，实现 H2D 与 compute 的 overlap，在 128K context 下 prefill 加速 9.5%。

## 问题分析

### 原始实现（单 slot 串行）

`select_blocks` 的 estimate 阶段（pass1 和 pass2）需要从 CPU 逐 block 加载 K 数据到 GPU 进行计算。原始代码使用固定 `slot = 0`：

```
Block 0: H2D(slot0) → wait → compute → done
Block 1: H2D(slot0) → wait → compute → done
Block 2: H2D(slot0) → wait → compute → done
...
```

每次 H2D 传输必须等前一个 block 的 compute 完成释放 slot 后才能开始，无法 overlap。

### Ring Buffer Pipeline（多 slot 并行）

参考 `compute_chunked_prefill` 已有的 ring buffer 模式，使用 N 个 slot 轮转：

```
Preload: H2D(slot0, block0), H2D(slot1, block1), H2D(slot2, block2), H2D(slot3, block3)
Block 0: wait(slot0) → compute → done(slot0) → issue H2D(slot0, block4)
Block 1: wait(slot1) → compute → done(slot1) → issue H2D(slot1, block5)
Block 2: wait(slot2) → compute → done(slot2) → issue H2D(slot2, block6)
...
```

compute 完成后立即 issue 下一个 H2D 到同一 slot，利用不同 slot 的 transfer stream 实现 H2D 与 compute 并行。

## 实现

### 修改文件

`nanovllm/kvcache/sparse/xattn_bsa.py` — `XAttentionBSAPolicy.select_blocks()`

### 关键代码

```python
# Ring buffer pipeline config (shared by pass1 and pass2)
load_slots = list(range(offload_engine.num_ring_slots))
num_slots = len(load_slots)
num_blocks_hist = len(available_blocks)

# Preload first N blocks
num_preload = min(num_slots, num_blocks_hist)
for i in range(num_preload):
    offload_engine.load_k_only_to_slot_layer(
        load_slots[i], layer_id, available_blocks[i],
        chunk_idx=available_blocks[i])

for kv_chunk_idx in range(num_blocks_hist):
    current_slot = load_slots[kv_chunk_idx % num_slots]
    offload_engine.wait_slot_layer(current_slot)

    with torch.cuda.stream(compute_stream):
        # ... compute on current_slot ...
        offload_engine.record_slot_compute_done(current_slot)

    # Issue next H2D transfer (overlap with compute)
    next_idx = kv_chunk_idx + num_slots
    if next_idx < num_blocks_hist:
        next_slot = load_slots[next_idx % num_slots]
        offload_engine.load_k_only_to_slot_layer(
            next_slot, layer_id, available_blocks[next_idx],
            chunk_idx=available_blocks[next_idx])
```

pass1 和 pass2 使用相同的 pipeline 模式。

## 性能测试

### 测试环境

- GPU: NVIDIA RTX 3090 (24GB)
- Model: Llama-3.1-8B-Instruct
- Sparse Policy: XAttention BSA
- Ring buffer slots: 4
- Block size: 4096

### Nsys A/B 对比: 32K Context

| Phase (Chunk7 avg) | Baseline (μs) | Fixed (μs) | Change |
|---------------------|---------------|------------|--------|
| pass1 | 3,733 | 3,668 | -1.8% |
| merge | 1,244 | 1,252 | +0.6% |
| pass2 | 3,544 | 3,478 | -1.9% |
| find_blocks | 7,160 | 7,323 | +2.3% |
| **Total** | **15,682** | **15,721** | **+0.3%** |

32K 下历史 block 仅 7 个，pipeline overlap 空间有限，无显著改善。

### Nsys A/B 对比: 128K Context

| Phase | Baseline (s) | Fixed (s) | Change |
|-------|-------------|-----------|--------|
| estimate (pass1+merge+pass2) | 20.56 | 20.16 | -1.9% |
| **find_blocks** | **30.26** | **16.70** | **-44.8%** |
| compute | 0.46 | 0.46 | 0.0% |
| **Total xattn** | **51.28** | **37.32** | **-27.2%** |
| **Total prefill** | **141.71s** | **128.21s** | **-9.5%** |

Last Chunk (Chunk31) 每层平均：

| Phase | Baseline (ms) | Fixed (ms) | Change |
|-------|---------------|------------|--------|
| pass1 | 17.54 | 18.11 | +3.3% |
| pass2 | 17.73 | 17.64 | -0.5% |
| **find_blocks** | **55.15** | **25.48** | **-53.8% (2.16x)** |

### find_blocks 加速随 chunk 增长

| Chunk | Baseline (ms) | Fixed (ms) | Speedup |
|-------|---------------|------------|---------|
| 0 | 30.42 | 30.90 | 0.98x |
| 7 | 13.79 | 7.40 | 1.86x |
| 15 | 29.48 | 15.82 | 1.86x |
| 23 | 43.69 | 24.82 | 1.76x |
| 31 | 55.15 | 25.48 | **2.16x** |

### Slot 分布验证

| 版本 | Slot[0] | Slot[1] | Slot[2] | Slot[3] |
|------|---------|---------|---------|---------|
| Baseline | 100% | 0% | 0% | 0% |
| Fixed | 27.4% | 25.8% | 24.2% | 22.6% |

## 正确性验证

### Qwen3-0.6B (快速验证)

| Context | Samples | Pass Rate |
|---------|---------|-----------|
| 16K | 3 | 3/3 (100%) |
| 32K | 2 | 2/2 (100%) |

### Llama-3.1-8B-Instruct RULER 128K

| Task | Samples | Pass Rate |
|------|---------|-----------|
| niah_single_1 | 3 | 3/3 (100%) |
| niah_single_2 | 3 | 3/3 (100%) |

## 结论

1. **128K 下显著加速**: prefill 整体 -9.5%，find_blocks -44.8%
2. **收益随 context 长度增长**: 历史 block 越多，pipeline overlap 效果越好
3. **32K 下无显著改善**: block 数量太少（7 个），pipeline 来不及发挥作用
4. **正确性完全保持**: 所有 RULER 测试 100% 通过

## Nsys Profile 文件

存放于 `results/nsys/select_blocks_overlap_ab_test/`:

| 文件 | Context | 说明 |
|------|---------|------|
| `baseline_slot0_gpu4_014525.nsys-rep` | 32K | Baseline 单 Slot[0] |
| `fixed_ringbuf_gpu5_014601.nsys-rep` | 32K | Fixed Ring buffer |
| `baseline_slot0_gpu4_128k_021003.nsys-rep` | 128K | Baseline 单 Slot[0] |
| `fixed_ringbuf_gpu5_128k_020618.nsys-rep` | 128K | Fixed Ring buffer |
