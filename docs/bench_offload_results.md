# CPU Offload Benchmark Results

本文档记录 `bench_offload.py` 在不同配置下的性能测试结果。

## 测试环境

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA A100-SXM4-80GB |
| 模型 | Llama-3.1-8B-Instruct |
| GPU slots | 4 |

## Sparse Policy 配置

| 策略 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| FULL | Full Attention | Full Attention | 基线，加载所有 blocks |
| XATTN_BSA | XAttention (tau=0.95, stride=8) | Full Attention (fallback) | 稀疏 prefill |

## 测试结果

### Block Size 4096 (推荐)

#### GPU-only 模式

| 上下文 | Full Attention | XAttention | 相对性能 |
|--------|----------------|------------|----------|
| 32K | 4863 tok/s | 5587 tok/s | **+14.9%** ✅ |
| 64K | 3373 tok/s | 4766 tok/s | **+41.3%** ✅ |

#### CPU Offload 模式 (优化后, 2026-01-28)

| 上下文 | Full Attention | XAttention | 相对性能 |
|--------|----------------|------------|----------|
| 32K | 4678 tok/s | 4398 tok/s | **-6.0%** |
| 64K | 3331 tok/s | 3203 tok/s | **-3.8%** |
| 128K | 2144 tok/s | 2196 tok/s | **+2.4%** ✅ |

#### CPU Offload 模式 (优化前, 2026-01-27)

| 上下文 | Full Attention | XAttention | 相对性能 |
|--------|----------------|------------|----------|
| 32K | 4648 tok/s | 4002 tok/s | **-13.9%** ❌ |
| 64K | 3329 tok/s | 2642 tok/s | **-20.6%** ❌ |
| 128K | 2122 tok/s | 867 tok/s | **-59.1%** ❌ |

### Block Size 256 (小 block 测试)

#### CPU Offload 模式 (64K)

| 策略 | 耗时 | 吞吐量 | 相对性能 |
|------|------|--------|----------|
| Full Attention | 401.04s | 163.41 tok/s | baseline |
| XAttention BSA | 390.35s | 167.89 tok/s | **+2.7%** ✅ |

### Block Size 1024 (历史测试)

#### CPU Offload 模式

| 上下文 | Full Attention | XAttention | 相对性能 |
|--------|----------------|------------|----------|
| 32K | 1587.74 tok/s | 1172.33 tok/s | -26% |
| 128K | 552.63 tok/s | 466.17 tok/s | -16% |

## 关键发现

### 1. GPU-only vs CPU Offload 模式差异

| 模式 | XAttention 效果 | 原因 |
|------|-----------------|------|
| **GPU-only** | ✅ 显著加速 (+15% ~ +41%) | 计算是瓶颈，稀疏注意力减少 FLOPs |
| **CPU Offload (优化后)** | ✅ 长上下文略有收益 | estimate_block_size 优化减少估计开销 |
| **CPU Offload (优化前)** | ❌ 性能下降 (-14% ~ -59%) | 传输是瓶颈，稀疏估计增加额外开销 |

### 2. Block Size 对性能的影响

| Block Size | 64K Full (Offload) | 特点 |
|------------|-------------------|------|
| 4096 | 3329 tok/s | ⭐ 最佳性能 |
| 1024 | ~1500 tok/s | 中等 |
| 256 | 163 tok/s | 极慢（20x 下降） |

**原因**: 更小的 block = 更多的 blocks = 更多 H2D 传输开销

### 3. XAttention 在小 Block Size 下反转

当 block size = 256 时，XAttention 反而略有优势 (+2.7%)：
- 256 个 blocks (vs 16 个 @ 4096)
- 稀疏跳过的 blocks 比例更明显
- 但绝对性能极差，不推荐使用

### 4. estimate_block_size 优化效果 (2026-01-28)

```
Offload 模式 XAttention 相对性能变化:
         优化前    优化后    改进
32K:    -13.9%   -6.0%    +7.9pp
64K:    -20.6%   -3.8%    +16.8pp
128K:   -59.1%   +2.4%    +61.5pp ✅
```

优化内容：
- `estimate_block_size` 从 4096 改为 1024
- `softmax_fuse_block_sum` kernel 时间从 48% 降到 1% (44x 加速)
- 选择策略从 mask + voting 改为 score + threshold

优化后结论：
- **128K 长上下文 XAttention 反超 Full Attention**
- 短上下文仍有少量开销，但已显著减少

## 结论

### 推荐配置 (优化后, 2026-01-28)

| 场景 | 推荐策略 | Block Size |
|------|----------|------------|
| GPU-only (VRAM 充足) | XAttention | 4096 |
| CPU Offload (128K+) | XAttention | 4096 |
| CPU Offload (32K-64K) | Full Attention 或 XAttention | 4096 |

### XAttention 适用条件 (优化后)

✅ **适合**:
- GPU-only 模式（计算密集）
- CPU Offload + 长上下文（128K+）有正向收益
- 长上下文（64K+）收益更大

⚠️ **中性**:
- CPU Offload + 中等上下文（32K-64K）：略慢 3-6%，可接受

❌ **不推荐**:
- 短上下文（<32K）收益不明显

## 运行命令

```bash
# GPU-only 模式
CUDA_VISIBLE_DEVICES=0 python bench.py --max-len 65536 --block-size 4096 --gpu-util 0.7
CUDA_VISIBLE_DEVICES=0 python bench.py --max-len 65536 --block-size 4096 --gpu-util 0.7 --policy xattn

# CPU Offload 模式 (推荐 block-size 4096)
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 65536 --block-size 4096
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 65536 --block-size 4096 --enable-xattn

# CPU Offload 模式 (小 block size 测试)
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 65536 --block-size 256
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --max-len 65536 --block-size 256 --enable-xattn

# 调整 XAttention 参数
CUDA_VISIBLE_DEVICES=0 python bench_offload.py --enable-xattn --xattn-threshold 0.8 --xattn-stride 16
```

## FlashInfer Merge 优化 (2026-01-28)

将 Triton 实现的 `merge_attention_outputs` 替换为 FlashInfer 的 `cascade.merge_state`。

### 性能对比 (Full Attention, block-size 4096)

| 上下文 | Triton merge | FlashInfer merge | 提升 |
|--------|--------------|------------------|------|
| 32K | 4678 tok/s | 4717 tok/s | **+0.8%** |
| 64K | 3331 tok/s | 3411 tok/s | **+2.4%** |
| 128K | 2144 tok/s | 2178 tok/s | **+1.6%** |

### 关键发现

1. **端到端提升有限**（0.8% ~ 2.4%）：merge 操作不是主要瓶颈
   - H2D 传输占主导（64K 传输 64GB）
   - Attention 计算是另一主要耗时
   - Merge 在总耗时中占比很小

2. **Merge kernel 单独对比**（长序列时 FlashInfer 优势明显）：

| seq_len | heads | Triton (ms) | FlashInfer (ms) | Speedup |
|---------|-------|-------------|-----------------|---------|
| 4096 | 32 | 0.129 | 0.087 | **1.49x** |
| 8192 | 32 | 0.251 | 0.147 | **1.70x** |
| 16384 | 32 | 0.499 | 0.274 | **1.82x** |

3. **短序列 FlashInfer 反而慢**：格式转换开销（squeeze, transpose, contiguous）

### 技术细节

- **LSE 格式差异**：FlashInfer 使用 log2，flash_attn 使用 ln
- **转换系数**：`LOG2_E = 1.4427`（ln → log2），`LN_2 = 0.6931`（log2 → ln）
- **FlashInfer attention JIT 问题**：CUDA 版本兼容性问题，仅使用 merge_state

### 代码位置

- `nanovllm/ops/chunked_attention.py`: `merge_attention_outputs_flashinfer()`
- `nanovllm/kvcache/sparse/full_policy.py`: 3 处 import 更新
- `nanovllm/kvcache/sparse/xattn_bsa.py`: 1 处 import 更新

## 更新记录

- 2026-01-28: **FlashInfer merge 替换 Triton merge**，端到端提升 0.8% ~ 2.4%
- 2026-01-28: **estimate_block_size 优化后重新测试**，128K XAttention 反超 Full (+2.4%)
- 2026-01-27: 添加 GPU-only vs Offload 对比，block size 影响分析
- 2026-01-27: 初始测试，Llama-3.1-8B-Instruct, A100 80GB
