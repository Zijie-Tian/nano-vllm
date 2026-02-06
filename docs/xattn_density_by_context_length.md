# XAttention Density vs Context Length Benchmark

**测试日期**: 2026-02-05
**GPU**: RTX 3090 (GPU 4)
**模式**: CPU Offload + XAttn BSA
**数据集**: RULER niah_single_1 (1 sample)

---

## 测试配置

| 参数 | 值 |
|------|-----|
| sparse_policy | XATTN_BSA |
| sparse_threshold | 0.9 |
| enable_cpu_offload | true |
| chunk_size (Q) | 4,096 tokens |
| BSA block_size | 128 tokens |

### 测试命令模板

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model <MODEL_PATH> \
    --data-dir tests/data/ruler_<CTX> \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len <MAX_LEN> \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

---

## Llama-3.1-8B-Instruct 测试结果

**模型**: `~/models/Llama-3.1-8B-Instruct`
**架构**: GQA (32 layers, 8 KV heads)

### Density 汇总表

| Context | max-model-len | Compute Density | Min Density | Min Layer | Layer 0 Density | Comm Density |
|---------|---------------|-----------------|-------------|-----------|-----------------|--------------|
| 4K | 5000 | 51.33% | 31.93% | Layer 3 | 62.46% | N/A (无历史块) |
| 8K | 10000 | 48.78% | 27.90% | Layer 5 | 62.34% | 100% |
| 16K | 20000 | 45.37% | 23.57% | Layer 5 | 59.85% | 100% |
| 32K | 40960 | 38.10% | 17.52% | Layer 5 | 49.84% | 100% |
| 64K | 72000 | 28.27% | 11.87% | Layer 5 | 36.91% | 100% |
| 128K | 135000 | **23.49%** | **8.29%** | Layer 3 | 32.55% | 100% |

> **Comm Density 说明**: 由于 32 heads 的 `any()` 并集效应，BSA 级别的稀疏无法传导到 CPU block 级别。
> 详见 [`docs/xattn_density_types.md`](xattn_density_types.md) 中的数学证明。

### 测试命令

```bash
# 4K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_4k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 5000 \
    --enable-offload \
    --sparse-policy XATTN_BSA

# 8K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_8k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 10000 \
    --enable-offload \
    --sparse-policy XATTN_BSA

# 16K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_16k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 20000 \
    --enable-offload \
    --sparse-policy XATTN_BSA

# 32K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA

# 64K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_64k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 72000 \
    --enable-offload \
    --sparse-policy XATTN_BSA

# 128K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_128k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 135000 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```

### Llama 关键发现

1. **Density 随 context 增长持续下降**:
   - 4K → 128K: Compute density 从 51% 降到 23%
   - Min layer density 从 32% 降到 8.29%

2. **最稀疏的层**:
   - 4K/128K: Layer 3 最稀疏
   - 8K/16K/32K/64K: Layer 5 最稀疏

3. **128K 时某些层极其稀疏**:
   - Layer 3 的 density 仅 8.29%
   - 理论上可节省 91.7% 的该层通信量

---

## GLM-4-9B-Chat-1M 测试结果

**模型**: `~/models/GLM-4-9B-Chat-1M`
**架构**: MQA (40 layers, 1 KV head group → 展开为 32 heads)

### Density 汇总表

| Context | max-model-len | Compute Density | Min Density | Min Layer | Layer 0 Density | Comm Density |
|---------|---------------|-----------------|-------------|-----------|-----------------|--------------|
| 4K | 5000 | 56.89% | 32.81% | Layer 6 | 87.74% | N/A (无历史块) |
| 8K | 10000 | 49.43% | 21.07% | Layer 6 | 79.28% | 100% |
| 16K | 20000 | 43.57% | 15.72% | Layer 6 | 74.32% | 100% |
| 32K | 40960 | 40.36% | 13.85% | Layer 6 | 71.45% | 100% |
| 64K | 72000 | 37.26% | 11.90% | Layer 6 | 68.52% | 100% |
| 128K | 135000 | 35.25% | 10.06% | Layer 6 | 64.78% | 100% |
| 256K | 270000 | 25.54% | 6.21% | Layer 6 | 62.60% | 100% |
| 512K | 530000 | 24.46% | 6.69% | Layer 6 | 58.66% | 100% |
| 768K | 800000 | **23.40%** | **6.09%** | Layer 6 | 56.95% | 100% |

### 测试命令

```bash
# 4K-128K: 同 Llama 命令，替换模型路径为 ~/models/GLM-4-9B-Chat-1M

# 256K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_256k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 270000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --dtype bfloat16

# 512K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_512k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 530000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --dtype bfloat16

# 768K
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_768k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 800000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --dtype bfloat16
```

---

## 模型对比分析

### Llama vs GLM Density 对比

| Context | Llama Compute | GLM Compute | Llama Min | GLM Min |
|---------|---------------|-------------|-----------|---------|
| 4K | 51.33% | 56.89% | 31.93% | 32.81% |
| 8K | 48.78% | 49.43% | 27.90% | 21.07% |
| 16K | 45.37% | 43.57% | 23.57% | 15.72% |
| 32K | 38.10% | 40.36% | 17.52% | 13.85% |
| 64K | 28.27% | 37.26% | 11.87% | 11.90% |
| 128K | 23.49% | 35.25% | 8.29% | 10.06% |
| 256K | N/A | 25.54% | N/A | 6.21% |
| 512K | N/A | 24.46% | N/A | 6.69% |
| 768K | N/A | **23.40%** | N/A | **6.09%** |

### 架构差异影响

| 特性 | Llama-3.1-8B | GLM-4-9B |
|------|--------------|----------|
| 层数 | 32 | 40 |
| KV Heads | 8 (GQA) | 2 (MQA) |
| Attention 类型 | Grouped Query | Multi-Query |
| 最大 Context | 128K | 1M |

### GLM 关键发现

1. **Layer 6 始终是最稀疏的层**:
   - 所有 context length (4K-768K) 下，Layer 6 都是 min density 层
   - 与 Llama 的 Layer 3/5 不同，GLM 的稀疏模式非常稳定

2. **Density 下降趋势与 Llama 类似**:
   - 4K → 128K: Compute density 从 56.89% 降到 35.25%
   - 256K → 768K: 进一步降到 25.54% → 24.46% → 23.40%

3. **超长 context (256K+) 时 density 趋于稳定**:
   - 256K/512K/768K 的 compute density 非常接近 (25.5% → 24.5% → 23.4%)
   - 暗示存在一个 ~23% 的 density 下限

4. **极长 context 时 Layer 6 极其稀疏**:
   - 768K 时 Layer 6 density 仅 6.09%
   - 理论上可节省 93.9% 的该层计算量

---

## 结论

### Density 趋势

1. **Context 越长，Density 越低**: 这是预期行为，长文本中稀疏注意力模式更明显
2. **不同层的 Density 差异大**: Llama Layer 3/5、GLM Layer 6 特别稀疏
3. **模型架构影响 Density**: GQA vs MQA 导致不同的稀疏模式和统计行为

### 模型对比

| 指标 | Llama-3.1-8B | GLM-4-9B |
|------|--------------|----------|
| 最稀疏层 | Layer 3/5 (变化) | Layer 6 (稳定) |
| 128K min density | 8.29% | 10.06% |
| 768K min density | N/A | 6.09% |
| Density 下降速率 | 较快 | 较慢 |
| 极长 context density | N/A (max 128K) | ~23% (768K) |

### Comm Density = 100% 的根因

所有 context length 下 comm density 均为 100%，原因是 `select_blocks` 中的三级 `any()` 聚合:
- 32 个 head 的并集 → 不同 head 关注不同位置
- 所有 Q BSA block 的并集 → 不同 Q 位置关注不同 K 位置
- CPU block 内 32 个 BSA 子块的并集 → 粒度放大 32 倍

即使单个 head 的 compute density 仅 38%，32 heads 并集后覆盖率 → 100%。

### 优化建议

1. **按层独立优化**: 针对最稀疏的层使用更细粒度的 block selection
2. **动态阈值**: 根据 context length 动态调整 sparse_threshold
3. **Per-head H2D**: 每个 head 独立传输所需 blocks（需要修改 offload 架构）
4. **减小 CPU block 粒度**: 从 4096 降到更小值（但增加 pipeline 管理开销）

---

## 相关文档

- [`docs/xattn_768k_density_benchmark.md`](xattn_768k_density_benchmark.md) - 768K 详细测试报告
- [`docs/xattn_density_types.md`](xattn_density_types.md) - Compute vs Comm Density 定义
- [`docs/xattn_density_benchmark.md`](xattn_density_benchmark.md) - 早期 Density 基准测试
