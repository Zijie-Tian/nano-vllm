# BLASST Density 数据采集指南

BLASST sparse attention 的 per-layer per-head density 数据采集与分析流程。

---

## 概述

BLASST 策略在 chunked prefill 过程中会记录两种 density 指标：

| 指标 | 含义 | 粒度 |
|------|------|------|
| **Compute Density** | mask_buffer 为 1 的比例（实际计算的 q×kv sub-block pair 占比） | per-head, per-chunk, per-layer |
| **KV Density** | 任意 q-block 需要的 KV sub-block 占比（IO 视角） | per-head, per-chunk, per-layer |

> **注意**：当前 `blasst.py` 中 `collect_density = True` 会对所有层采集数据。如果只需要部分层的数据，可改回 `collect_density = layer_id == 0` 等条件。

---

## 数据采集步骤

### 1. 运行 RULER 测试并收集日志

```bash
# 设置环境
source build/nano-vllm-envs.sh

# 动态 lambda (默认 a=16384, λ=a/L)
CUDA_VISIBLE_DEVICES=<GPU_ID> python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy BLASST \
    2>&1 | tee /tmp/blasst_dynamic.log

# 固定 lambda (示例: λ=0.001)
CUDA_VISIBLE_DEVICES=<GPU_ID> python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy BLASST \
    --blasst-lambda 0.001 \
    2>&1 | tee /tmp/blasst_lambda001.log
```

**可用 lambda 参数**：
- 不指定 `--blasst-lambda`：使用动态公式 `λ = a / seq_len`
- `--blasst-lambda 0.1`：较激进的 pruning
- `--blasst-lambda 0.001`：中等 pruning
- `--blasst-lambda 0.0001`：保守 pruning（接近 FullAttention）

### 2. 导出 CSV 数据

```bash
python scripts/export_blasst_density_csv.py <log_file> <output_dir>
```

**示例**：
```bash
python scripts/export_blasst_density_csv.py \
    /tmp/blasst_dynamic.log \
    results/density/dynamic_lambda
```

**输出文件**：
```
results/density/dynamic_lambda/
├── compute_layer_00.csv ~ compute_layer_31.csv   # per-head compute density
├── kvcache_layer_00.csv ~ kvcache_layer_31.csv   # per-head KV density
└── summary.csv                                    # scalar 汇总
```

**CSV 格式**（rows=chunks, cols=heads）：
```csv
chunk,H0,H1,H2,...,H31,avg
C0,16.00,12.80,23.80,...,1.60,15.53
C1,26.10,18.60,35.60,...,0.80,14.17
...
```

### 3. 生成分析报告（可选）

```bash
python scripts/analyze_blasst_density.py <log_file>
```

输出 Layer×Chunk 表格、层组平均、Top 密/疏层、Head 分析等汇总信息到 stdout。

---

## 已有数据

| 目录 | Lambda | Avg Compute | Avg KV Density |
|------|--------|------------|----------------|
| `results/density/dynamic_lambda/` | ~0.5 (动态) | 5.3% | 32.2% |
| `results/density/lambda_01/` | 0.1 | 10.9% | 57.1% |
| `results/density/lambda_001/` | 0.001 | 44.0% | 85.4% |
| `results/density/lambda_0001/` | 0.0001 | 74.9% | 98.5% |

测试条件：Llama-3.1-8B-Instruct, RULER niah_single_1, 32K context, RTX 3090, 全部 100% 准确率。

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `nanovllm/kvcache/sparse/blasst.py` | density 数据采集源码（`collect_density` 控制开关） |
| `scripts/export_blasst_density_csv.py` | 日志解析 → per-layer CSV 导出 |
| `scripts/analyze_blasst_density.py` | 日志解析 → stdout 汇总分析 |
| `results/density/` | CSV 数据存储目录 |
| `docs/blasst_perhead_density_analysis.md` | 之前的 per-head density 分析文档 |
