# test_ruler.py 使用指南

RULER benchmark 综合测试工具，用于评估 LLM 长上下文能力。

**测试日期**: 2026-02-05
**测试 GPU**: RTX 3090 (GPU 4)

---

## 支持的任务

| 类别 | 任务 |
|------|------|
| NIAH (Needle-In-A-Haystack) | `niah_single_1/2/3`, `niah_multikey_1/2/3`, `niah_multiquery`, `niah_multivalue` |
| QA (Question Answering) | `qa_1`, `qa_2` |
| Recall | `cwe`, `fwe`, `vt` |

---

## 基本命令格式

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py [OPTIONS]
```

---

## 参数说明

### 必要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `~/models/Llama-3.1-8B-Instruct` | 模型路径 |
| `--data-dir` | `tests/data/ruler_64k` | 数据目录 |
| `--max-model-len` | 65664 | 最大上下文长度 |

### 数据选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | 全部 | 逗号分隔的数据集名 |
| `--num-samples` | 0 (全部) | 每个数据集测试样本数 |
| `--sample-indices` | - | 指定样本索引 (如 `0,5,10`) |

### Offload 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-offload` | False | 启用 CPU offload 模式 |
| `--num-gpu-blocks` | 4 | GPU 上的 KV cache blocks 数量 |
| `--block-size` | 4096 | KV cache block 大小 (tokens) |
| `--num-kv-buffers` | 4 | Ring buffer 数量 |
| `--gpu-utilization` | 0.9 | GPU 显存利用率 |

### Sparse Attention 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sparse-policy` | - | 稀疏策略: `FULL`, `QUEST`, `XATTN_BSA` |
| `--sparse-threshold` | 0.9 | XAttn cumulative attention 阈值 |
| `--sparse-samples` | 128 | XAttn 每 chunk 采样数 |
| `--sparse-stride` | 8 | XAttn Q/K 下采样步长 |

### 输出控制

| 参数 | 说明 |
|------|------|
| `--quiet` / `-q` | 安静模式 |
| `--json-output` | JSON 格式输出 |
| `--fresh-llm` | 每个样本重新初始化 LLM |

### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dtype` | auto | 模型数据类型 (`bfloat16`, `float16`) |
| `--use-cuda-graph` | False | 启用 CUDA Graph |
| `--max-new-tokens` | 16 | 最大生成 token 数 |

---

## 已验证的命令示例

以下命令均在 RTX 3090 (24GB) 上测试通过。

### 1. 基础 Offload 测试 (32K)

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload
```

**结果**: 100% 准确率, 耗时 ~16s

### 2. Offload + XAttention BSA (32K)

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

**结果**: 100% 准确率, compute density ~50%, 耗时 ~19s

### 3. Offload + XAttention BSA (64K)

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_64k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 72000 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

**结果**: 100% 准确率, compute density ~37%, 耗时 ~52s

### 4. 多数据集多样本测试

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1,qa_1 \
    --num-samples 2 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```

**结果**: 4/4 (100%), 耗时 ~71s

### 5. 指定样本索引测试

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --sample-indices 0,5,10 \
    --max-model-len 40960 \
    --enable-offload
```

### 6. JSON 输出模式 (用于脚本)

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --json-output
```

**输出格式**:
```json
{
  "total_correct": 1,
  "total_samples": 1,
  "overall_accuracy": 1.0,
  "avg_score": 1.0,
  "time": 30.44,
  "tasks": {"niah_single_1": {"correct": 1, "total": 1, "accuracy": 1.0}},
  "failed_samples": {}
}
```

### 7. 安静模式

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --quiet
```

### 8. 调整 GPU blocks 数量

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --num-gpu-blocks 8 \
    --sparse-policy XATTN_BSA
```

### 9. GLM-4 模型测试

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --dtype bfloat16
```

**结果**: 100% 准确率, 耗时 ~17s

---

## 数据目录结构

```
tests/data/
├── ruler_4k/      # 4K context
├── ruler_8k/      # 8K context
├── ruler_16k/     # 16K context
├── ruler_32k/     # 32K context (推荐测试)
├── ruler_64k/     # 64K context
├── ruler_128k/    # 128K context
├── ruler_256k/    # 256K context
├── ruler_512k/    # 512K context
├── ruler_768k/    # 768K context
└── ruler_1m/      # 1M context
```

每个目录包含 13 个任务子目录，每个任务有 `validation.jsonl` 文件。

---

## GPU 与模式选择

| GPU 显存 | 推荐模式 | 说明 |
|---------|---------|------|
| 24GB (3090/4090) | `--enable-offload` | 必须使用 offload |
| 40GB+ (A100) | 两种模式均可 | 可测试 GPU-only |

**RTX 3090 限制**: 由于显存限制，必须使用 `--enable-offload` 参数。

---

## max-model-len 设置指南

| 数据目录 | 推荐 max-model-len | 说明 |
|---------|-------------------|------|
| ruler_4k | 5000 | 留出 output 空间 |
| ruler_8k | 9000 | |
| ruler_16k | 17000 | |
| ruler_32k | 40960 | |
| ruler_64k | 72000 | |
| ruler_128k | 135000 | |

**公式**: `max_model_len >= max_input_len + max_new_tokens`

---

## DensityObserver 输出

使用 `--sparse-policy XATTN_BSA` 时自动启用，输出示例：

```
============================================================
Density Statistics (XAttention BSA)
============================================================
[DensityObserver] Mode: offload
  Compute density: 0.3691 (min: 0.3691 @ layer 0)
  Comm density:    1.0000 (CPU block granularity)
  Savings ratio:   0.0% H2D transfer reduction
  Num layers: 1
  Layer 0 density: 0.369052
```

| 指标 | 说明 |
|------|------|
| Compute density | BSA block (128 tokens) 粒度的计算密度 |
| Comm density | CPU block (4096 tokens) 粒度的通信密度 |
| Savings ratio | H2D 传输减少比例 |

---

## 常见问题

### 1. OOM 错误

**原因**: 显存不足
**解决**:
- 使用 `--enable-offload`
- 降低 `--gpu-utilization`
- 减少 `--num-gpu-blocks`

### 2. 模型加载失败

**原因**: 模型配置不兼容
**解决**:
- 检查 `--dtype` 参数 (GLM 模型需要 `--dtype bfloat16`)
- 确认模型路径正确

### 3. 准确率异常

**原因**: 状态泄漏
**解决**: 使用 `--fresh-llm` 参数为每个样本重新初始化 LLM

---

## 相关文档

- [`docs/xattn_density_types.md`](xattn_density_types.md) - Compute vs Comm density 解释
- [`docs/xattn_density_alignment_verification.md`](xattn_density_alignment_verification.md) - GPU-only vs Offload 对齐验证
- [`docs/ruler_benchmark_results_32k.md`](ruler_benchmark_results_32k.md) - RULER 32K 基准测试结果
