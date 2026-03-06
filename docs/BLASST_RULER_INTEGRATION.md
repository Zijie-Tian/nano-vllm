# BLASST RULER Benchmark Integration

BLASST (Block-wise Sparse Attention with Softmax Thresholding) 稀疏注意力算法在 RULER benchmark 中的使用指南。

## 快速开始

```bash
# 默认配置运行 (lambda=0.5)
./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic blasst

# 自定义 lambda 参数
BLASST_LAMBDA=0.3 ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic blasst
```

## 参数配置

### 核心参数

| 环境变量 | 说明 | 默认值 | 推荐值 |
|----------|------|--------|--------|
| `BLASST_LAMBDA` | 块选择阈值 λ | 0.5 | 0.3-0.7 |

- **lambda 越小**：选择越少的块，越稀疏，速度更快
- **lambda 越大**：选择更多的块，越接近 full attention，精度更高

### 其他环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `GPU_LIST` | GPU 列表 | 2,3,4,5 |
| `MODEL_DIR` | 模型目录 | /home/zijie/models |

## 结果目录结构

结果文件夹自动包含 lambda 参数值：

```
benchmark_root/
├── blasst_lambda0.5_llama3.1-8b-nanovllm/    # lambda=0.5 (默认)
├── blasst_lambda0.3_llama3.1-8b-nanovllm/    # lambda=0.3
└── blasst_lambda0.7_llama3.1-8b-nanovllm/    # lambda=0.7
```

## 配置说明

### 当前默认配置

- **任务数量**: 13 个 RULER synthetic 任务
- **每个任务样本数**: 1 (NUM_SAMPLES=1)
- **GPU**: 2,3,4,5 (4卡并行)
- **序列长度**: 32K (config_models.sh 中 SEQ_LENGTHS 配置)

### 修改配置

#### 修改序列长度

编辑 `eval/RULER/scripts/config_models.sh`:

```bash
SEQ_LENGTHS=(
    32768    # 32K (当前配置)
    # 65536  # 64K
    # 131072 # 128K
)
```

#### 修改 GPU

编辑 `scripts/run_ruler.sh`:

```bash
export GPU_LIST="${GPULIST:-${GPU_LIST:-2,3,4,5}}"
```

或运行时指定：

```bash
GPU_LIST=0,1 ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic blasst
```

#### 修改样本数

编辑 `eval/RULER/scripts/config_tasks.sh`:

```bash
NUM_SAMPLES=1  # 每个任务的样本数
```

## 批量测试脚本示例

```bash
#!/bin/bash
# 批量测试不同 lambda 值

MODEL="llama3.1-8b-nanovllm"
BENCHMARK="synthetic"
METRIC="blasst"

for LAMBDA in 0.3 0.5 0.7; do
    echo "Testing BLASST with lambda=$LAMBDA"
    BLASST_LAMBDA=$LAMBDA ./scripts/run_ruler.sh $MODEL $BENCHMARK $METRIC
done
```

## 故障排查

### GPU 内存不足

- 确保 `NANOVLLM_CPU_OFFLOAD=true` (默认启用)
- 减少 `NUM_GPU_BLOCKS` (默认 2)

### 模型找不到

检查 `MODEL_DIR` 环境变量：

```bash
export MODEL_DIR=/path/to/models
```

### 任务未运行

检查 `eval/RULER/scripts/config_tasks.sh` 中的任务列表是否已启用。

## 相关文件

| 文件 | 用途 |
|------|------|
| `scripts/run_ruler.sh` | 主入口脚本 |
| `eval/RULER/scripts/run.sh` | RULER 执行脚本 (含文件夹命名逻辑) |
| `eval/RULER/scripts/pred/call_api.py` | API 调用 (含 BLASST 参数传递) |
| `eval/RULER/scripts/config_tasks.sh` | 任务配置 (样本数、任务列表) |
| `eval/RULER/scripts/config_models.sh` | 模型配置 (序列长度) |

## 参考

- BLASST 实现: `3rdparty/nanovllm/nanovllm/kvcache/sparse/blasst.py`
- nanovllm 配置: `3rdparty/nanovllm/nanovllm/config.py`
