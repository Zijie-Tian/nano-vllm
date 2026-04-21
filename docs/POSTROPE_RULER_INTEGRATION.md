# POSTROPE RULER Benchmark Integration

POSTROPE 是 NanoVLLM 中显式的 post-RoPE baseline policy。本指南说明如何通过 COMPASS 的 `run_ruler.sh` 入口在 RULER benchmark 中运行 POSTROPE，并给出推荐的**配置驱动**测试方式。

## 快速开始

### 配置驱动启动（推荐）

先在配置文件里设置：

- `eval/RULER/scripts/config_models.sh`：控制 `SEQ_LENGTHS`
- `eval/RULER/scripts/config_tasks.sh`：控制 `NUM_SAMPLES` 与 `synthetic` 任务列表
- `eval/RULER/scripts/eval.sh`：保持 `synthetic` 任务列表与 `config_tasks.sh` 同步

然后直接启动：

```bash
conda run -n ruler bash -lc '
cd /home/zijie/Code/COMPASS &&
GPULIST=0,1,2,3,4,5 ./scripts/run_ruler.sh \
  llama3.1-8b-nanovllm synthetic postrope
'
```

### 临时 override 启动（仅临时调试）

如果只是一次性的 smoke/debug，也可以临时覆盖：

```bash
conda run -n ruler bash -lc '
cd /home/zijie/Code/COMPASS &&
GPULIST=0,1,2,3,4,5 ./scripts/run_ruler.sh \
  llama3.1-8b-nanovllm synthetic postrope \
  --seq_len 32768 \
  --num_samples 1 \
  --task niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery
'
```

> 但如果这组任务/长度要反复使用，优先改配置文件，不要长期依赖 `--task` / `--seq_len` / `--num_samples`。

## 代码路径

POSTROPE 在 COMPASS RULER 路径中的接线如下：

| 文件 | 作用 |
|------|------|
| `scripts/run_ruler.sh` | 顶层入口，接受 `postrope` metric |
| `eval/RULER/scripts/run.sh` | 负责并行调度、长度循环、任务循环 |
| `eval/RULER/scripts/pred/call_api.py` | 将 `postrope` 映射到 NanoVLLM 的 `POSTROPE` sparse policy |
| `eval/RULER/scripts/config_tasks.sh` | 配置样本数与任务列表 |
| `eval/RULER/scripts/eval.sh` | 评估侧任务列表，必须与 `config_tasks.sh` 同步 |
| `eval/RULER/scripts/config_models.sh` | 配置序列长度 |

当前映射关系为：

```python
'postrope' -> 'POSTROPE'
```

## 推荐配置方式

### 1. 32K / 8-task / 1-sample debug 配置

#### `eval/RULER/scripts/config_models.sh`

```bash
SEQ_LENGTHS=(
    # 4096
    # 8192
    # 16384
    32768
    # 65536
    # 131072
)
```

#### `eval/RULER/scripts/config_tasks.sh`

```bash
NUM_SAMPLES=1

synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    # "vt"
    # "cwe"
    # "fwe"
    # "qa_1"
    # "qa_2"
)
```

#### `eval/RULER/scripts/eval.sh`

```bash
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    # "vt"
    # "cwe"
    # "fwe"
    # "qa_1"
    # "qa_2"
)
```

#### 启动命令

```bash
conda run -n ruler bash -lc '
cd /home/zijie/Code/COMPASS &&
GPULIST=0,1,2,3,4,5 ./scripts/run_ruler.sh \
  llama3.1-8b-nanovllm synthetic postrope
'
```

### 2. 全量 synthetic / 100-sample / 128K 及以下

#### `eval/RULER/scripts/config_models.sh`

```bash
SEQ_LENGTHS=(
    4096
    8192
    16384
    32768
    65536
    131072
    # 262144
    # 524288
    # 786432
    # 1048576
)
```

#### `eval/RULER/scripts/config_tasks.sh`

```bash
NUM_SAMPLES=100

synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
```

#### `eval/RULER/scripts/eval.sh`

```bash
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
```

#### 启动命令

```bash
conda run -n ruler bash -lc '
cd /home/zijie/Code/COMPASS &&
GPULIST=0,1,2,3,4,5 ./scripts/run_ruler.sh \
  llama3.1-8b-nanovllm synthetic postrope
'
```

## GPU 调度行为

`run.sh` 在 NanoVLLM 模式下使用 GPU-locked scheduling：

- `GPULIST=0,1,2,3,4,5` 时，会先把前 6 个任务分发到 6 张卡
- 空闲 GPU 会自动领取剩余任务
- 不需要手工拆任务

## 经验验证

已验证的 debug 配置：

- model: `llama3.1-8b-nanovllm`
- metric: `postrope`
- GPUs: `0,1,2,3,4,5`
- lengths: `32768`
- tasks: 8 个 (`niah_*` debug 集)
- samples: `1`

结果：

| Task | Score | Nulls |
|------|------:|------:|
| `niah_single_1` | 100.0 | 0/1 |
| `niah_single_2` | 100.0 | 0/1 |
| `niah_single_3` | 100.0 | 0/1 |
| `niah_multikey_1` | 100.0 | 0/1 |
| `niah_multikey_2` | 100.0 | 0/1 |
| `niah_multikey_3` | 100.0 | 0/1 |
| `niah_multivalue` | 100.0 | 0/1 |
| `niah_multiquery` | 100.0 | 0/1 |

对应结果目录：

```bash
eval/RULER/scripts/benchmark_root/postrope_llama3.1-8b-nanovllm/synthetic/32768/pred/
```

其中：

- `summary.csv`：汇总分数
- `submission.csv`：submission 格式结果
- `*.jsonl`：每个 task 的预测文件

## 注意事项

### 1. `config_tasks.sh` 和 `eval.sh` 必须同步

如果只改了 `config_tasks.sh`，没有同步 `eval.sh`，评估阶段很容易出现任务集合不一致。

### 2. 部分任务 debug 时可能看到 missing-file 提示

即使只想跑 8 个 debug task，评估阶段仍可能看到：

- `Prediction file vt.jsonl is not found`
- `Prediction file qa_1.jsonl is not found`

这通常表示当前 benchmark 仍以 synthetic 全量任务视角进行检查，而不是 POSTROPE 推理失败。判断是否真正成功，优先看：

- `summary.csv`
- 已生成的 `pred/*.jsonl`
- 日志中的 `Sparse policy initialized: POSTROPE`

### 3. GPU 选择必须显式指定

运行 GPU 测试前应显式指定卡号，例如：

```bash
GPULIST=0,1,2,3,4,5 ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic postrope
```

## 相关文档

- [`docs/ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)
- [`docs/BLASST_RULER_INTEGRATION.md`](BLASST_RULER_INTEGRATION.md)
- [`docs/RULER_NANOVLLM_XATTN_INTEGRATION.md`](RULER_NANOVLLM_XATTN_INTEGRATION.md)
- [`docs/RULER_NANOVLLM_XATTN_PATHS_ANALYSIS.md`](RULER_NANOVLLM_XATTN_PATHS_ANALYSIS.md)
