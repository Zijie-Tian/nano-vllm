# LongBench Benchmark Guide

本指南说明 COMPASS 中 LongBench 的命令入口、模型/子集配置方式，以及结果输出规则。

## 1. 入口文件

### 推荐入口
- 顶层脚本：`scripts/run_longbench.sh`

### 底层实现
- 运行封装：`eval/LongBench/scripts/run.sh`
- 模型配置：`eval/LongBench/scripts/config_models.sh`
- 子集配置：`eval/LongBench/scripts/config_tasks.sh`
- 推理：`eval/LongBench/scripts/pred.py`
- 评测：`eval/LongBench/scripts/eval.py`
- 上游基准资源：`eval/LongBench/upstream`
- 独立 standard runner 指南：`docs/LONG_BENCH_STANDARD_RUNNER_GUIDE.md`

## 2. 默认行为

- 默认输出目录：`eval/LongBench/benchmark_root`
- 顶层脚本默认 backend：`torch`
- 任何 GPU 运行都应显式指定：`CUDA_VISIBLE_DEVICES=<GPU_ID>`
- 直接使用 `run_longbench.sh` 时，默认数据目录可通过：
  - `--data-root <PATH>`
  - 或环境变量 `LONG_BENCH_DATA_ROOT`

## 3. 模型配置方法

`config_models.sh` 提供模型别名到真实路径/模板参数的映射。

当前内置别名：

| 别名 | 实际模型 |
|------|----------|
| `tiny-gpt2` | `sshleifer/tiny-gpt2` |
| `llama3.1-8b-instruct` | `${MODEL_DIR}/Llama-3.1-8B-Instruct` |
| `llama3.1-nemotron-8b-ultralong-1m-instruct` | `${MODEL_DIR}/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct` |

默认 `MODEL_DIR`：
```bash
/home/zijie/models
```

因此推荐直接用模型别名：
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh llama3.1-8b-instruct all torch
```

也兼容直接传模型路径：
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh /home/zijie/models/Llama-3.1-8B-Instruct all torch
```

## 4. 子集配置方法

`config_tasks.sh` 当前只保留两个命名 preset：

### `all`
表示完整 LongBench 全部 21 个子集。

### `triattention`
表示用户指定的 16 个英文/代码相关子集：

| 缩写 | LongBench dataset |
|------|-------------------|
| NarrQA | `narrativeqa` |
| Qasp | `qasper` |
| MFQA | `multifieldqa_en` |
| HpQA | `hotpotqa` |
| 2Wik | `2wikimqa` |
| Musi | `musique` |
| GovR | `gov_report` |
| QMSu | `qmsum` |
| MNew | `multi_news` |
| TREC | `trec` |
| TriQA | `triviaqa` |
| SSum | `samsum` |
| PaRe | `passage_retrieval_en` |
| PaCn | `passage_count` |
| LCC | `lcc` |
| ReBe | `repobench-p` |

> `Avg` 不是可运行子集，它是最终 `result.json` 中的统计概念，不需要单独配置。

除了两个 preset，也支持：
- 单个数据集名
- 逗号分隔的多个数据集名

例如：
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh llama3.1-8b-instruct trec torch

CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh \
  llama3.1-8b-instruct narrativeqa,trec,passage_count torch
```

## 5. 常用命令

### 5.1 跑完整 LongBench
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh \
  llama3.1-8b-instruct all torch \
  --data-root ~/data/LongBench
```

### 5.2 跑 triattention 子集
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh \
  llama3.1-8b-instruct triattention torch \
  --data-root ~/data/LongBench
```

### 5.3 每个子集只跑 1 个样本
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh \
  llama3.1-8b-instruct triattention torch \
  --data-root ~/data/LongBench \
  --num-samples 1
```

### 5.4 运行 LongBench-E
`--e` 只切换数据来源，不改变 preset 名称。

例如：
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_longbench.sh \
  llama3.1-8b-instruct all torch \
  --data-root ~/data/LongBench \
  --e
```

### 5.5 直接使用底层脚本
```bash
cd eval/LongBench/scripts
CUDA_VISIBLE_DEVICES=0 bash run.sh \
  llama3.1-8b-instruct triattention \
  --backend torch \
  --data-root ~/data/LongBench \
  --num-samples 1
```

## 6. 输出规则

默认输出目录：
```bash
eval/LongBench/benchmark_root
```

结果位置：
```bash
eval/LongBench/benchmark_root/pred/<MODEL_NAME>/
```

其中包括：
- `<dataset>.jsonl`：逐样本推理结果
- `result.json`：这次指定 datasets 的评测结果

## 7. 覆盖行为

重新运行同一个模型名 + 同一个输出目录时：
- 每个 `<dataset>.jsonl` 会先删除再重写
- 不会在旧文件后继续追加
- `eval.py` 只评估本次传入的 datasets，不会把输出目录里其他旧数据集的 jsonl 混进去

## 8. 数据目录要求

推荐将 LongBench 数据放在：
```bash
~/data/LongBench
```

当前支持以下本地布局之一：
- `~/data/LongBench/data.zip`
- `~/data/LongBench/data/<subset>.jsonl`
- 通过 `--data-root <PATH>` 指向同类结构

## 9. 备注

- 当前 `torch` 路径已经做过实际 smoke test。
- `nanovllm` 入口参数已保留，但需要单独做端到端验证时再跑。
