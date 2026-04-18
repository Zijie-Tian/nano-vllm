# LongBench Standard Runner Guide

本文档说明 `tests/test_longbench_standard_runner.py` 的定位、参数、输入输出格式，以及推荐使用方式。

这个脚本虽然放在 `tests/` 下，但它的职责不是单元测试，而是：

- 使用 **upstream LongBench prompt / max generation config / metrics**；
- 使用本地 LongBench 数据；
- 走一条不依赖 `eval/LongBench/scripts/run.sh` / `pred.py` 的 **独立 runner 路径**；
- 为当前工程提供一个更接近 upstream 行为的对照基线。

---

## 1. 脚本定位

### 1.1 它解决什么问题？

当前工程里已经有一套 LongBench 主路径：

- `scripts/run_longbench.sh`
- `eval/LongBench/scripts/run.sh`
- `eval/LongBench/scripts/pred.py`
- `eval/LongBench/scripts/model_wrappers.py`

这套主路径适合日常 benchmark、模型 alias 管理，以及接入 COMPASS 自己的压缩/patch 逻辑。

而 `tests/test_longbench_standard_runner.py` 的目标不同：

> **尽量直接地复用 upstream LongBench 的 prompt / max_gen / metrics 设定，跑一条更“标准”的 HuggingFace 生成路径。**

因此它适合做：

- baseline 对照；
- 与当前 COMPASS LongBench 主路径做结果比较；
- 排查“到底是模型行为差异，还是 runner / wrapper 差异”这类问题；
- 在不引入 COMPASS 特定 patch 的前提下做快速复现。

### 1.2 它不做什么？

它**不会**：

- 解析 `config_models.sh` 里的模型 alias；
- 自动接入 `compression_method` / TriAttention / XAttention / COMPASS patch；
- 走 `eval/LongBench/scripts/eval.py` 的后处理流程；
- 复用 `run_longbench.sh` 的输出目录命名逻辑。

也就是说，这个脚本默认就是一条：

> **upstream 配置 + 本地数据 + HuggingFace `AutoModelForCausalLM.generate()`**

的独立路径。

---

## 2. 依赖与代码位置

### 2.1 脚本路径

```bash
tests/test_longbench_standard_runner.py
```

### 2.2 它依赖的配置/资源

| 路径 | 用途 |
|------|------|
| `eval/LongBench/upstream/LongBench/config/dataset2prompt.json` | 每个数据集的 prompt 模板 |
| `eval/LongBench/upstream/LongBench/config/dataset2maxlen.json` | 每个数据集的 `max_new_tokens` |
| `eval/LongBench/upstream/LongBench/metrics.py` | 优先使用 upstream metrics |
| `eval/LongBench/scripts/fallback_metrics.py` | 当 upstream metrics 依赖缺失时的回退实现 |

### 2.3 当前脚本的路径假设

脚本内部当前写死了：

```python
REPO_ROOT = Path("/mnt/data/tzj/Code/COMPASS")
```

因此如果仓库不在这个路径下，脚本将找不到 `eval/LongBench/upstream/LongBench` 等依赖目录。

如果你把仓库移动到别的位置，运行前要先把这个常量改掉，或者把脚本改成基于 `__file__` 自动推导仓库根目录。

---

## 3. 输入数据要求

### 3.1 默认数据目录

默认：

```bash
~/data/LongBench
```

可以通过 `--data-root` 覆盖。

### 3.2 支持的数据布局

脚本按下面顺序寻找本地数据：

1. `<data-root>/data/<dataset>.jsonl`
2. `<data-root>/<dataset>.jsonl`
3. `<data-root>/data.zip` 中的 `data/<dataset>.jsonl`
4. `<data-root>/LongBench/data.zip` 中的 `data/<dataset>.jsonl`

因此下面几种布局都可以：

```bash
~/data/LongBench/data/narrativeqa.jsonl
~/data/LongBench/narrativeqa.jsonl
~/data/LongBench/data.zip
~/data/LongBench/LongBench/data.zip
```

### 3.3 `all` 包含哪些数据集？

`--datasets all` 会跑全部 21 个 LongBench 子集：

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `multifieldqa_zh`
- `hotpotqa`
- `2wikimqa`
- `musique`
- `dureader`
- `gov_report`
- `qmsum`
- `multi_news`
- `vcsum`
- `trec`
- `triviaqa`
- `samsum`
- `lsht`
- `passage_count`
- `passage_retrieval_en`
- `passage_retrieval_zh`
- `lcc`
- `repobench-p`

如果只想跑部分数据集，可以用逗号分隔：

```bash
--datasets narrativeqa,trec,passage_count
```

---

## 4. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | 无 | **必填**。传 HuggingFace 模型目录或模型名。 |
| `--model-name` | `basename(model_path)` | 输出目录名；不传时自动取模型路径最后一级。 |
| `--device` | `None` | 显式设备，如 `cuda:0` / `cpu`。不传时使用 `device_map=auto`。 |
| `--data-root` | `~/data/LongBench` | LongBench 本地数据根目录。 |
| `--output-root` | `eval/LongBench/benchmark_root/standard_runner` | 输出根目录。 |
| `--datasets` | `all` | 要跑的数据集列表。 |
| `--num-samples` | `100` | 每个数据集最多读取多少条样本。 |
| `--max-model-len` | `None` | 手动覆盖模型上下文长度。 |
| `--dtype` | `bfloat16` | 传给 `torch.<dtype>` 的 dtype 名称。 |
| `--seed` | `42` | Python / NumPy / Torch 随机种子。 |

### 4.1 `--device` 的行为

- 传 `--device cuda:0` 时：
  - 脚本会使用 `device_map={"": "cuda:0"}`
  - 模型整体放到指定设备
- 不传 `--device` 时：
  - 脚本会使用 `device_map="auto"`
  - 由 Transformers / Accelerate 自己决定如何放置模型

对于 GPU 运行，仍建议显式写：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID>
```

以符合当前仓库约定。

### 4.2 `--max-model-len` 的自动推断逻辑

如果没有手动传 `--max-model-len`，脚本会依次尝试：

1. 读取 `<model-path>/config.json` 中的：
   - `max_position_embeddings`
   - `seq_length`
   - `model_max_length`
2. 读取 `tokenizer.model_max_length`
3. 如果都拿不到，回退到：

```python
32768
```

### 4.3 prompt 截断逻辑

如果某条样本的 prompt 长度超过 `max_model_len`，脚本不会简单地只保留前半段或后半段，而是：

- 取前半段 token；
- 再取后半段 token；
- 拼接成新的 prompt。

也就是所谓的“头尾拼接”截断逻辑：

```text
left half + right half
```

这和 LongBench 常见的长上下文截断做法一致。

---

## 5. 运行流程

脚本主流程如下：

1. 解析命令行参数；
2. 固定随机种子；
3. 加载 upstream 的 `dataset2prompt.json` 和 `dataset2maxlen.json`；
4. 尝试加载 upstream `metrics.py`，失败时回退到 `fallback_metrics.py`；
5. 用 HuggingFace `AutoTokenizer` / `AutoModelForCausalLM` 加载模型；
6. 自动推断 `max_model_len`；
7. 逐个数据集读取样本、格式化 prompt、截断、生成、评分；
8. 将每个数据集的逐样本输出写入 jsonl；
9. 将最终分数写入 `result.json`。

### 5.1 聊天模板处理

大多数数据集会先经过 `build_chat()` 包装。

但以下数据集不会额外包 chat 模板：

- `trec`
- `triviaqa`
- `samsum`
- `lsht`
- `lcc`
- `repobench-p`

这和脚本中的 `NO_CHAT_WRAP_DATASETS` 常量保持一致。

### 5.2 `samsum` 的特殊终止逻辑

`samsum` 会额外把换行 token 也当作停止条件之一，避免生成过长摘要：

- `eos_token_id`
- `newline_token[-1]`

其他数据集则使用普通的 deterministic generation：

- `num_beams=1`
- `do_sample=False`
- `temperature=1.0`

---

## 6. 输出目录与结果格式

### 6.1 默认输出目录

```bash
eval/LongBench/benchmark_root/standard_runner
```

实际结果路径是：

```bash
eval/LongBench/benchmark_root/standard_runner/pred/<model-name>/
```

### 6.2 输出文件

每个数据集会生成：

```bash
<dataset>.jsonl
```

每行包含：

- `pred`
- `answers`
- `all_classes`
- `length`

同时还会生成：

```bash
result.json
```

它直接保存这次运行涉及的数据集分数，例如：

```json
{
  "narrativeqa": 22.31,
  "trec": 63.5
}
```

### 6.3 覆盖行为

如果某个数据集的输出 jsonl 已存在，脚本会先删掉旧文件再重写：

```python
if out_path.exists():
    out_path.unlink()
```

因此同一 `output-root + model-name + dataset` 的重复运行会覆盖旧结果，而不是追加。

---

## 7. 推荐命令

### 7.1 最小 smoke test

只跑一个 GPU、一个数据集、一个样本：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
python tests/test_longbench_standard_runner.py \
  --model-path /home/zijie/models/Qwen3-8B \
  --model-name Qwen3-8B-standard-runner \
  --device cuda:0 \
  --data-root ~/data/LongBench \
  --datasets narrativeqa \
  --num-samples 1
```

### 7.2 跑一组英文 LongBench 子集

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
python tests/test_longbench_standard_runner.py \
  --model-path /home/zijie/models/Qwen3-8B \
  --model-name Qwen3-8B-standard-runner \
  --device cuda:0 \
  --data-root ~/data/LongBench \
  --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_retrieval_en,passage_count,lcc,repobench-p \
  --num-samples 2 \
  --dtype bfloat16
```

### 7.3 自定义输出目录

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
python tests/test_longbench_standard_runner.py \
  --model-path /home/zijie/models/Qwen3-8B \
  --model-name Qwen3-8B-standard-runner \
  --device cuda:0 \
  --data-root ~/data/LongBench \
  --output-root /tmp/longbench-standard-runner \
  --datasets trec,passage_count \
  --num-samples 4
```

### 7.4 显式覆盖上下文长度

当模型 `config.json` 里的上下文长度不可靠时，可以手动指定：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
python tests/test_longbench_standard_runner.py \
  --model-path /home/zijie/models/Qwen3-8B \
  --model-name Qwen3-8B-standard-runner \
  --device cuda:0 \
  --data-root ~/data/LongBench \
  --datasets narrativeqa \
  --num-samples 1 \
  --max-model-len 32768
```

---

## 8. 和现有 LongBench 主路径的区别

| 维度 | `tests/test_longbench_standard_runner.py` | `scripts/run_longbench.sh` 主路径 |
|------|-------------------------------------------|----------------------------------|
| Prompt / `max_gen` 来源 | upstream `dataset2prompt` / `dataset2maxlen` | 本仓库 LongBench 封装 |
| 模型加载 | 直接 `AutoModelForCausalLM` | 可能经过 wrapper / alias / patch |
| 是否支持模型 alias | 否 | 是 |
| 是否支持 compression method | 否 | 是 |
| 结果打分 | 脚本内直接打分 | 通常走 `eval.py` |
| 默认输出目录 | `benchmark_root/standard_runner` | `benchmark_root` |
| 适用场景 | baseline / upstream-like 对照 | 日常 benchmark / 工程主路径 |

如果你要验证：

- “模型本身在 LongBench 上是什么行为”；
- “当前主路径和更标准的 HF 路径差了多少”；
- “某个回归是不是 runner 引入的”；

优先用这个 standard runner 更合适。

---

## 9. 注意事项

1. **文件名虽然在 `tests/` 下，但它本质是 CLI runner。**
2. **它当前依赖固定的仓库绝对路径。** 换目录后要先改 `REPO_ROOT`。
3. **`--dtype` 必须是 `torch` 上存在的 dtype 名称**，例如 `float16`、`bfloat16`、`float32`。
4. **它默认直接写 `result.json`，不需要再单独跑 `eval.py` 才能得到分数。**
5. **它不会自动接入 TriAttention / XAttention / COMPASS patch。** 如果你要测这些工程特性，应该走主 LongBench 路径。
