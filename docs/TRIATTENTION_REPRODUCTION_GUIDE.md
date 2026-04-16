# TriAttention Reproduction Guide

本文档说明当前 COMPASS 工程里 **TriAttention 的复现逻辑**、**当前工程实现和标准实现的关系**、以及 **standard 对照脚本的使用方法**。

---

## 1. 目标

当前工程里 TriAttention 的目标不是只“跑通一个算法名字”，而是做到：

1. 在 COMPASS 工程内部拥有一套独立的 TriAttention 纯 torch 实现；
2. 能通过 LongBench torch 路径直接触发；
3. 能和标准 TriAttention 实现进行逐样本对齐比较；
4. 在给定模型（例如 `Qwen3-8B`）上，对齐 standard 实现的输出与分数。

因此，这套复现逻辑天然分成两条线：

- **COMPASS 内部实现路径**
- **standard 对照路径**

---

## 2. 代码结构总览

### 2.1 COMPASS 内部 TriAttention 实现

当前工程内新增/使用的核心文件：

| 文件 | 作用 |
|------|------|
| `compass/src/TriAttention.py` | 当前工程内独立实现的 TriAttention 核心逻辑 |
| `compass/src/triattention_utils.py` | RoPE 反解、频域转换、打分相关工具函数 |
| `compass/src/triattention_stats_utils.py` | stats metadata 校验 |
| `scripts/calibrate_triattention.py` | 生成当前工程 TriAttention 使用的 calibration stats |

### 2.2 LongBench torch 接入路径

| 文件 | 作用 |
|------|------|
| `eval/LongBench/scripts/config_models.sh` | 定义 TriAttention 模型 alias（如 `qwen3-8b-triattention`） |
| `eval/LongBench/scripts/run.sh` | LongBench 底层 runner，负责解析 alias、参数、输出目录 |
| `eval/LongBench/scripts/pred.py` | LongBench 推理入口 |
| `eval/LongBench/scripts/model_wrappers.py` | `TorchModel` 内部真正构造模型并应用 `compass.src.TriAttention` |
| `scripts/run_longbench.sh` | 顶层用户入口 |

### 2.3 standard 对照路径

| 文件 | 作用 |
|------|------|
| `tests/test_triattention_standard_longbench.py` | 使用上游 standard TriAttention 实现做 LongBench 对照运行 |

---

## 3. 当前工程里的 TriAttention 是怎么被触发的？

### 3.1 最推荐的入口：模型 alias

例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
TRIATTENTION_STATS_PATH=/path/to/qwen3_8b.pt \
./scripts/run_longbench.sh \
  qwen3-8b-triattention triattention torch \
  --data-root ~/data/LongBench \
  --num-samples 2
```

这个命令会触发下面的链路：

```text
scripts/run_longbench.sh
  -> eval/LongBench/scripts/run.sh
    -> eval/LongBench/scripts/config_models.sh
      -> 解析 qwen3-8b-triattention alias
    -> eval/LongBench/scripts/pred.py
      -> eval/LongBench/scripts/model_wrappers.py::TorchModel
        -> compass.src.TriAttention.apply_triattention_patch(...)
```

### 3.2 具体 alias 负责什么？

`eval/LongBench/scripts/config_models.sh` 里的：

- `qwen3-8b`
- `qwen3-8b-triattention`

会统一决定：

- `model_path`
- `model_name`
- `template_type`
- `max_model_len`
- 是否启用：
  - `compression_method=triattention`
- 是否从环境变量读取：
  - `TRIATTENTION_STATS_PATH`
  - `TRIATTENTION_BUDGET`

这一步很重要，因为它保证：

> LongBench 跑 COMPASS TriAttention 时，不会因为漏传 `max_model_len` 或 stats 路径而导致和 standard 实现不一致。

---

## 4. 当前工程里的 TriAttention torch 路径怎么工作？

### 4.1 `TorchModel` 的分支逻辑

在 `eval/LongBench/scripts/model_wrappers.py` 里：

- 如果 `compression_method != triattention`
  - 走普通 `AutoModelForCausalLM` 路径
- 如果 `compression_method == triattention`
  - 直接加载模型
  - 再调用：

```python
compass.src.TriAttention.apply_triattention_patch(...)
```

这意味着当前 COMPASS 的 TriAttention 逻辑并不是塞进普通 prefill metric 里，而是：

> 作为一个独立的 **post-forward cache compression** 路径存在。

这点和 standard 实现是一致的。

---

### 4.2 为什么不是继续走旧的 `load_llama.py + forward_eval` 语义？

因为 TriAttention 和当前 `metric=xattn/compass/full/...` 那些逻辑职责不同。

当前 `load_llama.py` 中原本的逻辑主要是：

- 改 prefill attention 计算
- 改 block sparse attention 路径

而 TriAttention 的本体是：

- patch `model.forward`
- 在 forward 后根据 `past_key_values` 做 KV 压缩

所以如果一边：
- 套 `forward_eval`

一边又：
- 套 `apply_triattention_patch`

会改变标准 TriAttention 的语义。

因此当前复现逻辑里，TriAttention 模式的原则是：

> **不要再额外套 COMPASS 原有 prefill attention patch，直接走标准 TriAttention 语义。**

---

## 5. standard 对照脚本是怎么工作的？

standard 路径在：

```bash
tests/test_triattention_standard_longbench.py
```

它的目标是：

> 用上游 standard TriAttention 实现，在相同 LongBench prompt / 相同模型 / 相同 stats / 相同 generation 设置下，生成一个“标准参考输出”。

### 5.1 它的逻辑

它会：

1. 从 `LongBench` 数据里读取指定子集
2. 使用当前 COMPASS 里的：
   - `dataset2prompt`
   - `dataset2maxlen`
3. 对 prompt 做同样的 truncation
4. 用：

```python
from triattention.methods.triattention import apply_triattention_patch
```

对模型打 patch
5. 调用标准 `model.generate(...)`
6. 将结果写到：

```bash
eval/LongBench/benchmark_root/pred/<model-name>/
```

### 5.2 为什么这个脚本放在 `tests/`？

因为它的职责是：

- 作为标准实现的**可重复对照基线**
- 方便未来继续做：
  - regression check
  - exact match 对比
  - score 对齐验证

也就是说它不是日常 benchmark runner，而是：

> **标准参考实现的验证脚本**

---

## 6. standard 脚本怎么用？

### 6.1 跑指定 16 个 triattention 子集，2 sample，对照用

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
python tests/test_triattention_standard_longbench.py \
  --model-path /home/zijie/models/Qwen3-8B \
  --stats-path /home/zijie/Code/triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
  --data-root ~/data/LongBench \
  --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_retrieval_en,passage_count,lcc,repobench-p \
  --num-samples 2 \
  --model-name Qwen3-8B-standard-triattention
```

### 6.2 对结果打分

```bash
python eval/LongBench/scripts/eval.py \
  --model-name Qwen3-8B-standard-triattention \
  --output-root /home/zijie/Code/COMPASS/eval/LongBench/benchmark_root \
  --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_retrieval_en,passage_count,lcc,repobench-p
```

---

## 7. 当前 COMPASS 路径怎么用？

### 7.1 推荐命令

```bash
CUDA_VISIBLE_DEVICES=0 \
TRIATTENTION_STATS_PATH=/home/zijie/Code/triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
./scripts/run_longbench.sh \
  qwen3-8b-triattention triattention torch \
  --data-root ~/data/LongBench \
  --num-samples 2
```

这个命令的好处是：

- `qwen3-8b-triattention` alias 已经包含：
  - 正确模型路径
  - `max_model_len=16384`
  - `compression_method=triattention`
- 避免你手动漏传关键参数

### 7.2 也可以手动显式传参

```bash
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_longbench.sh \
  /home/zijie/models/Qwen3-8B triattention torch \
  --data-root ~/data/LongBench \
  --num-samples 2 \
  --compression-method triattention \
  --triattention-stats-path /home/zijie/Code/triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
  --triattention-budget 2048 \
  --max-model-len 16384
```

但这种方式更容易漏参数，所以一般不推荐。

---

## 8. 为什么之前会出现和 standard 不一致？

这次对齐过程中，关键发现有这些：

### 8.1 Qwen3 一开始走错了加载路径
之前 COMPASS 路径里在某些情况下会把：

- `Qwen3-8B`

错误地放进偏 Llama 的加载逻辑里，导致 warning：

```text
You are using a model of type qwen3 to instantiate a model of type llama
```

这个问题会直接破坏对齐。

### 8.2 直接用裸模型路径时，容易漏掉 `max_model_len`
如果不用 `qwen3-8b-triattention` alias，而是直接传模型路径：

```bash
/home/zijie/models/Qwen3-8B
```

LongBench runner 默认未必知道应该给它：

```bash
max_model_len = 16384
```

这会导致：

- prompt truncation 行为和 standard 实现不一致

而一旦 prompt truncation 不一致，后面整条生成链路都不会一致。

### 8.3 TriAttention 模式不能再叠加旧的 prefill patch 语义
如果一边用：

- `forward_eval`

一边又用：

- `apply_triattention_patch`

那就已经不是 standard TriAttention 的语义了。

所以当前复现逻辑明确改成：

> TriAttention 模式下，只保留标准 TriAttention 的 forward patch / KV compression 语义。

---

## 9. 当前“复现逻辑”的正确理解

你现在工程里的 TriAttention 复现逻辑，其实是下面这套：

### 9.1 离线阶段
通过：

```bash
scripts/calibrate_triattention.py
```

生成模型专属 stats 文件。

### 9.2 在线阶段：COMPASS 版本
通过：

```bash
run_longbench.sh -> run.sh -> pred.py -> model_wrappers.TorchModel -> compass.src.TriAttention
```

跑当前工程内的 TriAttention。

### 9.3 在线阶段：standard 版本
通过：

```bash
tests/test_triattention_standard_longbench.py
```

直接跑上游 standard TriAttention。

### 9.4 对齐判定
比较：

- `result.json`
- 每个 `<dataset>.jsonl` 的逐样本输出

如果：
- score 全一致
且
- 每条输出都 exact match

那就认为：

> 当前 COMPASS 里的 TriAttention 对 standard 实现复现成功。

---

## 10. 当前已经验证到什么程度？

在下面条件下：

- 模型：`Qwen3-8B`
- backend：`torch`
- GPU：`CUDA_VISIBLE_DEVICES=0`
- 子集：`triattention`
- 每个子集：`2 sample`

当前工程里的 COMPASS TriAttention 和 standard 实现已经做到了：

- **分数完全一致**
- **逐样本输出完全一致**

这说明当前这条复现链路已经对齐。

---

## 11. 实际使用时的注意事项

### 11.1 优先用 alias，不要裸传模型路径
推荐：

```bash
qwen3-8b-triattention
```

不要总是手动传裸路径，否则容易漏掉：

- `max_model_len`
- `stats path`
- `compression_method`

### 11.2 stats 文件必须和模型匹配
例如：

- `Qwen3-8B` 要用自己的 stats
- 不能把 Llama stats 拿来直接给 Qwen 用

### 11.3 只对 torch 路径完成了严格对齐
当前这套说明和对齐结论只针对：

- **torch backend**

不是：
- nanovllm
- vLLM plugin

### 11.4 对比时必须固定随机性和 prompt 处理
要想比较严格一致，必须同时固定：

- seed
- prompt template
- truncation 逻辑
- generation kwargs
- stats 文件
- model path

否则就很容易出现“看起来算法不一致，其实只是前处理不一致”。

---

## 12. 推荐的最小复现命令

### 当前工程内 TriAttention

```bash
CUDA_VISIBLE_DEVICES=0 \
TRIATTENTION_STATS_PATH=/home/zijie/Code/triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
./scripts/run_longbench.sh \
  qwen3-8b-triattention triattention torch \
  --data-root ~/data/LongBench \
  --num-samples 2
```

### standard 对照

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
python tests/test_triattention_standard_longbench.py \
  --model-path /home/zijie/models/Qwen3-8B \
  --stats-path /home/zijie/Code/triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
  --data-root ~/data/LongBench \
  --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_retrieval_en,passage_count,lcc,repobench-p \
  --num-samples 2 \
  --model-name Qwen3-8B-standard-triattention
```

---

## 13. 总结

这份文档最核心的结论就是：

> 当前 COMPASS 里的 TriAttention 复现，不是简单“加了个算法名字”，而是通过  
> **本地 TriAttention 实现 + LongBench torch 接入 + standard 对照脚本**  
> 建立了一条可验证、可对齐、可复现的闭环。

而 `tests/test_triattention_standard_longbench.py` 的作用，就是：

> 提供一个长期稳定的 standard reference，用来持续检查当前工程里的 TriAttention 是否还和标准实现保持一致。*** Update File: /home/zijie/Code/COMPASS/AGENTS.md
