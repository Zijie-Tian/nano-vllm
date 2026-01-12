# RULER Migration Findings

## 调用链分析

### 1. 顶层入口分析

```
scripts/run_ruler_tasks.sh
    ├── scripts/run_ruler_docker.sh (Docker 模式，使用 HF/xattn)
    │   └── docker run ... bash -c "cd eval/RULER/scripts && ./run.sh ..."
    │
    └── scripts/run_ruler_nanovllm.sh (NanoVLLM 模式)
        └── docker run ... bash -c "cd eval/RULER/scripts && ./run.sh ..."
```

### 2. RULER run.sh 调用链

```
eval/RULER/scripts/run.sh
    │
    ├── source config_models.sh     # 加载模型配置
    │   └── MODEL_SELECT() 函数定义模型路径和框架
    │
    ├── source config_tasks.sh      # 加载任务配置
    │   └── synthetic=() 数组定义任务列表
    │
    ├── 循环 SEQ_LENGTHS:
    │   │
    │   ├── python data/prepare.py  # 数据准备
    │   │   ├── imports: template.py, tokenizer.py
    │   │   ├── imports: synthetic/*.py (niah.py, qa.py, etc.)
    │   │   └── imports: synthetic/constants.py
    │   │
    │   ├── python pred/call_api.py # 模型推理
    │   │   ├── from xattn.src.load_llama import FastPrefillConfig  ← 关键依赖
    │   │   ├── imports: model_wrappers.py
    │   │   │   └── from xattn.src.load_llama import load_model, FastPrefillConfig
    │   │   └── imports: client_wrappers.py (API clients)
    │   │
    │   └── python eval/evaluate.py # 结果评估
    │       └── imports: eval/synthetic/constants.py
```

### 3. xattn 包内部依赖

```
xattn/src/load_llama.py
    ├── from xattn.threshold.llama_threshold import llama_fuse_16, llama_fuse_8, llama_fuse_4
    ├── import flashinfer  ← 外部依赖 (必需)
    │
    ├── from xattn.src.Xattention import Xattention_prefill (try/except)
    ├── from xattn.src.Minference import Minference_prefill (try/except)
    ├── from xattn.src.Fullprefill import Full_prefill (try/except)
    ├── from xattn.src.Flexprefill import Flexprefill_prefill (try/except)
    ├── from xattn.src.Compass import Compass_prefill (try/except)
    ├── from xattn.src.AvgPool import AvgPool_prefill (try/except)
    │
    └── from xattn.src.utils import *
```

```
xattn/src/AvgPool.py
    └── from block_sparse_attn import block_sparse_attn_func  ← 外部依赖 (必需)

xattn/src/Fullprefill.py
    └── import flashinfer  ← 外部依赖 (必需)
```

---

## 文件依赖矩阵

### RULER Python 文件依赖

| 文件 | 依赖的模块 | xattn 依赖 | 外部依赖 |
|------|-----------|------------|----------|
| `pred/call_api.py` | model_wrappers, yaml, tqdm, nemo | FastPrefillConfig | torch |
| `pred/model_wrappers.py` | - | load_model, FastPrefillConfig | torch, transformers, nanovllm |
| `pred/client_wrappers.py` | - | 无 | requests, openai, google-generativeai |
| `data/prepare.py` | template, tokenizer, synthetic/* | 无 | nltk, yaml |
| `data/tokenizer.py` | - | 无 | transformers, tiktoken, nemo |
| `data/synthetic/niah.py` | tokenizer | 无 | wonderwords, nltk, numpy, nemo |
| `eval/evaluate.py` | synthetic/constants | 无 | pandas, nltk, nemo |

### xattn 内部文件依赖

| 文件 | 内部依赖 | 外部依赖 |
|------|----------|----------|
| `load_llama.py` | threshold/*, src/* | flashinfer, transformers |
| `AvgPool.py` | - | block_sparse_attn, torch |
| `Fullprefill.py` | - | flashinfer, torch |
| `Xattention.py` | 待分析 | 待分析 |
| `Compass.py` | 待分析 | 待分析 |
| `Flexprefill.py` | 待分析 | 待分析 |
| `Minference.py` | 待分析 | 待分析 |
| `utils.py` | 待分析 | 待分析 |
| `kernels.py` | 待分析 | 待分析 |

---

## 关键外部依赖

### 必需依赖

1. **flashinfer**
   - 用途：Full_prefill 和 decode 阶段的高效 attention 计算
   - 使用位置：`Fullprefill.py`, `load_llama.py`

2. **block_sparse_attn**
   - 用途：AvgPool 的 block sparse attention 实现
   - 使用位置：`AvgPool.py`
   - 注意：这是一个特殊依赖，可能需要单独安装或编译

3. **nemo-toolkit**
   - 用途：manifest_utils (JSONL 读写), SentencePieceTokenizer
   - 使用位置：`call_api.py`, `evaluate.py`, `niah.py`, `tokenizer.py`

4. **transformers**
   - 用途：模型加载和 tokenizer
   - 使用位置：`model_wrappers.py`, `load_llama.py`, `tokenizer.py`

### 可选依赖

5. **nano-vllm**
   - 用途：NanoVLLM 推理引擎 (CPU offload 支持)
   - 使用位置：`model_wrappers.py` (NanoVLLMModel class)

---

## Import 修改清单

### 需要从 `xattn.` 改为 `compass.` 的文件

1. **eval/RULER/scripts/pred/call_api.py**
   ```python
   # 旧
   from xattn.src.load_llama import FastPrefillConfig
   # 新
   from compass.src.load_llama import FastPrefillConfig
   ```

2. **eval/RULER/scripts/pred/model_wrappers.py**
   ```python
   # 旧
   from xattn.src.load_llama import load_model, FastPrefillConfig
   # 新
   from compass.src.load_llama import load_model, FastPrefillConfig
   ```

3. **compass/src/load_llama.py** (迁移后)
   ```python
   # 旧
   from xattn.threshold.llama_threshold import llama_fuse_16, llama_fuse_8, llama_fuse_4
   from xattn.src.Xattention import Xattention_prefill
   from xattn.src.Minference import Minference_prefill
   from xattn.src.Fullprefill import Full_prefill
   from xattn.src.Flexprefill import Flexprefill_prefill
   from xattn.src.Compass import Compass_prefill
   from xattn.src.AvgPool import AvgPool_prefill
   from xattn.src.utils import *

   # 新
   from compass.threshold.llama_threshold import llama_fuse_16, llama_fuse_8, llama_fuse_4
   from compass.src.Xattention import Xattention_prefill
   from compass.src.Minference import Minference_prefill
   from compass.src.Fullprefill import Full_prefill
   from compass.src.Flexprefill import Flexprefill_prefill
   from compass.src.Compass import Compass_prefill
   from compass.src.AvgPool import AvgPool_prefill
   from compass.src.utils import *
   ```

---

## Docker 脚本修改点

### run_ruler_docker.sh

1. 修改 PROJECT_DIR mount:
   ```bash
   # 旧
   -v $PROJECT_DIR:/workspace/x-attention
   # 新
   -v $PROJECT_DIR:/workspace/compass
   ```

2. 修改 PYTHONPATH:
   ```bash
   # 旧
   -e PYTHONPATH=/workspace/x-attention:/workspace/nano-vllm
   # 新
   -e PYTHONPATH=/workspace/compass:/workspace/nano-vllm
   ```

3. 修改工作目录:
   ```bash
   # 旧
   -w /workspace/x-attention
   # 新
   -w /workspace/compass
   ```

### run_ruler_nanovllm.sh

同上修改。

---

## 数据文件清单

### 必需数据文件
- `eval/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json` - NIAH essay haystack

### 可选数据文件 (QA 任务)
- `eval/RULER/scripts/data/synthetic/json/hotpotqa.json`
- `eval/RULER/scripts/data/synthetic/json/squad.json`

### 下载脚本
- `eval/RULER/scripts/data/synthetic/json/download_qa_dataset.sh`
- `eval/RULER/scripts/data/synthetic/json/download_paulgraham_essay.py`

---

## 模型配置 (config_models.sh)

### 支持的模型

| 模型名称 | 模型路径 | 模板类型 | 框架 |
|---------|---------|---------|------|
| `llama3.1-8b-chat` | `${MODEL_DIR}/Llama-3.1-8B-Instruct` | meta-llama3 | hf |
| `qwen3-4b-nanovllm` | `${MODEL_DIR}/Qwen3-4B-Instruct-2507` | qwen | nanovllm |
| `llama3.1-8b-nanovllm` | `${MODEL_DIR}/Llama-3.1-8B-Instruct` | meta-llama3 | nanovllm |

---

## 任务配置 (config_tasks.sh)

### 默认参数
- `NUM_SAMPLES=5`
- `SEQ_LENGTHS=(32768)` - 可配置多个长度

### 可用任务 (synthetic.yaml)
| 任务 | 基础任务 | 描述 |
|------|---------|------|
| niah_single_1 | niah | Needle-in-a-haystack (repeat haystack) |
| niah_single_2 | niah | NIAH (essay haystack) |
| niah_single_3 | niah | NIAH (UUID values) |
| niah_multikey_* | niah | 多 key 变体 |
| niah_multivalue | niah | 多 value 变体 |
| niah_multiquery | niah | 多 query 变体 |
| vt | variable_tracking | 变量追踪 |
| cwe | common_words_extraction | 常见词提取 |
| fwe | freq_words_extraction | 频率词提取 |
| qa_1 | qa | SQuAD QA |
| qa_2 | qa | HotpotQA |
