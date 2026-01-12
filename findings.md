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
    ├── import flashinfer  ← 外部依赖
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
    └── from block_sparse_attn import block_sparse_attn_func  ← 外部依赖

xattn/src/Fullprefill.py
    └── import flashinfer  ← 外部依赖
```

---

## 文件依赖矩阵

### RULER Python 文件依赖

| 文件 | 依赖的模块 | xattn 依赖 | nemo 依赖 |
|------|-----------|------------|-----------|
| `pred/call_api.py` | model_wrappers, yaml, tqdm | FastPrefillConfig | manifest_utils |
| `pred/model_wrappers.py` | - | load_model, FastPrefillConfig | - |
| `pred/client_wrappers.py` | - | 无 | - |
| `data/prepare.py` | template, tokenizer, synthetic/* | 无 | - |
| `data/tokenizer.py` | - | 无 | SentencePieceTokenizer |
| `data/synthetic/niah.py` | tokenizer | 无 | manifest_utils |
| `eval/evaluate.py` | synthetic/constants | 无 | manifest_utils |

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

## 环境依赖说明

根据 **x-attention/task_plan.md** 的规划：

| 依赖 | 版本 | 安装方式 | 状态 |
|------|------|---------|------|
| torch | 2.9.1 | pip (cuda128) | ✅ 由 x-attention 管理 |
| transformers | 4.57.3 | pip | ✅ 由 x-attention 管理 |
| nemo-toolkit | 2.6.1 (base) | pip | ✅ 由 x-attention 管理 |
| flash-attn | fork | 源码编译 | ✅ 由 x-attention 管理 |
| flashinfer | fork | 源码编译 | ✅ 由 x-attention 管理 |

**重要**: NeMo 安装为 base only，以下模块**不可用**：
- `nemo.collections.asr` (ASR extra)
- `nemo.collections.nlp` (NLP extra)

---

## Import 修改清单

### 1. xattn → compass 修改

#### compass/src/load_llama.py (迁移后)
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

#### eval/RULER/scripts/pred/call_api.py
```python
# 旧
from xattn.src.load_llama import FastPrefillConfig
# 新
from compass.src.load_llama import FastPrefillConfig
```

#### eval/RULER/scripts/pred/model_wrappers.py
```python
# 旧
from xattn.src.load_llama import load_model, FastPrefillConfig
# 新
from compass.src.load_llama import load_model, FastPrefillConfig
```

### 2. NeMo ASR Extra → 本地实现修改

**背景**: NeMo 2.6.1 base 已安装，但 ASR extra (`nemo.collections.asr`) 不可用。
需要创建本地 `manifest_utils.py` 替代。

#### 涉及文件
- `eval/RULER/scripts/pred/call_api.py`
- `eval/RULER/scripts/eval/evaluate.py`
- `eval/RULER/scripts/data/synthetic/niah.py`

```python
# 旧 (NeMo ASR extra - 不可用)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

# 新 (本地实现)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.manifest_utils import read_manifest, write_manifest
```

#### eval/RULER/scripts/data/tokenizer.py

检查是否依赖 NeMo NLP extra：
```python
# 如果有旧的 NeMo NLP 依赖
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

# 替换为 HuggingFace 或 sentencepiece
from transformers import AutoTokenizer
# 或
import sentencepiece as spm
```

**注意**: 大部分 RULER 代码已使用 HuggingFace tokenizer，此项可能无需修改。

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
