# RULER Benchmark 代码迁移计划

## Goal
将 x-attention 项目中的 RULER 测试基准代码完整迁移到 COMPASS 项目中，包括所有相关脚本、Python 模块、算子依赖，并重新设定 import 路径。

**注意**:
- 环境配置（conda 环境、依赖安装、3rdparty submodules）由 `x-attention/task_plan.md` 规定
- **本项目完全不使用 Docker**，所有运行均基于 conda 环境

## Current Status: `planning`
- Start Date: 2026-01-13

---

## Phase 1: 创建项目目录结构 `pending`

### 目标
建立 COMPASS 项目的基础目录结构。

### 目录结构
```
COMPASS/
├── compass/                    # 主包 (原 xattn)
│   ├── __init__.py
│   ├── src/                    # 核心算子实现
│   │   ├── __init__.py
│   │   ├── load_llama.py
│   │   ├── AvgPool.py
│   │   ├── Fullprefill.py
│   │   ├── Xattention.py
│   │   ├── Flexprefill.py
│   │   ├── Minference.py
│   │   ├── Compass.py
│   │   ├── kernels.py
│   │   └── utils.py
│   └── threshold/              # 阈值配置
│       ├── __init__.py
│       ├── llama_threshold.py
│       └── profile_threshold/
├── eval/                       # 评估模块
│   └── RULER/                  # RULER benchmark
│       ├── scripts/
│       │   ├── run.sh
│       │   ├── eval.sh
│       │   ├── config_models.sh
│       │   ├── config_tasks.sh
│       │   ├── synthetic.yaml
│       │   ├── data/
│       │   ├── pred/
│       │   ├── eval/
│       │   └── utils/
│       └── requirements.txt
├── scripts/                    # 顶层运行脚本
│   └── run_ruler.sh
└── 3rdparty/                   # (由 x-attention task_plan 管理)
    ├── flash-attention/
    └── flashinfer/
```

### 任务
- [ ] 创建 `compass/` 包目录
- [ ] 创建 `compass/src/` 子目录
- [ ] 创建 `compass/threshold/` 子目录
- [ ] 创建 `eval/RULER/scripts/` 目录结构
- [ ] 创建 `scripts/` 顶层脚本目录

---

## Phase 2: 迁移核心算子模块 (xattn → compass) `pending`

### 目标
将 xattn 包完整迁移为 compass 包，修改所有内部引用。

### 文件迁移列表

| 源路径 (x-attention/xattn/) | 目标路径 (COMPASS/compass/) |
|----------------------------|---------------------------|
| `__init__.py` | `__init__.py` |
| `src/__init__.py` | `src/__init__.py` |
| `src/load_llama.py` | `src/load_llama.py` |
| `src/AvgPool.py` | `src/AvgPool.py` |
| `src/Fullprefill.py` | `src/Fullprefill.py` |
| `src/Xattention.py` | `src/Xattention.py` |
| `src/Flexprefill.py` | `src/Flexprefill.py` |
| `src/Minference.py` | `src/Minference.py` |
| `src/Compass.py` | `src/Compass.py` |
| `src/kernels.py` | `src/kernels.py` |
| `src/utils.py` | `src/utils.py` |
| `threshold/__init__.py` | `threshold/__init__.py` |
| `threshold/llama_threshold.py` | `threshold/llama_threshold.py` |
| `threshold/profile_threshold/` | `threshold/profile_threshold/` |

### Import 修改 (`xattn.` → `compass.`)

**compass/src/load_llama.py**:
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

## Phase 3: 迁移 RULER Benchmark 模块 `pending`

### 目标
完整迁移 RULER 测试框架到 `eval/RULER/` 目录。

### 3.1 Shell 脚本迁移

| 源路径 (x-attention/eval/RULER/scripts/) | 目标路径 |
|----------------------------------------|----------|
| `run.sh` | `eval/RULER/scripts/run.sh` |
| `eval.sh` | `eval/RULER/scripts/eval.sh` |
| `config_models.sh` | `eval/RULER/scripts/config_models.sh` |
| `config_tasks.sh` | `eval/RULER/scripts/config_tasks.sh` |
| `synthetic.yaml` | `eval/RULER/scripts/synthetic.yaml` |

### 3.2 Python 模块迁移

#### pred/ 目录

| 源文件 | 需要修改的 import |
|--------|-------------------|
| `pred/call_api.py` | `from xattn.src.load_llama` → `from compass.src.load_llama` |
| `pred/model_wrappers.py` | `from xattn.src.load_llama` → `from compass.src.load_llama` |
| `pred/client_wrappers.py` | 无需修改 |
| `pred/serve_vllm.py` | 无需修改 |
| `pred/serve_trt.py` | 无需修改 |

#### data/ 目录

| 源文件 | 需要修改的 import |
|--------|-------------------|
| `data/prepare.py` | 检查是否有 nemo 依赖需要修改 |
| `data/template.py` | 无 |
| `data/tokenizer.py` | 检查 nemo 依赖 |
| `data/synthetic/niah.py` | 检查 nemo 依赖 |
| `data/synthetic/constants.py` | 无 |
| `data/synthetic/variable_tracking.py` | 检查 |
| `data/synthetic/common_words_extraction.py` | 检查 |
| `data/synthetic/freq_words_extraction.py` | 检查 |
| `data/synthetic/qa.py` | 检查 |

#### eval/ 目录

| 源文件 | 需要修改的 import |
|--------|-------------------|
| `eval/evaluate.py` | 检查 nemo 依赖 |
| `eval/synthetic/constants.py` | 无 |

#### utils/ 目录 (新增)

根据 x-attention task_plan，需要创建本地 `manifest_utils.py` 替代 nemo 依赖：
- [ ] 创建 `eval/RULER/scripts/utils/__init__.py`
- [ ] 创建 `eval/RULER/scripts/utils/manifest_utils.py`

### 3.3 数据文件迁移

| 源文件 | 目标路径 |
|--------|----------|
| `data/synthetic/json/PaulGrahamEssays.json` | 同路径 |
| `data/synthetic/json/hotpotqa.json` | 同路径 (可选) |
| `data/synthetic/json/squad.json` | 同路径 (可选) |
| `data/synthetic/json/download_*.py` | 同路径 |

---

## Phase 4: 创建顶层运行脚本 (conda 模式) `pending`

### 目标
创建基于 conda 环境的运行脚本，**完全不使用 Docker**。

### 文件: `scripts/run_ruler.sh`

```bash
#!/bin/bash
# RULER Benchmark Runner (Conda Mode)
# Usage: ./scripts/run_ruler.sh [MODEL_NAME] [BENCHMARK] [METRIC]

set -e

#############################################
# Configuration
#############################################

# Conda environment
CONDA_ENV="${CONDA_ENV:-ruler}"

# Model settings
MODEL_NAME="${1:-llama3.1-8b-chat}"
BENCHMARK="${2:-synthetic}"
METRIC="${3:-full}"  # Options: full, xattn, avgpool, compass, minfer, flex

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
MODEL_DIR="${MODEL_DIR:-/home/zijie/models}"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

#############################################
# Activate conda and run
#############################################

echo "========================================"
echo "RULER Benchmark (Conda Mode)"
echo "========================================"
echo "Conda Env:    $CONDA_ENV"
echo "Project:      $PROJECT_DIR"
echo "Model Dir:    $MODEL_DIR"
echo "Model:        $MODEL_NAME"
echo "Benchmark:    $BENCHMARK"
echo "Metric:       $METRIC"
echo "========================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Run RULER benchmark
cd "${PROJECT_DIR}/eval/RULER/scripts"
./run.sh "$MODEL_NAME" "$BENCHMARK" "$METRIC"
```

### 关键修改
1. 移除所有 Docker 相关代码
2. 使用 `conda activate` 激活环境
3. 设置 `PYTHONPATH` 指向项目根目录
4. 直接调用 `run.sh`

---

## Phase 5: 创建本地 manifest_utils (替代 NeMo ASR Extra) `pending`

### 目标
NeMo 2.6.1 base 已安装（由 x-attention task_plan 管理），但 **NeMo base 不包含 ASR extra**，
因此 `nemo.collections.asr.parts.utils.manifest_utils` 不可用，需要创建本地实现。

### 环境说明

| 组件 | 状态 | 说明 |
|------|------|------|
| `nemo-toolkit==2.6.1` | ✅ 已安装 (base) | 由 x-attention task_plan 管理 |
| `nemo.collections.asr` | ❌ 不可用 | NeMo base 不含 ASR extra |
| `nemo.collections.nlp` | ❌ 不可用 | NeMo base 不含 NLP extra |

### 需要本地实现的功能

| 原 import | 本地替代 | 涉及文件 |
|-----------|---------|----------|
| `nemo.collections.asr.parts.utils.manifest_utils` | `utils/manifest_utils.py` | call_api.py, evaluate.py, niah.py 等 |

### 本地 manifest_utils.py 实现

根据 x-attention/task_plan.md Phase 1 的规划：

```python
# eval/RULER/scripts/utils/manifest_utils.py
import json

def read_manifest(path):
    """Read JSONL manifest file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_manifest(path, data):
    """Write JSONL manifest file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### 修改 import 语句

```python
# 旧 (NeMo ASR extra - 不可用)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

# 新 (本地实现)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.manifest_utils import read_manifest, write_manifest
```

### tokenizer.py 处理

检查 `data/tokenizer.py` 是否有 NeMo NLP 依赖：
- 如有 `nemo.collections.nlp` 依赖，改用 `transformers` 或 `sentencepiece`
- 大部分情况下可直接使用 HuggingFace tokenizer

---

## Phase 6: 验证和测试 `pending`

### 目标
确保迁移后的代码可以在 **conda 环境**下正常运行。

### 前置条件
- [ ] conda 环境 `ruler` 已创建 (由 x-attention task_plan 管理)
- [ ] 3rdparty/flash-attention 已编译安装
- [ ] 3rdparty/flashinfer 已编译安装
- [ ] 模型文件已下载到 `$MODEL_DIR`

---

### Test 1: 环境验证 `pending`

#### 1.1 激活 conda 环境
```bash
conda activate ruler
```

#### 1.2 验证 Python 版本
```bash
python --version
# 期望: Python 3.10.x
```

#### 1.3 验证 PyTorch 安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# 期望: PyTorch: 2.9.1, CUDA: True
```

#### 1.4 验证 transformers 版本
```bash
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
# 期望: transformers: 4.57.3
```

#### 1.5 验证 flashinfer 安装
```bash
python -c "import flashinfer; print('flashinfer OK')"
# 期望: flashinfer OK
```

#### 1.6 验证 flash_attn 安装
```bash
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')"
# 期望: flash_attn: 2.x.x
```

---

### Test 2: compass 包导入测试 `pending`

#### 2.1 设置 PYTHONPATH
```bash
export PYTHONPATH="/home/zijie/Code/COMPASS:$PYTHONPATH"
```

#### 2.2 测试 compass 包基础导入
```bash
python -c "import compass; print('compass package OK')"
# 期望: compass package OK
```

#### 2.3 测试 compass.src 模块导入
```bash
python -c "from compass.src import load_llama; print('load_llama OK')"
# 期望: load_llama OK
```

#### 2.4 测试 FastPrefillConfig 导入
```bash
python -c "from compass.src.load_llama import FastPrefillConfig; print('FastPrefillConfig OK')"
# 期望: FastPrefillConfig OK
```

#### 2.5 测试 load_model 函数导入
```bash
python -c "from compass.src.load_llama import load_model; print('load_model OK')"
# 期望: load_model OK
```

#### 2.6 测试 threshold 模块导入
```bash
python -c "from compass.threshold.llama_threshold import llama_fuse_16; print(f'llama_fuse_16 shape: {len(llama_fuse_16)}')"
# 期望: llama_fuse_16 shape: 32
```

#### 2.7 测试各算子模块导入 (允许部分失败)
```bash
python -c "
modules = ['Xattention', 'Minference', 'Fullprefill', 'Flexprefill', 'Compass', 'AvgPool']
for mod in modules:
    try:
        exec(f'from compass.src.{mod} import *')
        print(f'{mod}: OK')
    except Exception as e:
        print(f'{mod}: FAILED - {e}')
"
```

---

### Test 3: RULER 数据准备测试 `pending`

#### 3.1 测试 manifest_utils 导入
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts
python -c "
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
from utils.manifest_utils import read_manifest, write_manifest
print('manifest_utils OK')
"
# 期望: manifest_utils OK
```

#### 3.2 测试 tokenizer 模块导入
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts/data
python -c "from tokenizer import select_tokenizer; print('tokenizer OK')"
# 期望: tokenizer OK
```

#### 3.3 测试 template 模块导入
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts/data
python -c "from template import Templates; print(f'Templates: {list(Templates.keys())}')"
# 期望: Templates: ['base', 'meta-chat', 'vicuna-chat', ...]
```

#### 3.4 测试数据生成 (niah_single_1, 短序列)
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts

# 设置环境变量
export PYTHONPATH="/home/zijie/Code/COMPASS:$PYTHONPATH"
export MODEL_DIR="/home/zijie/models"

# 运行数据准备 (4K 长度, 2 samples, 快速测试)
python data/prepare.py \
    --save_dir ./benchmark_root/test_model/synthetic/4096/data \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path "${MODEL_DIR}/Llama-3.1-8B-Instruct" \
    --tokenizer_type hf \
    --max_seq_length 4096 \
    --num_samples 2 \
    --model_template_type base

# 验证生成的文件
ls -la ./benchmark_root/test_model/synthetic/4096/data/niah_single_1/
# 期望: validation.jsonl 文件存在

# 检查文件内容
head -n 1 ./benchmark_root/test_model/synthetic/4096/data/niah_single_1/validation.jsonl
# 期望: {"index": 0, "input": "...", "outputs": [...], "length": ...}
```

#### 3.5 清理测试数据
```bash
rm -rf ./benchmark_root/test_model/
```

---

### Test 4: 模型加载测试 `pending`

#### 4.1 测试 HuggingFace 模型加载 (基础)
```bash
cd /home/zijie/Code/COMPASS
export PYTHONPATH="/home/zijie/Code/COMPASS:$PYTHONPATH"

python -c "
from compass.src.load_llama import load_model, FastPrefillConfig
import torch

# 创建配置
config = FastPrefillConfig(metric='full')  # 使用 full attention 测试

# 加载模型 (这会下载/加载模型权重)
print('Loading model...')
model, tokenizer = load_model(
    fastprefillconfig=config,
    name_or_path='/home/zijie/models/Llama-3.1-8B-Instruct'
)
print(f'Model loaded: {type(model).__name__}')
print(f'Tokenizer loaded: {type(tokenizer).__name__}')

# 释放显存
del model
torch.cuda.empty_cache()
print('Test passed!')
"
```

#### 4.2 测试模型推理 (短序列)
```bash
python -c "
from compass.src.load_llama import load_model, FastPrefillConfig
import torch

config = FastPrefillConfig(metric='full')
model, tokenizer = load_model(
    fastprefillconfig=config,
    name_or_path='/home/zijie/models/Llama-3.1-8B-Instruct'
)

# 简单推理测试
prompt = 'Hello, my name is'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Prompt: {prompt}')
print(f'Response: {response}')

del model
torch.cuda.empty_cache()
"
```

---

### Test 5: RULER 推理测试 (call_api.py) `pending`

#### 5.1 准备测试数据
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts
export PYTHONPATH="/home/zijie/Code/COMPASS:$PYTHONPATH"
export MODEL_DIR="/home/zijie/models"

# 生成测试数据
python data/prepare.py \
    --save_dir ./benchmark_root/test_run/synthetic/4096/data \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path "${MODEL_DIR}/Llama-3.1-8B-Instruct" \
    --tokenizer_type hf \
    --max_seq_length 4096 \
    --num_samples 2 \
    --model_template_type meta-llama3
```

#### 5.2 运行 call_api.py (HF backend, full attention)
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts

python pred/call_api.py \
    --data_dir ./benchmark_root/test_run/synthetic/4096/data \
    --save_dir ./benchmark_root/test_run/synthetic/4096/pred \
    --benchmark synthetic \
    --task niah_single_1 \
    --server_type hf \
    --model_name_or_path "${MODEL_DIR}/Llama-3.1-8B-Instruct" \
    --metric full \
    --temperature 0.0

# 验证输出
ls -la ./benchmark_root/test_run/synthetic/4096/pred/
# 期望: niah_single_1.jsonl 文件存在

cat ./benchmark_root/test_run/synthetic/4096/pred/niah_single_1.jsonl
# 期望: 包含 "pred" 字段的 JSON 行
```

#### 5.3 运行 call_api.py (HF backend, xattn)
```bash
python pred/call_api.py \
    --data_dir ./benchmark_root/test_run/synthetic/4096/data \
    --save_dir ./benchmark_root/test_run/synthetic/4096/pred_xattn \
    --benchmark synthetic \
    --task niah_single_1 \
    --server_type hf \
    --model_name_or_path "${MODEL_DIR}/Llama-3.1-8B-Instruct" \
    --metric xattn \
    --temperature 0.0
```

#### 5.4 运行 call_api.py (HF backend, avgpool)
```bash
python pred/call_api.py \
    --data_dir ./benchmark_root/test_run/synthetic/4096/data \
    --save_dir ./benchmark_root/test_run/synthetic/4096/pred_avgpool \
    --benchmark synthetic \
    --task niah_single_1 \
    --server_type hf \
    --model_name_or_path "${MODEL_DIR}/Llama-3.1-8B-Instruct" \
    --metric avgpool \
    --avgpool_topk 64 \
    --temperature 0.0
```

---

### Test 6: RULER 评估测试 (evaluate.py) `pending`

#### 6.1 运行评估
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts

python eval/evaluate.py \
    --data_dir ./benchmark_root/test_run/synthetic/4096/data \
    --pred_dir ./benchmark_root/test_run/synthetic/4096/pred \
    --benchmark synthetic \
    --task niah_single_1

# 期望输出: 评估结果，包含准确率等指标
```

#### 6.2 验证评估结果文件
```bash
ls -la ./benchmark_root/test_run/synthetic/4096/pred/
# 期望: summary-niah_single_1.csv 或类似的结果文件
```

---

### Test 7: 完整流程测试 (run.sh) `pending`

#### 7.1 修改 config_tasks.sh 为快速测试
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts

# 备份原配置
cp config_tasks.sh config_tasks.sh.bak

# 修改为快速测试配置
cat > config_tasks.sh << 'EOF'
NUM_SAMPLES=2  # 快速测试只用 2 samples
REMOVE_NEWLINE_TAB=false
STOP_WORDS=""

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi

synthetic=(
    "niah_single_1"
)
EOF
```

#### 7.2 运行完整流程
```bash
cd /home/zijie/Code/COMPASS
export PYTHONPATH="/home/zijie/Code/COMPASS:$PYTHONPATH"

# 运行顶层脚本
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full
```

#### 7.3 验证结果
```bash
# 检查输出目录
ls -la eval/RULER/scripts/benchmark_root/llama3.1-8b-chat/synthetic/*/pred/

# 检查 summary 文件
cat eval/RULER/scripts/benchmark_root/llama3.1-8b-chat/synthetic/*/pred/summary*.csv
```

#### 7.4 恢复原配置
```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts
mv config_tasks.sh.bak config_tasks.sh
```

---

### Test 8: 不同 metric 测试 `pending`

#### 8.1 测试所有支持的 metric
```bash
cd /home/zijie/Code/COMPASS

for metric in full xattn avgpool compass minfer flex; do
    echo "=========================================="
    echo "Testing metric: $metric"
    echo "=========================================="

    ./scripts/run_ruler.sh llama3.1-8b-chat synthetic "$metric" || echo "FAILED: $metric"
done
```

---

### Test 9: 长序列测试 (可选) `pending`

#### 9.1 修改 SEQ_LENGTHS
```bash
# 在 config_tasks.sh 或 run.sh 中修改 SEQ_LENGTHS
SEQ_LENGTHS=(4096 8192 16384 32768)
```

#### 9.2 运行长序列测试
```bash
# 注意: 长序列测试需要更多显存
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full
```

---

### Test 10: 清理测试数据 `pending`

```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts

# 删除测试生成的数据
rm -rf ./benchmark_root/test_run/
rm -rf ./benchmark_root/test_model/

# 保留正式测试结果 (如果需要)
# ls benchmark_root/
```

---

## 测试检查清单

| Test | 描述 | 状态 | 备注 |
|------|------|------|------|
| Test 1.1 | conda 环境激活 | `pending` | |
| Test 1.2 | Python 版本检查 | `pending` | 期望 3.10.x |
| Test 1.3 | PyTorch 安装验证 | `pending` | 期望 2.9.1 |
| Test 1.4 | transformers 版本 | `pending` | 期望 4.57.3 |
| Test 1.5 | flashinfer 安装 | `pending` | |
| Test 1.6 | flash_attn 安装 | `pending` | |
| Test 2.1 | PYTHONPATH 设置 | `pending` | |
| Test 2.2 | compass 包导入 | `pending` | |
| Test 2.3 | compass.src 导入 | `pending` | |
| Test 2.4 | FastPrefillConfig 导入 | `pending` | |
| Test 2.5 | load_model 导入 | `pending` | |
| Test 2.6 | threshold 导入 | `pending` | |
| Test 2.7 | 算子模块导入 | `pending` | 允许部分失败 |
| Test 3.1 | manifest_utils 导入 | `pending` | |
| Test 3.2 | tokenizer 导入 | `pending` | |
| Test 3.3 | template 导入 | `pending` | |
| Test 3.4 | 数据生成测试 | `pending` | |
| Test 4.1 | 模型加载测试 | `pending` | |
| Test 4.2 | 模型推理测试 | `pending` | |
| Test 5.1 | 准备推理测试数据 | `pending` | |
| Test 5.2 | call_api (full) | `pending` | |
| Test 5.3 | call_api (xattn) | `pending` | |
| Test 5.4 | call_api (avgpool) | `pending` | |
| Test 6.1 | evaluate.py 运行 | `pending` | |
| Test 6.2 | 评估结果验证 | `pending` | |
| Test 7.1 | 配置快速测试 | `pending` | |
| Test 7.2 | 完整流程测试 | `pending` | |
| Test 7.3 | 结果验证 | `pending` | |
| Test 8.1 | 多 metric 测试 | `pending` | |
| Test 9.1-9.2 | 长序列测试 | `pending` | 可选 |
| Test 10 | 清理测试数据 | `pending` | |

---

## Import 修改完整清单

### compass 包内部修改

| 文件 | 修改内容 |
|------|----------|
| `compass/src/load_llama.py` | 所有 `from xattn.` → `from compass.` |

### RULER 模块修改

| 文件 | 修改内容 |
|------|----------|
| `eval/RULER/scripts/pred/call_api.py` | `from xattn.src.load_llama` → `from compass.src.load_llama` |
| `eval/RULER/scripts/pred/model_wrappers.py` | `from xattn.src.load_llama` → `from compass.src.load_llama` |
| `eval/RULER/scripts/pred/call_api.py` | `from nemo...manifest_utils` → `from utils.manifest_utils` |
| `eval/RULER/scripts/eval/evaluate.py` | `from nemo...manifest_utils` → `from utils.manifest_utils` |
| `eval/RULER/scripts/data/synthetic/niah.py` | `from nemo...manifest_utils` → `from utils.manifest_utils` |
| `eval/RULER/scripts/data/tokenizer.py` | NeMo tokenizer → sentencepiece/transformers |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| - | - | - |

---

## Notes

1. **完全不使用 Docker**，所有运行基于 conda 环境
2. 环境配置由 `x-attention/task_plan.md` 管理，包括：
   - conda 环境 `ruler` 创建
   - PyTorch 2.9.1 + CUDA 12.8
   - transformers 4.57.3
   - **nemo-toolkit 2.6.1 (base)**
   - flash-attention / flashinfer 源码编译
3. 原 xattn 包被重命名为 compass 包
4. 所有 `from xattn.` 的 import 需要修改为 `from compass.`
5. **NeMo 已安装 (base only)**，但 ASR/NLP extra 不可用：
   - `nemo.collections.asr.parts.utils.manifest_utils` → 本地 `utils/manifest_utils.py`
   - `nemo.collections.nlp` 相关 → 使用 transformers/sentencepiece
6. 3rdparty 目录（flash-attention, flashinfer）由 x-attention task_plan 管理
7. 测试时需要确保 `PYTHONPATH` 包含项目根目录
