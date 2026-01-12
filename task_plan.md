# RULER Benchmark Migration Plan

## Goal
将 x-attention 项目中的 RULER 测试基准代码完整迁移到 COMPASS 项目中，包括所有相关脚本、Python 模块、算子依赖，并重新设定 import 路径。

## Current Status: Planning
- Start Date: 2026-01-13
- Status: `planning`

---

## Phase 1: 创建项目目录结构 `pending`

### 目标
建立 COMPASS 项目的基础目录结构，与 RULER 测试相关的模块组织。

### 任务
1. 创建主要目录结构：
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
   │       ├── requirements.txt
   │       ├── Dockerfile
   │       └── README.md
   └── scripts/                    # 顶层运行脚本
       ├── run_ruler_docker.sh
       ├── run_ruler_nanovllm.sh
       └── run_ruler_tasks.sh
   ```

### 文件清单
- [ ] 创建 compass/ 包目录
- [ ] 创建 eval/RULER/ 目录结构
- [ ] 创建 scripts/ 目录

---

## Phase 2: 迁移核心算子模块 (xattn -> compass) `pending`

### 目标
将 xattn 包完整迁移为 compass 包，修改所有内部引用。

### 源文件列表 (x-attention/xattn/)
| 源路径 | 目标路径 | 需要修改的 import |
|--------|----------|-------------------|
| `__init__.py` | `compass/__init__.py` | 无 |
| `src/load_llama.py` | `compass/src/load_llama.py` | `xattn.` → `compass.` |
| `src/AvgPool.py` | `compass/src/AvgPool.py` | 无 |
| `src/Fullprefill.py` | `compass/src/Fullprefill.py` | 无 |
| `src/Xattention.py` | `compass/src/Xattention.py` | 待检查 |
| `src/Flexprefill.py` | `compass/src/Flexprefill.py` | 待检查 |
| `src/Minference.py` | `compass/src/Minference.py` | 待检查 |
| `src/Compass.py` | `compass/src/Compass.py` | 待检查 |
| `src/kernels.py` | `compass/src/kernels.py` | 待检查 |
| `src/utils.py` | `compass/src/utils.py` | 待检查 |
| `threshold/llama_threshold.py` | `compass/threshold/llama_threshold.py` | 无 |
| `threshold/profile_threshold/` | `compass/threshold/profile_threshold/` | 待检查 |

### 关键修改点
1. `load_llama.py` 中的 import 修改：
   - `from xattn.threshold.llama_threshold import ...` → `from compass.threshold.llama_threshold import ...`
   - `from xattn.src.Xattention import ...` → `from compass.src.Xattention import ...`
   - 同理修改所有其他 xattn 引用

---

## Phase 3: 迁移 RULER Benchmark 模块 `pending`

### 目标
完整迁移 RULER 测试框架到 eval/RULER/ 目录。

### 3.1 核心脚本迁移
| 源路径 | 目标路径 |
|--------|----------|
| `eval/RULER/scripts/run.sh` | `eval/RULER/scripts/run.sh` |
| `eval/RULER/scripts/eval.sh` | `eval/RULER/scripts/eval.sh` |
| `eval/RULER/scripts/config_models.sh` | `eval/RULER/scripts/config_models.sh` |
| `eval/RULER/scripts/config_tasks.sh` | `eval/RULER/scripts/config_tasks.sh` |
| `eval/RULER/scripts/synthetic.yaml` | `eval/RULER/scripts/synthetic.yaml` |

### 3.2 Python 模块迁移
#### pred/ 目录
| 源文件 | 需要修改的 import |
|--------|-------------------|
| `pred/call_api.py` | `from xattn.src.load_llama import FastPrefillConfig` → `from compass.src.load_llama import FastPrefillConfig` |
| `pred/model_wrappers.py` | `from xattn.src.load_llama import load_model, FastPrefillConfig` → `from compass.src.load_llama import ...` |
| `pred/client_wrappers.py` | 无需修改 |
| `pred/serve_vllm.py` | 无需修改 |
| `pred/serve_trt.py` | 无需修改 |

#### data/ 目录
| 源文件 | 需要修改的 import |
|--------|-------------------|
| `data/prepare.py` | 无 |
| `data/template.py` | 无 |
| `data/tokenizer.py` | 无 |
| `data/synthetic/niah.py` | 无 |
| `data/synthetic/constants.py` | 无 |
| `data/synthetic/variable_tracking.py` | 待检查 |
| `data/synthetic/common_words_extraction.py` | 待检查 |
| `data/synthetic/freq_words_extraction.py` | 待检查 |
| `data/synthetic/qa.py` | 待检查 |

#### eval/ 目录
| 源文件 | 需要修改的 import |
|--------|-------------------|
| `eval/evaluate.py` | 无 |
| `eval/synthetic/constants.py` | 无 |

### 3.3 数据文件迁移
- `data/synthetic/json/PaulGrahamEssays.json` - 必需
- `data/synthetic/json/hotpotqa.json` - 可选 (QA 任务)
- `data/synthetic/json/squad.json` - 可选 (QA 任务)

---

## Phase 4: 迁移顶层运行脚本 `pending`

### 目标
迁移并修改顶层脚本以适配新的项目结构。

### 文件列表
| 源路径 | 目标路径 | 修改内容 |
|--------|----------|----------|
| `scripts/run_ruler_tasks.sh` | `scripts/run_ruler_tasks.sh` | 修改路径引用 |
| `scripts/run_ruler_docker.sh` | `scripts/run_ruler_docker.sh` | 修改 mount 路径，PYTHONPATH |
| `scripts/run_ruler_nanovllm.sh` | `scripts/run_ruler_nanovllm.sh` | 修改 mount 路径，PYTHONPATH |

### 关键修改
1. Docker 脚本中的路径修改：
   - `-v $PROJECT_DIR:/workspace/x-attention` → `-v $PROJECT_DIR:/workspace/compass`
   - `-e PYTHONPATH=/workspace/x-attention:/workspace/nano-vllm` → `-e PYTHONPATH=/workspace/compass:/workspace/nano-vllm`
   - 工作目录修改

---

## Phase 5: 迁移配置文件和依赖 `pending`

### 目标
迁移项目配置和依赖定义。

### 文件列表
- `eval/RULER/requirements.txt` - RULER 特定依赖
- `eval/RULER/Dockerfile` - Docker 构建文件
- `eval/RULER/build_docker.sh` - Docker 构建脚本
- `eval/RULER/.dockerignore` - Docker 忽略文件

---

## Phase 6: 验证和测试 `pending`

### 目标
确保迁移后的代码可以正常运行。

### 验证步骤
1. [ ] Python import 测试：验证 compass 包可以正确导入
2. [ ] RULER 数据准备测试：运行 data/prepare.py
3. [ ] 模型推理测试：运行 pred/call_api.py (HF backend)
4. [ ] 评估测试：运行 eval/evaluate.py
5. [ ] 完整流程测试：运行 run.sh

---

## Dependencies

### Python 包依赖 (核心)
- torch>=2.4.0
- transformers>=4.45.2
- flashinfer (用于 Full_prefill 和 decode)
- block_sparse_attn (用于 AvgPool_prefill)
- nemo-toolkit[all] (用于 manifest_utils, tokenizer)
- wonderwords (用于数据生成)
- nltk (用于句子分割)
- pandas (用于结果输出)
- pyyaml (用于配置读取)

### 外部依赖
- nano-vllm (NanoVLLM 推理引擎，通过 PYTHONPATH 引入)

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| - | - | - |

---

## Notes
- 原 xattn 包被重命名为 compass 包
- 所有 `from xattn.` 的 import 需要修改为 `from compass.`
- RULER benchmark 主要依赖 nemo-toolkit 的 manifest_utils
- block_sparse_attn 是 AvgPool 算法的关键依赖，需要确保在目标环境可用
- flashinfer 是 Full_prefill 和 decode 阶段的关键依赖
