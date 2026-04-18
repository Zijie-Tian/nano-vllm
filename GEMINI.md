# Antigravity Configuration - COMPASS Project

## 📚 Documentation References (Lazy Loading)

> **为减少 runtime token 消耗，详细文档按需加载。需要时读取对应文档。**

| 文档 | 路径 | 用途 | 何时读取 |
|------|------|------|----------|
| 环境配置指南 | `docs/ENVIRONMENT_SETUP.md` | 完整环境配置步骤、依赖版本、常见问题 | 配置新环境时 |
| RULER XAttention 集成 | `docs/RULER_NANOVLLM_XATTN_INTEGRATION.md` | NanoVLLM XAttention 集成实现细节与测试结果 | 使用 RULER nanovllm 后端时 |
| XAttn Chunked Prefill | `docs/XATTN_CHUNKED_PREFILL.md` | Chunked prefill 的 XAttention 实现、API 说明 | 使用 chunked prefill 或 xattn_chunked metric 时 |
| BLASST RULER 集成 | `docs/BLASST_RULER_INTEGRATION.md` | BLASST 稀疏注意力与 RULER 集成 | 使用 BLASST metric 时 |
| LongBench Benchmark 指南 | `docs/LONG_BENCH_BENCHMARK_GUIDE.md` | LongBench 运行命令、模型别名、子集 preset、输出规则 | 配置或运行 LongBench 测试时 |
| TriAttention Calibration 指南 | `docs/TRIATTENTION_CALIBRATION_GUIDE.md` | TriAttention 标定脚本的作用、统计对象、公式与参数说明 | 理解或使用 TriAttention 标定时 |
| TriAttention 复现指南 | `docs/TRIATTENTION_REPRODUCTION_GUIDE.md` | 当前工程 TriAttention 的复现逻辑、LongBench 接入方式，以及 standard 对照脚本使用方法 | 对比当前实现与 standard TriAttention 时 |
| LongBench Standard Runner 指南 | `docs/LONG_BENCH_STANDARD_RUNNER_GUIDE.md` | `tests/test_longbench_standard_runner.py` 的定位、参数、输入输出与推荐命令 | 需要跑独立的 upstream-like LongBench 对照路径时 |
| GQA Dense Head 问题 | `docs/GQA_DENSE_HEAD_ISSUE.md` | GQA 中 Dense Head 问题分析 | 处理 GQA 稀疏化时 |
| GQA 解决方案 | `docs/gqa-dense-head-solutions/*.md` | 10 种 GQA Dense Head 解决方案 | 评估/实现 GQA 优化方案时 |
| Q Head Regrouping | `docs/04_q_head_regrouping.md` | Q-Head 重组方案 | 实现 Q-Head 重组时 |
| KV Dim Reorder | `docs/05_kv_dim_reorder.md` | KV 维度重排方案 | 实现 KV 维度重排时 |
| RULER 路径分析 | `docs/RULER_NANOVLLM_XATTN_PATHS_ANALYSIS.md` | RULER NanoVLLM 路径分析 | 调试 RULER 路径问题时 |
| 可行性报告模板 | `docs/FEASIBILITY_REPORT_TEMPLATE.md` | 可行性分析报告标准格式 | 编写可行性报告时 |
| 依赖列表 | `requirements.txt` | Python 依赖及版本 | pip install 时 |

---

## 🎯 Project-Specific: COMPASS RULER Benchmark

### 运行 RULER 测试
```bash
conda activate ruler
export PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH
export MODEL_DIR=/home/zijie/models
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd eval/RULER/scripts
bash run.sh llama3.1-8b-chat synthetic --metric full
```

### 可用 Metric
| Metric | 描述 |
|--------|------|
| `full` | 完整 attention (基准) |
| `xattn` | X-attention (full prefill) |
| `xattn_chunked` | X-attention (chunked prefill) |
| `avgpool` | 平均池化稀疏 |
| `minfer` | Minference |
| `compass` | COMPASS 方法 |
| `flex` | FlexPrefill |

### 指定任务
```bash
# 单任务
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1

# 多任务
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1,vt,qa_1
```

### 环境变数及快速测试
| Variable | Default | Description |
|----------|---------|-------------|
| `CONDA_ENV` | `ruler` | Conda environment name |
| `MODEL_DIR` | `/home/zijie/models` | Model weights directory |
| `CUDA_VISIBLE_DEVICES` | - | GPU selection |

测试 imports 和下载数据集：
```bash
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH python -c "from compass.src.Compass import Compass; print('OK')"
cd eval/RULER && bash setup.sh
```

---

## 📌 General Rules

**CRITICAL: 以后所有的 rule 必须直接写到此 `GEMINI.md` 中，不再使用 `.agents/rules` 目录存放规则文件。**

- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Never save working files, text/mds and tests to the root folder.
- 添加新文档时，必须同步更新 `GEMINI.md` 和 `CLAUDE.md` 的文档引用表。
- When updating the `nanovllm` submodule, ALWAYS ensure it is checked out to the `tzj/minference` branch.

### 🛡️ 3rdparty Policy
`3rdparty/` 目录下的代码是固定的稳定版本，仅供 COMPASS 使用，**不应在此修改**。
- ❌ 不可修改：`nanovllm`, `flash-attention`, `flashinfer`
- **开发版本位置**：需要修改时在独立开发目录进行（如 `/home/zijie/Code/nano-vllm`）。
- **工作流程**：在开发仓库修改并测试 -> 更新 COMPASS 的 3rdparty submodule -> COMPASS 测试。临时测试可修改 PYTHONPATH 指向开发目录。

### 📝 Documentation Management & Lazy Loading
- **GEMINI.md 和 CLAUDE.md 中只保留：** 核心配置和指令、文档引用表、简短的项目概述。
- **docs/ 目录存放：** 任何超过 50 行的详细说明（架构、实现、debug技巧等）、API参考文档、独立故障排除或功能指南。
- **理论与实测**：技术文档中应包含理论分析（含公式估算）与实测数据的对比，误差应<10%。过长的文档考虑重构以节约 token。

### 📊 Feasibility Report Rule
当进行**评估方案可行性**分析时，必须按照 `docs/FEASIBILITY_REPORT_TEMPLATE.md` 中的模板格式编写。
1. Executive Summary 中**必须包含可行性评分 (X/10)** 及其原因分类（✅/⚠️/❌）。无评分的报告视为不完整。
2. 必填章节不可省略：详细设计 (含伪代码), 相关工作 (≥3篇引用), 理论分析 (含量化对比表), 实现复杂度 (含时间估计), 潜在风险 (含 corner cases), COMPASS 结合点 (具体到文件), 结论 (包含 next steps)。
3. 保存在指定的 `docs/` 子目录下，命名格式为 `XX_方案简称.md`。

### 🖥️ GPU Testing Rules
**GPU Card Assignment (CRITICAL) - Before executing ANY GPU command:**
1. 检查 user 是否指定了 GPU。
2. 如果 user 未指定，**必须停止并询问用户**("Which GPU should I use?")。严禁假设或猜测。
3. 任何 GPU 执行命令必须显式加前缀 `CUDA_VISIBLE_DEVICES=X`。
测试前可用 `nvidia-smi` 检查模型并估计 VRAM 使用（如 LLaMA 3.1 8B 128k 约需 82GB）。

### 💾 KVCache-RoPE Data On-Demand Download
`results/kvcache-rope/` 存放从阿里云盘下载的数据，用于理论验证分析。
- **强制规则**：在执行任何需要此数据的代码前，必须先执行 `ls results/kvcache-rope/{model}/{length}/layer_{xx}.pt` 检查本地文件是否存在。
- 如果不存在，提醒用户从阿里云盘 (`/data/COMPASS/kvcache-rope/`) 下载，并确认文件就绪(单层约 603MB)后再执行后续操作。下载时记得 unset 代理。

### ⚙️ RULER Task Configuration Rules
RULER benchmark 的 task 配置分布在 `eval/RULER/scripts/config_tasks.sh` 和 `eval/RULER/scripts/eval.sh`，修改时**必须同步更新两处文件**以确保 `synthetic` 数组一致。
- **使用 `#` 注释来禁用 task，绝对禁止删除 task 对应的代码行。**

### 🧪 Testing Guidelines
- **文件命名与结构**：所有测试文件名为 `test_*.py`。采用 "educational scripts" 风格，重在演示代码流转和调用细节，而不是单纯地断言。辅助步骤提取在头部，主流程写成顶级执行脚本。
- **结果输出**：尽量使用 `assert`，尽可能少地使用 print，最后输出 "PASSED"。
