# Claude Code Configuration - COMPASS Project

## 📚 Documentation References (Lazy Loading)

> **为减少 runtime token 消耗，详细文档按需加载。需要时读取对应文档。**

| 文档 | 路径 | 用途 | 何时读取 |
|------|------|------|----------|
| 环境配置指南 | `docs/ENVIRONMENT_SETUP.md` | 完整环境配置步骤、依赖版本、常见问题 | 配置新环境时 |
| Claude 配置任务 | `docs/CLAUDE_SETUP_TASK.md` | Claude 专用的详细配置步骤 (7步) | 在新主机配置环境时 |
| RULER XAttention 集成 | `docs/RULER_NANOVLLM_XATTN_INTEGRATION.md` | NanoVLLM XAttention 集成实现细节与测试结果 | 使用 RULER nanovllm 后端时 |
| XAttn Chunked Prefill | `docs/XATTN_CHUNKED_PREFILL.md` | Chunked prefill 的 XAttention 实现、API 说明、使用方式 | 使用 chunked prefill 或 xattn_chunked metric 时 |
| LongBench Benchmark 指南 | `docs/LONG_BENCH_BENCHMARK_GUIDE.md` | LongBench 运行命令、模型别名、子集 preset、输出规则 | 配置或运行 LongBench 测试时 |
| TriAttention Calibration 指南 | `docs/TRIATTENTION_CALIBRATION_GUIDE.md` | TriAttention 标定脚本的作用、统计对象、公式与参数说明 | 理解或使用 TriAttention 标定时 |
| TriAttention 复现指南 | `docs/TRIATTENTION_REPRODUCTION_GUIDE.md` | 当前工程 TriAttention 的复现逻辑、LongBench 接入方式，以及 standard 对照脚本使用方法 | 对比当前实现与 standard TriAttention 时 |
| 依赖列表 | `requirements.txt` | Python 依赖及版本 | pip install 时 |
| **文档规则** | `.claude/rules/documentation-lazy-loading.md` | 文档写作的 lazy loading 规范 | 创建新文档时 |

### 使用示例
```bash
# 在新主机配置环境时，读取 Claude 配置任务文档
Read("docs/CLAUDE_SETUP_TASK.md")

# 按文档中的 7 个步骤依次执行配置
```

---

## 🎯 Project-Specific: COMPASS RULER Benchmark

### 运行 RULER 测试
```bash
conda activate ruler
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH
export MODEL_DIR=/path/to/models
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

### 指定任务
```bash
# 单任务
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1

# 多任务
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1,vt,qa_1
```

---
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
When updating the `nanovllm` submodule, ALWAYS ensure it is checked out to the `tzj/minference` branch.
