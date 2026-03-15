# Antigravity Configuration - COMPASS Project

## 📚 Documentation References (Lazy Loading)

> **为减少 runtime token 消耗，详细文档按需加载。需要时读取对应文档。**

| 文档 | 路径 | 用途 | 何时读取 |
|------|------|------|----------|
| 环境配置指南 | `docs/ENVIRONMENT_SETUP.md` | 完整环境配置步骤、依赖版本、常见问题 | 配置新环境时 |
| RULER XAttention 集成 | `docs/RULER_NANOVLLM_XATTN_INTEGRATION.md` | NanoVLLM XAttention 集成实现细节与测试结果 | 使用 RULER nanovllm 后端时 |
| XAttn Chunked Prefill | `docs/XATTN_CHUNKED_PREFILL.md` | Chunked prefill 的 XAttention 实现、API 说明 | 使用 chunked prefill 或 xattn_chunked metric 时 |
| BLASST RULER 集成 | `docs/BLASST_RULER_INTEGRATION.md` | BLASST 稀疏注意力与 RULER 集成 | 使用 BLASST metric 时 |
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

---

## 📌 General Rules

- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Never save working files, text/mds and tests to the root folder.
- 添加新文档时，必须同步更新 `GEMINI.md` 和 `CLAUDE.md` 的文档引用表。
- When updating the `nanovllm` submodule, ALWAYS ensure it is checked out to the `tzj/minference` branch.
