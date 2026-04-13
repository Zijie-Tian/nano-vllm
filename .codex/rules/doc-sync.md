# Documentation Sync Rule

## 强制规则 (Mandatory Rule)

**每当 `docs/` 目录下的文档发生任何结构性变更（新增、重命名、删除或核心意图改变）时，必须同步更新所有的环境配置入口文件。**

### 必须同步更新的文件清单

1. `AGENTS.md` (为 Codex / 项目主控提供索引)
2. `CLAUDE.md` (为 Claude 实例提供索引)
3. `GEMINI.md` (为 Gemini 实例提供索引)

### 更新内容

必须在上述三个文件的 **Documentation Index** (或等效的表格) 中，精确添加或修改对应的条目：
- `Document` 列: 文档的相对路径，如 ``[`docs/new_doc.md`](docs/new_doc.md)``
- `Purpose` 列: 一句话概括文档的核心作用、涉及的系统机制或总结的数据指标。

### 触发条件

- 创建了新的解释文档、Benchmark 报告、Debug 记录或架构说明。
- 重构了现有的 Markdown 文档名称，或合并了多个文档。
- 废弃了旧的临时分析记录。

### 执行检查表 (Checklist)

- [ ] `docs/your_new_file.md` 已写入并完成最后审查。
- [ ] 已打开 `AGENTS.md` 并更新 Index 表格。
- [ ] 已打开 `CLAUDE.md` 并更新 Index 表格。
- [ ] 已打开 `GEMINI.md` 并更新 Index 表格。
