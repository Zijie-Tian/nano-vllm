---
description: Documentation lazy loading rule for reducing token consumption at runtime
---

# Documentation Lazy Loading Rule

## Purpose
减少 runtime 加载的 token 量，通过将详细文档分离到 `docs/` 目录，仅在需要时按需加载。

## Rule

### 1. 文档分离原则

**GEMINI.md / CLAUDE.md 中只保留：**
- 核心配置和指令（必须每次加载）
- 文档引用表（指向详细文档的索引）
- 简短的项目概述

**docs/ 目录存放：**
- 详细的配置指南
- 步骤式教程
- 故障排除文档
- API 参考文档
- 任何超过 50 行的详细说明

### 2. 文档引用表格式

在 GEMINI.md 和 CLAUDE.md 中使用以下表格格式引用文档：

```markdown
## 📚 Documentation References (Lazy Loading)

> **为减少 runtime token 消耗，详细文档按需加载。需要时读取对应文档。**

| 文档 | 路径 | 用途 | 何时读取 |
|------|------|------|----------|
| 文档名称 | `docs/xxx.md` | 简短描述 | 触发条件 |
```

### 3. 何时创建新文档

创建新的独立文档当：
- 内容超过 50 行
- 内容是特定场景下才需要的（如环境配置、故障排除）
- 内容是教程或步骤式指南
- 内容可能需要频繁更新

### 4. 文档命名规范

```
docs/
├── ENVIRONMENT_SETUP.md      # 环境配置类
├── TROUBLESHOOTING.md        # 故障排除类
├── API_REFERENCE.md          # API 参考类
└── <FEATURE>_GUIDE.md        # 功能指南类
```

### 5. 读取时机指南

| 触发条件 | 应读取的文档 |
|----------|--------------|
| 配置新环境 | ENVIRONMENT_SETUP.md |
| 遇到错误 | TROUBLESHOOTING.md |
| 需要 API 细节 | API_REFERENCE.md |
| 实现特定功能 | 相关 _GUIDE.md |

### 6. 双配置维护

添加新文档时，**必须同步更新 GEMINI.md 和 CLAUDE.md 的文档引用表**，确保两份配置文件的索引保持一致。

## Benefits

1. **Token 节省**: 配置文件保持精简，减少每次会话加载的 token
2. **按需加载**: 只在实际需要时读取详细文档
3. **易于维护**: 详细文档独立更新，不影响核心配置
4. **清晰索引**: 文档引用表提供清晰的文档地图
