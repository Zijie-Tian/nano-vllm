---
description: 3rdparty directory is read-only; modifications must be done in separate dev repos
---

# 3rdparty Policy

## 3rdparty 目录说明

`3rdparty/` 目录下的代码是**固定的稳定版本**，仅供 COMPASS 项目使用，**不应在此修改**。

### 目录结构

| 路径 | 用途 | 是否可修改 |
|------|------|-----------| 
| `COMPASS/3rdparty/nanovllm` | nanovllm 稳定版（submodule） | ❌ 不可修改 |
| `COMPASS/3rdparty/flash-attention` | flash-attention 稳定版 | ❌ 不可修改 |
| `COMPASS/3rdparty/flashinfer` | flashinfer 稳定版 | ❌ 不可修改 |

### 开发版本位置

如需修改或调试 3rdparty 依赖，应在**独立的开发目录**进行：

| 项目 | 开发目录 | 说明 |
|------|----------|------|
| nanovllm | `/home/zijie/Code/nano-vllm` | nanovllm 主开发仓库 |

### 工作流程

1. **发现 bug / 需要新功能**：
   - 将问题文档写到开发仓库的 `docs/` 目录
   - 例如：`/home/zijie/Code/nano-vllm/docs/issue_xxx.md`

2. **修复完成后**：
   - 在开发仓库提交并测试
   - 更新 COMPASS 的 3rdparty submodule 到新版本

3. **COMPASS 测试**：
   - 使用 `3rdparty/` 下的稳定版本运行测试
   - 如需临时测试开发版，修改 PYTHONPATH 指向开发目录

### 示例

```bash
# 使用稳定版（默认）
./run.sh llama3.1-8b-nanovllm synthetic --metric full

# 临时使用开发版测试
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH ./run.sh llama3.1-8b-nanovllm synthetic --metric full
```
