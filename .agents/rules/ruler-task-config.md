---
description: RULER task configuration must be synced between config_tasks.sh and eval.sh
---

# RULER Task Configuration Rules

## Task 配置文件位置

RULER benchmark 的 task 配置分布在两个文件中，修改时**必须同步更新**：

| 文件 | 路径 | 用途 |
|------|------|------|
| config_tasks.sh | `eval/RULER/scripts/config_tasks.sh` | 推理阶段 task 列表 |
| eval.sh | `eval/RULER/scripts/eval.sh` | 评估阶段 task 列表 |

## 修改规则

### 1. 同步修改

修改任何 task 配置时，**必须同时修改两个文件**，确保 `synthetic` 数组保持一致。

### 2. 使用注释禁用，禁止删除

**正确做法**：使用 `#` 注释禁用 task
```bash
synthetic=(
    "niah_single_1"
    # "niah_single_2"    # 已禁用
    # "niah_single_3"    # 已禁用
    "niah_multikey_1"
)
```

**错误做法**：直接删除 task 行
```bash
# ❌ 不要这样做！
synthetic=(
    "niah_single_1"
    "niah_multikey_1"
)
```

### 3. 完整 task 列表参考

始终保留完整的 task 列表作为参考：
```bash
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
```

## Checklist

修改 RULER task 配置前：
- [ ] 确认要修改的 task
- [ ] 打开 config_tasks.sh
- [ ] 打开 eval.sh
- [ ] 使用注释（#）禁用/启用 task
- [ ] 确保两个文件的 synthetic 数组完全一致
- [ ] 不要删除任何 task 行
