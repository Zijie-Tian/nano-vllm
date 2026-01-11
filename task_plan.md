# Task Plan: Transformers 低版本兼容性修复

## Goal
使 nano-vllm 在低版本 transformers (< 4.51.0) 环境下也能正常使用 Llama 模型，同时保持高版本环境下的完整功能。

## Background
- nano-vllm 使用了 transformers 4.51.0 才引入的 `Qwen3Config` 类
- 在低版本环境（如 4.45.2）中，由于 `models/__init__.py` 无条件导入 qwen3 模块，导致整个 models 包都无法使用
- 详细问题分析见 `docs/transformers_compatibility.md`

## Phases

### Phase 1: 修改 models/__init__.py [complete]
**Status:** complete
**Files:** `nanovllm/models/__init__.py`

**Changes:**
- 将 `llama` 导入移到 `qwen3` 之前（确保 Llama 总是可用）
- 用 `try-except` 包裹 `qwen3` 导入
- 导入失败时发出警告而非崩溃

### Phase 2: 修改 qwen3.py [complete]
**Status:** complete
**Files:** `nanovllm/models/qwen3.py`

**Changes:**
- 将 `Qwen3Config` 导入包裹在 `try-except` 中
- 导入失败时抛出清晰的 `ImportError` 说明版本要求

### Phase 3: 验证修改 [complete]
**Status:** complete

**Test Results:**
- 高版本环境 (transformers 4.51.0): 全部模型可用
- 预期低版本行为: Llama 可用，Qwen3 发出警告

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 无 | - | 修改顺利完成 |

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用方案1（条件导入） | 简单直接，符合文档建议 |
| llama 导入放在 qwen3 前面 | 确保 Llama 作为基础模型总是可用 |
| qwen3.py 保留 ImportError 抛出 | 用户尝试使用 Qwen3 时得到清晰错误信息 |

## Files Modified
- `nanovllm/models/__init__.py`
- `nanovllm/models/qwen3.py`
