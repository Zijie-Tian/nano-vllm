# Findings: Transformers 兼容性研究

## Key Discoveries

### 1. Qwen3Config 版本依赖
- `Qwen3Config` 首次出现在 transformers **4.51.0**
- transformers 4.45.2 只包含 `Qwen2` 系列模型
- 版本对比：

| transformers 版本 | Qwen3 支持 | 可用 Qwen 模型 |
|------------------|-----------|---------------|
| < 4.51.0 | 不支持 | qwen2, qwen2_audio, qwen2_moe, qwen2_vl |
| >= 4.51.0 | 支持 | qwen2 系列 + qwen3, qwen3_moe |

### 2. 级联导入问题
导入失败的根本原因是 `__init__.py` 的无条件导入：

```python
# 原代码 - 问题所在
from nanovllm.models import qwen3  # 失败
from nanovllm.models import llama  # 因为上一行失败，这里也无法执行
```

这导致即使只想使用 Llama 模型也会失败。

### 3. Qwen3Config 使用位置
在 `qwen3.py` 中的使用：
- Line 4: 导入语句
- Line 128: `Qwen3DecoderLayer.__init__` 类型注解
- Line 170: `Qwen3Model.__init__` 类型注解
- Line 202: `Qwen3ForCausalLM.__init__` 类型注解

### 4. 解决方案评估
| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| 条件导入 | 简单，隔离性好 | 无 | **采用** |
| AutoConfig + duck typing | 类型检查更灵活 | 过度工程 | 未采用 |
| 版本检查 + 全局标志 | 可以在启动时统一检查 | 增加复杂度 | 未采用 |

## References
- [Transformers Qwen3 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen3)
- 项目文档: `docs/transformers_compatibility.md`
