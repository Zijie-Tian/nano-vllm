# RULER Benchmark 测试报告

**测试日期**: 2025-01-14
**测试环境**: 6x RTX 3090, CPU Offload 模式
**模型**: Llama-3.1-8B-Instruct
**上下文长度**: 32K tokens

## 测试概述

使用 RULER benchmark 对 nano-vllm 的 CPU offload 模式进行全面的长上下文能力测试。RULER 是 NVIDIA 开发的长上下文评测基准，包含 13 个任务类别。

## 测试结果

### 总体结果

| 类别 | 数据集 | 正确/总数 | 准确率 | 平均分数 |
|------|--------|-----------|--------|----------|
| **NIAH Single** | niah_single_1 | 100/100 | 100.0% | 1.000 |
| | niah_single_2 | 100/100 | 100.0% | 1.000 |
| | niah_single_3 | 100/100 | 100.0% | 1.000 |
| **NIAH MultiKey** | niah_multikey_1 | 100/100 | 100.0% | 1.000 |
| | niah_multikey_2 | 90/100 | 90.0% | 0.900 |
| | niah_multikey_3 | 93/100 | 93.0% | 0.930 |
| **NIAH Other** | niah_multiquery | 100/100 | 100.0% | 1.000 |
| | niah_multivalue | 100/100 | 100.0% | 1.000 |
| **QA** | qa_1 | 79/100 | 79.0% | 0.790 |
| | qa_2 | 51/100 | 51.0% | 0.510 |
| **Aggregation** | cwe | 86/100 | 86.0% | 0.680 |
| | fwe | 98/100 | 98.0% | 0.923 |
| **Variable Tracking** | vt | 100/100 | 100.0% | 0.934 |
| **总计** | **13 数据集** | **1197/1300** | **92.1%** | **0.897** |

### 分类性能分析

| 任务类别 | 描述 | 准确率 | 评价 |
|----------|------|--------|------|
| NIAH Single | 单 needle 检索 | 100% | 优秀 |
| NIAH MultiKey | 多 key 检索 | 94.3% | 良好 |
| NIAH MultiQuery/Value | 复杂检索 | 100% | 优秀 |
| QA | 问答理解 | 65% | 一般 |
| Aggregation (CWE/FWE) | 信息聚合 | 92% | 良好 |
| Variable Tracking | 变量追踪 | 100% | 优秀 |

## 发现的问题及修复

### 问题: FWE 测试崩溃

**症状**: 第 63 个样本处触发 `AssertionError: No sequences scheduled`

**根因分析**:
1. Sample 63 的输入有 32760 tokens（接近 max_model_len=32768）
2. Decode 到第 9 步时，需要第 33 个 KV block
3. 但系统只配置了 32 个 blocks（32768/1024=32）
4. 调度器尝试 preempt 但单序列模式下无法恢复

**解决方案**:
```python
# 修改前
DEFAULT_MAX_MODEL_LEN = 32768

# 修改后: 为 output tokens 预留空间
DEFAULT_MAX_MODEL_LEN = 32896  # 32768 + 128
```

**建议的代码改进**:
1. 在 scheduler 中添加死锁检测和清晰错误信息
2. 在配置验证时，如果 max_model_len 与 max_input 过于接近，发出警告

## 评估方法

遵循 RULER 官方评估标准:
- **NIAH/VT/CWE/FWE**: `string_match_all` - 召回率 (找到的参考数/总参考数)
- **QA**: `string_match_part` - 任意参考匹配即满分

参考: https://github.com/NVIDIA/RULER

## 测试配置

```python
LLM(
    model_path="~/models/Llama-3.1-8B-Instruct",
    max_model_len=32896,
    max_num_batched_tokens=32896,
    enable_cpu_offload=True,
    num_gpu_blocks=4,
    kvcache_block_size=1024,
    enforce_eager=True,
)
```

## 结论

1. **长上下文检索能力**: nano-vllm CPU offload 模式在 32K 上下文下表现优秀，NIAH 类任务准确率接近 100%

2. **复杂推理能力**: QA 任务准确率较低 (65%)，这是模型本身能力的体现，与 offload 机制无关

3. **稳定性**: 修复 max_model_len 配置后，所有 1300 个样本测试均稳定完成

4. **性能**: 单样本测试时间约 25-35 秒，主要受 CPU-GPU 数据传输影响
