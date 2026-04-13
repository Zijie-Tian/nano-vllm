# RULER Chunked Attention Debugging

## 状态：已解决（环境问题）

## 问题描述

RULER benchmark (NIAH single needle) 在 nano-vllm 中失败：
- **compass 环境**: ✅ 正常工作
- **triattention 环境**: ❌ 输出垃圾文本

### 测试数据
- Model: Llama-3.1-8B-Instruct
- Context: ~32K tokens
- Expected: 检索到正确数字（如 "4214348"）
- triattention 输出: `The is. The is. The is. The is...`

## 根因

**cuDNN 版本差异**导致计算结果不同：

| 环境 | torch 版本 | cuDNN 版本 | RULER 结果 |
|------|-----------|------------|------------|
| **compass** | 2.4.0+cu121 | 90100 | ✅ PASS |
| **triattention** | 2.4.0+cu121 | 92000 | ❌ FAIL |

cuDNN 92000 是官方 PyTorch 2.4.0 wheel 自带的，而 compass 的 cuDNN 90100 是自定义构建的。

## 解决方案

### 推荐：使用 compass 环境进行开发测试

```bash
source /mnt/data/tzj/anaconda3/etc/profile.d/conda.sh
conda activate compass
```

### 替代：自定义 PyTorch 构建

如果必须在 triattention 环境工作，需要从源码编译 PyTorch，指定 cuDNN 9.1 版本。

## 验证方法

```bash
# compass 环境运行测试
source /mnt/data/tzj/anaconda3/etc/profile.d/conda.sh
conda activate compass
PYTHONPATH=/mnt/data/tzj/Code/nano-vllm:$PYTHONPATH python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 10 \
    --max-model-len 40960 \
    --enable-offload
```

## 调试记录

### 已排除的原因

1. ✅ nano-vllm 代码本身正确（compass 环境中通过）
2. ✅ flash_attn_with_lse 函数正确（两环境输出相同）
3. ✅ PyTorch SDPA 计算正确（强制使用 SDPA 仍失败）
4. ✅ 权重加载正确
5. ✅ merge_attention_outputs 逻辑正确

### 关键证据

LSE 值在两个环境中有显著差异：
- compass: `prev_lse mean ≈ 6.09`
- triattention: `prev_lse mean ≈ 9.60`

差异达 3.5，可能由 cuDNN 92000 的某些数值优化导致。

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-04-11 | 创建文档，开始调试 |
| 2026-04-11 | 发现问题在环境 (cuDNN 版本差异) |
| 2026-04-13 | 确认问题根因，更新文档 |
