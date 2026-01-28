# 1M+ 上下文长度模型列表

本文档收集了 Hugging Face 上支持 1M (1,048,576) 及以上上下文长度的开源模型。

> 更新时间: 2026-01-28

---

## 一、纯语言模型 (≤10B 参数)

### 1. 官方原版模型

| 厂商 | 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|------|--------|------|--------|------|
| **Qwen** | Qwen2.5-7B-Instruct-1M | 1M | 7B | 69.3K | [HF](https://hf.co/Qwen/Qwen2.5-7B-Instruct-1M) |
| **THUDM** | GLM-4-9B-Chat-1M | 1M | 9B | 5.0K | [HF](https://hf.co/zai-org/glm-4-9b-chat-1m) |
| **InternLM** | InternLM2.5-7B-Chat-1M | 1M | 7B | 322 | [HF](https://hf.co/internlm/internlm2_5-7b-chat-1m) |
| **NVIDIA** | Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct | 1M | 8B | 2.9K | [HF](https://hf.co/nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct) |
| **LWM** | LWM-Text-1M | 1M | 7B | 75 | [HF](https://hf.co/LargeWorldModel/LWM-Text-1M) |
| **LWM** | LWM-Text-Chat-1M | 1M | 7B | 3.0K | [HF](https://hf.co/LargeWorldModel/LWM-Text-Chat-1M) |

### 2. Gradient AI 扩展系列 (基于 Llama 3)

| 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|--------|------|--------|------|
| Llama-3-8B-Instruct-Gradient-1048k | **1M** | 8B | 44.8K | [HF](https://hf.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) |
| Llama-3-8B-Instruct-Gradient-4194k | **4M** | 8B | 9 | [HF](https://hf.co/gradientai/Llama-3-8B-Instruct-Gradient-4194k) |

### 3. 社区衍生版本 (Abliterated)

| 模型 | 上下文 | 基础模型 | 下载量 | 链接 |
|------|--------|----------|--------|------|
| Qwen2.5-7B-Instruct-1M-abliterated | 1M | Qwen2.5-7B | 375 | [HF](https://hf.co/huihui-ai/Qwen2.5-7B-Instruct-1M-abliterated) |
| Nemotron-8B-UltraLong-1M-Abliterated | 1M | Nemotron-8B | 46 | [HF](https://hf.co/SicariusSicariiStuff/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct_Abliterated) |

---

## 二、视觉-语言模型 (≤10B 参数)

### Qwen3 VL 系列

#### Instruct 版本

| 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|--------|------|--------|------|
| Qwen3-VL-2B-Instruct-1M-GGUF | 1M | 2B | 824 | [HF](https://hf.co/unsloth/Qwen3-VL-2B-Instruct-1M-GGUF) |
| Qwen3-VL-4B-Instruct-1M-GGUF | 1M | 4B | 936 | [HF](https://hf.co/unsloth/Qwen3-VL-4B-Instruct-1M-GGUF) |
| Qwen3-VL-8B-Instruct-1M-GGUF | 1M | 8B | 962 | [HF](https://hf.co/unsloth/Qwen3-VL-8B-Instruct-1M-GGUF) |

#### Thinking 推理版本

| 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|--------|------|--------|------|
| Qwen3-VL-2B-Thinking-1M-GGUF | 1M | 2B | 808 | [HF](https://hf.co/unsloth/Qwen3-VL-2B-Thinking-1M-GGUF) |
| Qwen3-VL-4B-Thinking-1M-GGUF | 1M | 4B | 666 | [HF](https://hf.co/unsloth/Qwen3-VL-4B-Thinking-1M-GGUF) |
| Qwen3-VL-8B-Thinking-1M-GGUF | 1M | 8B | 4.6K | [HF](https://hf.co/unsloth/Qwen3-VL-8B-Thinking-1M-GGUF) |

---

## 三、推荐模型 (≤10B)

| 用途 | 推荐模型 | 理由 |
|------|----------|------|
| **通用对话** | Qwen2.5-7B-Instruct-1M | 官方支持，RULER 93.1分，Apache 2.0 |
| **中英双语** | GLM-4-9B-Chat-1M | 清华出品，中文优化 |
| **最长上下文** | Llama-3-8B-Gradient-4194k | 支持 4M 上下文 |
| **多模态** | Qwen3-VL-8B-Thinking-1M | 视觉理解 + 推理能力 |
| **无审查** | Qwen2.5-7B-Instruct-1M-abliterated | 移除安全限制 |

---

## 四、VRAM 需求参考

| 模型规模 | 1M 上下文 VRAM | 备注 |
|----------|----------------|------|
| 7B (FP16) | ~120GB | 需多卡 |
| 7B (INT4) | ~40GB | 单卡 A100 可行 |
| 8B (FP16) | ~130GB | 需多卡 |
| 9B (FP16) | ~140GB | 需多卡 |

---

## 五、技术对比

| 模型系列 | 扩展技术 | RULER 得分 | 许可证 |
|---------|---------|------------|--------|
| Qwen2.5-1M | Dual Chunk Attention | 93.1 | Apache 2.0 |
| GLM-4-1M | - | 89.9 | 自定义 |
| Gradient-Llama | 渐进式扩展 | - | Llama 3 |
| Nemotron-1M | NVIDIA 训练 | - | CC-BY-NC-4.0 |
| LWM-1M | RingAttention | - | 开源 |

---

---

# 附录：大参数模型 (>10B)

> 以下模型参数量超过 10B，需要更多计算资源。

## A. 纯语言模型 (>10B)

### 官方模型

| 厂商 | 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|------|--------|------|--------|------|
| **Qwen** | Qwen2.5-14B-Instruct-1M | 1M | 14B | 4.7K | [HF](https://hf.co/Qwen/Qwen2.5-14B-Instruct-1M) |
| **MiniMax** | MiniMax-Text-01 | 1M | 456B MoE | 721 | [HF](https://hf.co/MiniMaxAI/MiniMax-Text-01) |
| **Gradient** | Llama-3-70B-Instruct-Gradient-1048k | 1M | 70B | 9 | [HF](https://hf.co/gradientai/Llama-3-70B-Instruct-Gradient-1048k) |

### Qwen3 Coder 系列 (MoE)

| 模型 | 上下文 | 总参数/激活参数 | 下载量 | 链接 |
|------|--------|-----------------|--------|------|
| Qwen3-Coder-30B-A3B-Instruct-1M-GGUF | 1M | 30B / 3B | 13.1K | [HF](https://hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF) |
| Qwen3-Coder-480B-A35B-Instruct-1M | 1M | 480B / 35B | 50 | [HF](https://hf.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-1M) |
| Qwen3-Coder-480B-A35B-Instruct-1M-GGUF | 1M | 480B / 35B | 1.7K | [HF](https://hf.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF) |
| Qwen3-Coder-42B-A3B-TOTAL-RECALL-1M | 1M | 42B / 3B | - | [HF](https://hf.co/DavidAU/Qwen3-Coder-42B-A3B-Instruct-TOTAL-RECALL-MASTER-CODER-M-1million-ctx) |

### 社区衍生版本

| 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|--------|------|--------|------|
| Qwen2.5-14B-Instruct-1M-abliterated | 1M | 14B | 147 | [HF](https://hf.co/huihui-ai/Qwen2.5-14B-Instruct-1M-abliterated) |

---

## B. 视觉-语言模型 (>10B)

### Meta Llama 4 系列 (MoE 多模态)

| 模型 | 上下文 | 总参数/激活参数 | 下载量 | 链接 |
|------|--------|-----------------|--------|------|
| Llama-4-Scout-17B-16E-Instruct | **10M** | 109B / 17B | 180K | [HF](https://hf.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |
| Llama-4-Maverick-17B-128E-Instruct | **1M** | 400B / 17B | 32.6K | [HF](https://hf.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |
| Llama-4-Scout-17B-16E | 10M | 109B / 17B | 8.4K | [HF](https://hf.co/meta-llama/Llama-4-Scout-17B-16E) |
| Llama-4-Maverick-17B-128E | 1M | 400B / 17B | 368 | [HF](https://hf.co/meta-llama/Llama-4-Maverick-17B-128E) |
| Llama-4-Maverick-17B-128E-Instruct-FP8 | 1M | 400B / 17B | 29.6K | [HF](https://hf.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) |

### Qwen3 VL 大模型系列

#### Dense 模型

| 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|--------|------|--------|------|
| Qwen3-VL-32B-Instruct-1M-GGUF | 1M | 32B | 1.2K | [HF](https://hf.co/unsloth/Qwen3-VL-32B-Instruct-1M-GGUF) |
| Qwen3-VL-32B-Thinking-1M-GGUF | 1M | 32B | 452 | [HF](https://hf.co/unsloth/Qwen3-VL-32B-Thinking-1M-GGUF) |

#### MoE 模型

| 模型 | 上下文 | 总参数/激活参数 | 下载量 | 链接 |
|------|--------|-----------------|--------|------|
| Qwen3-VL-30B-A3B-Instruct-1M-GGUF | 1M | 30B / 3B | 821 | [HF](https://hf.co/unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF) |
| Qwen3-VL-30B-A3B-Thinking-1M-GGUF | 1M | 30B / 3B | 944 | [HF](https://hf.co/unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF) |
| Qwen3-VL-235B-A22B-Instruct-1M-GGUF | 1M | 235B / 22B | 581 | [HF](https://hf.co/unsloth/Qwen3-VL-235B-A22B-Instruct-1M-GGUF) |
| Qwen3-VL-235B-A22B-Thinking-1M-GGUF | 1M | 235B / 22B | 733 | [HF](https://hf.co/unsloth/Qwen3-VL-235B-A22B-Thinking-1M-GGUF) |

#### MXFP4 量化版本

| 模型 | 上下文 | 规模 | 下载量 | 链接 |
|------|--------|------|--------|------|
| Qwen3-VL-30B-A3B-Instruct-1M-MXFP4_MOE-GGUF | 1M | 30B MoE | 689 | [HF](https://hf.co/noctrex/Qwen3-VL-30B-A3B-Instruct-1M-MXFP4_MOE-GGUF) |
| Qwen3-VL-30B-A3B-Thinking-1M-MXFP4_MOE-GGUF | 1M | 30B MoE | 565 | [HF](https://hf.co/noctrex/Qwen3-VL-30B-A3B-Thinking-1M-MXFP4_MOE-GGUF) |
| Qwen3-VL-235B-A22B-Instruct-1M-MXFP4_MOE-GGUF | 1M | 235B MoE | 136 | [HF](https://hf.co/noctrex/Qwen3-VL-235B-A22B-Instruct-1M-MXFP4_MOE-GGUF) |
| Qwen3-VL-235B-A22B-Thinking-1M-MXFP4_MOE-GGUF | 1M | 235B MoE | 244 | [HF](https://hf.co/noctrex/Qwen3-VL-235B-A22B-Thinking-1M-MXFP4_MOE-GGUF) |

---

## 统计汇总

| 类别 | ≤10B 模型数 | >10B 模型数 | 最大上下文 |
|------|-------------|-------------|-----------|
| 纯语言模型 | 10 | 8 | 4M |
| 视觉-语言模型 | 6 | 14 | 10M |
| **合计** | **16** | **22** | **10M** |

---

## 参考资源

- [Qwen2.5-1M 官方博客](https://qwenlm.github.io/blog/qwen2.5-1m/)
- [LongRoPE 论文](https://huggingface.co/papers/2402.13753)
- [InfiniteHiP 论文](https://huggingface.co/papers/2502.08910)
- [Top LLMs for Long Context Windows](https://www.siliconflow.com/articles/en/top-LLMs-for-long-context-windows)
