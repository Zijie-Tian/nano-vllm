# BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

BLASST 是一种针对长上下文推理优化的动态稀疏 Attention 策略。它利用 **Online Softmax** 的中间状态，在计算过程中动态决定是否跳过不重要的 KV 块。

---

## 1. 核心原理

BLASST 的核心思想是：如果当前 KV 块（Local Block）对最终 Softmax 结果的贡献极小，则可以跳过其 Attention 计算。

### 1.1 在线 Softmax 状态
在计算 $i$-th 块时，我们维护两个关键变量：
- $m_i$: 前 $i-1$ 个块的最大 Logit 值 (Running Maximum)。
- $l_i$: 前 $i-1$ 个块的归一化累加和 (Running Sum)。

对于当前第 $j$ 个块，其最大 Logit 值为 $m_{local}$。

### 1.2 跳过条件 (Skip Condition)
BLASST 定义了一个阈值 $\lambda \in (0, 1]$。当满足以下条件时，跳过该块的计算：
$$m_{local} - m_i < \ln(\lambda)$$

该条件意味着当前块的最大贡献相对于历史最大值的比例低于 $\lambda$，因此对输出的影响可以忽略不计。

### 1.3 动态阈值策略
为了平衡精度和速度，BLASST 使用随序列长度 $L$ 变化的动态阈值：
$$\lambda = \frac{a}{L}$$
- **短序列**：$L$ 较小时，$\lambda$ 较大（接近 1.0），计算更保守（少跳过），保证精度。
- **长序列**：$L$ 较大时，$\lambda$ 较小，计算更激进（多跳过），大幅提升性能。
- 参数 $a$ 默认为 `16384`，在 32K 长度下 $\lambda \approx 0.5$。

---

## 2. 系统架构

### 2.1 Triton Kernel 实现 (`nanovllm/ops/blasst_chunked_prefill.py`)
BLASST 的核心逻辑实现在 `blasst_chunked_prefill` Triton kernel 中：
- **Block-wise 遍历**：Kernel 内部遍历所有 KV 块。
- **动态判断**：在加载 $V$ 矩阵之前，先加载 $K$ 并计算 $m_{local}$。如果满足跳过条件，则直接进入下一个块，节省 $V$ 的读取和后续的 Dot Product + Softmax 更新开销。
- **LSE 输出**：计算完成后返回每个 Query Token 的 Log-Sum-Exp (LSE)，用于与后续块（如 Causal Chunk）进行 Online Merge。

### 2.3 密度监控 (Density Tracking)
从 v1.1 版本开始，BLASST 支持实时的密度监控：
- **实现机制**：使用 **Mask Buffer** 方案。算子层可选地接收一个布尔掩码矩阵，记录每个子块的计算/跳过决策。
- **日志输出**：`BLASSTPolicy` 默认在 **第 0 层 (Layer 0)** 统计并打印每个 Chunk 的平均密度。
- **开销**：Mask Buffer 写入开销极低 (~0.5%)，且在多 SM 并行环境下是线程安全的。

---

## 3. 配置参数
...
| `blasst_granularity` | 128 | 剪枝判断的 Token 粒度 |

### 3.1 调优参考
根据实验数据（详见 [BLASST 性能分析报告](blasst_performance_analysis.md)）：
- **λ = 0.3**: 高密度 (~94%)，最保守，适合追求极致精度。
- **λ = 0.5**: 中密度 (~85%)，默认均衡配置。
- **λ = 0.8**: 低密度 (~65%)，大幅剪枝，在 32K 任务下仍能保持 100% 精度。


| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blasst_a` | 16384 | 动态阈值公式 $\lambda = a / L$ 中的分子 |
| `blasst_fixed_lambda` | `None` | 如果设置（如 0.5），则忽略动态公式，使用固定阈值 |
| `blasst_granularity` | 128 | 剪枝判断的 Token 粒度 |

---

## 4. 使用指南

### 4.1 运行命令
使用 `--sparse-policy BLASST` 启用该策略。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy BLASST
```

### 4.2 性能调优
- **追求速度**：降低 `blasst_a` 或 `blasst_fixed_lambda`（如 0.3）。
- **追求精度**：提高 `blasst_a` 或使用较大的 `blasst_fixed_lambda`（如 0.8）。

---

## 5. 参考资料
- **论文**: *BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding* (arXiv:2512.12087)
- **实现文件**:
    - `nanovllm/ops/blasst_chunked_prefill.py` (Triton Kernel)
    - `nanovllm/kvcache/sparse/blasst.py` (Sparse Policy)
