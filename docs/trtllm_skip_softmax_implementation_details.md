# TensorRT-LLM Skip Softmax (BLASST) 实现细节

本文档详细描述了 TensorRT-LLM 中 **Skip Softmax**（内部代号 **BLASST**: Block-wise Lasso Adaptive Softmax）机制的实现细节。该技术旨在通过动态跳过对结果贡献微小的 KV Block，加速长文本场景下的 LLM 推理。

---

## 1. 算法核心原理

### 1.1 数学背景
Skip Softmax 利用了 Softmax 函数的指数特性。在 FlashAttention 的分块计算过程中，对于每一个 Query，内核会维护一个运行中的 **全局最大 Logit** ($m_{global}$)。对于一个新的 KV Block，计算其 **局部最大 Logit** ($\tilde{m} = \max(Q \cdot K_{block}^T)$)。

### 1.2 跳过准则 (Skip Condition)
算法定义了一个阈值 $\lambda$（即 `threshold_scale_factor`），当满足以下条件时，该 KV Block 被视为“可忽略”：

$$ \exp(\tilde{m} - m_{global}) < \frac{\lambda}{L} $$

其中 $L$ 是当前的序列长度。

### 1.3 优化的内容
当触发跳过条件时，内核将执行以下操作：
1. **跳过 Softmax 计算**：不再进行指数运算和累加。
2. **跳过 BMM2 ($P \cdot V$)**：不进行注意力权重与 Value 矩阵的乘法。
3. **跳过显存加载 (V-Loading)**：在解码阶段，这是最核心的优化，内核直接不从 HBM 读取该 Block 的 $V$ 数据，显著降低带宽压力。

**注意**：BMM1 ($Q \cdot K^T$) 不会被跳过，因为需要计算 $\tilde{m}$ 来进行判断。

---

## 2. 硬件与内核实现

Skip Softmax 是一个“内核级”优化，逻辑完全实现在 CUDA/Triton 内核内部，无需修改模型结构。

### 2.1 Hopper (SM90) 实现
- **Prefill (预填充)**：集成在 `fmha_v2` 内核中。
- **Decode (解码)**：集成在 `XQA`（Masked MHA）内核中。通过 `xqaDispatcher` 将阈值传递给底层内核。

### 2.2 Blackwell (SM100) 实现
- 集成在 `trtllm-gen` 内核中。
- **内核哈希 (Kernel Trait)**：在 `fmhaKernels.h` 中，`skipsSoftmax` 标志位（Bit 56）用于在构建时选择支持 Skip Softmax 的 `cubin` 版本。

### 2.3 数据路径
1. **API 层**：用户通过 `SkipSoftmaxAttentionConfig` 设置 `threshold_scale_factor`。
2. **Runtime 层**：参数通过 `XQAParams` 或 `Fused_multihead_attention_params_v2` 传递给底层。
3. **Kernel 层**：CUDA 内核根据 `params.skip_softmax_threshold_scale_factor` 执行分支判断。

---

## 3. 配置与校准机制

由于 $\lambda$ 的选取对精度和速度的平衡至关重要，TensorRT-LLM 提供了两种配置方式：

### 3.1 手动配置
直接在 YAML 或 LLM API 中设置 `threshold_scale_factor`。通常 Prefill 和 Decode 需要不同的阈值。

### 3.2 自动校准 (ModelOpt 路径)
通过 NVIDIA ModelOpt 进行离线校准，模型导出的 `config.json` 会包含针对该模型的指数拟合参数 $a$ 和 $b$。
TRT-LLM 会根据用户期望的**目标稀疏率** ($S$) 自动计算阈值：

$$ \text{threshold} = a \cdot \exp(b \cdot S) $$

这种方式极大地降低了用户的使用门槛，实现了近乎无损（Near-lossless）的加速。

---

## 4. 动态参数 $L$ 与工程权衡

在判断准则 $\exp(\tilde{m} - m_{global}) < \frac{\lambda}{L}$ 中，参数 $L$ 的处理是算法精度的核心。

### 4.1 $L$ 的定义与动态增长
- **定义**：$L$ 代表当前 Query 能够看到的 **总 KV 序列长度**。
- **Decoding 阶段**：每生成一个 Token，$L$ 就会自增 1。这意味着判断门槛 $\frac{\lambda}{L}$ 会随时间推移而**逐渐变严**。这种设计是为了补偿 Softmax 分母项随序列增长而导致的平均贡献下降，确保在不同长度下判断“重要性”的标准是连贯的。

### 4.2 Chunked Prefill 下的 $L$ 变化
在分块预填充（Chunked Prefill）模式下，$L$ 的行为具有阶段性：
- **分块演进**：当处理第 $N$ 个 Chunk 时，$L$ 取当前已处理的总 KV 长度。例如，Chunk 大小为 4k，处理第二个 Chunk 时 $L \approx 8192$。
- **阈值变动**：这意味着前面的 Chunk 面对的 $L$ 较小，阈值相对较松；后面的 Chunk 面对的 $L$ 较大，阈值相对较严。

### 4.3 分块与全量预填充的一致性辩证
**理论差异**：从数学上讲，由于分块时前面的 Chunk 采用的 $L$ 较小，其跳过 Block 的门槛比不分块（直接用总长 $L$）时要松。
**工程鲁棒性**：TensorRT-LLM 通过以下机制确保了精度：
1. **Softmax 指数的陡峭性**：真正重要的 Block（$\exp \approx 1$）远大于任何合理设置的阈值。
2. **模型校准 (ModelOpt)**：校准过程模拟了真实的分块推理场景，拟合出的参数已包含对这种长度变化的补偿。
3. **因果掩码约束**：由于 Query 只能看到之前的 Token，这种局部长度 $L$ 的变化在很大程度上符合 Softmax 的计算逻辑。

---

## 5. 性能预期

- **加速比**：在长文本（128k+）场景下，端到端加速可达 **1.2x - 1.4x**。
- **显存占用**：零额外显存开销。
- **兼容性**：支持 FP8 注意力、KV Cache 复用、MLA 和 GQA 结构。

---

## 参考资料
- Paper: [BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
- Blog: [Accelerating Long-Context Inference with Skip Softmax Attention](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)
