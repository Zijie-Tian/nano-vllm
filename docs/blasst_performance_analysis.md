# BLASST Performance Analysis and Sparsity Tracking

This document summarizes the findings from benchmarking and implementing sparsity tracking for the BLASST (Dynamic Blocked Attention Sparsity via Softmax Thresholding) operator.

## 1. Sparsity Tracking Methodologies

We evaluated two primary methods for tracking block-level sparsity (skipping rate) within the Triton kernel.

### 1.1 Atomic Add
- **Mechanism**: Use `tl.atomic_add` inside the kernel to increment a global counter whenever a block is skipped.
- **Overhead**: Extremely low (~0.1% - 0.2%).
- **Issues**: Highly susceptible to statistical errors when combined with Triton's `autotune`. Autotune runs multiple kernel configurations, often leading to double or triple counting. It can also suffer from race conditions or overflows if not carefully managed across many SMs.

### 1.2 Mask Buffer (Selected Solution)
- **Mechanism**: Provide an optional `[grid_0, grid_1, num_blocks]` buffer to the kernel. Each Triton program writes its own compute/skip decisions (1 or 0) to its dedicated slice of the buffer.
- **Overhead**: Very low (~0.5%).
- **Benefits**:
    - **Thread-safe**: No competition between SMs as each writes to unique memory addresses.
    - **Robust**: Unaffected by `autotune` double-counting (as it overwrites instead of increments).
    - **Detailed**: Allows for fine-grained analysis of *which* specific blocks were skipped, not just the total count.
- **Implementation**: Integrated into `nanovllm/ops/blasst_chunked_prefill.py` and utilized by `BLASSTPolicy` at Layer 0.

## 2. Sensitivity Analysis: Lambda ($\lambda$) vs. Density

The BLASST skip condition is defined as: `local_max - running_max < ln(λ)`. We tested various fixed $\lambda$ values using GLM-4-9B-Chat-1M on a 32K context task.

| Lambda ($\lambda$) | Average Density | Sparsity (Skipped) | Observations |
| :--- | :--- | :--- | :--- |
| **0.3** | ~94% | ~6% | Conservative pruning. High density ensures high precision but limited speedup. |
| **0.5** | ~86% | ~14% | Balanced setting. Significant pruning with 100% accuracy. |
| **0.8** | ~68% | ~32% | Aggressive pruning. Still maintained 100% accuracy on NIAH. |

### Key Insight
- **Inverse Relationship**: A **larger** $\lambda$ leads to **lower** density (more skipping). This is because `ln(λ)` becomes closer to 0, making the skip condition `local_max - running_max < ln(λ)` easier to satisfy.
- **Stable Precision**: Even at ~30% skipping rate ($\lambda=0.8$), the model correctly retrieved information in the Needle-In-A-Haystack (NIAH) test.

## 3. Real Data Statistics (GLM-4-9B)

Using real KV cache traces from long-context sequences, we observed the following density patterns at $\lambda=0.5$:

- **16K Context**: ~9.8% Density (90%+ skipping).
- **128K Context**: ~4.8% Density (95%+ skipping).

*Note: Densities observed in synthetic Ruler tasks (85-90%) are generally higher than in real-world conversational data, where attention is often even more sparse.*

## 4. Operational Recommendations

1. **Monitoring**: Keep Mask Buffer tracking enabled at Layer 0 for continuous observability.
2. **Tuning**:
    - For **Safety**: Use $\lambda \in [0.3, 0.5]$.
    - For **Throughput**: Use $\lambda \in [0.7, 0.8]$.
3. **Dynamic Strategy**: The default formula $\lambda = a / L$ (with $a=16384$) effectively scales $\lambda$ down as sequence length $L$ increases, which we now know **increases density** (makes skipping harder) for very long sequences to preserve precision. This may need further tuning based on the density observations above.
