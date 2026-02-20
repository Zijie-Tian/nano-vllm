# Solution 5: KV Hidden-Dim Reorder（KV 维度重排）

## Executive Summary

**核心思想**：在同一 KV group 内，对 Q 和 K 的 d_head 维度同时施加相同的 permutation，将 K 中"天然 dense 的维度"（即在所有 token 位置上数值差异小、方差低的维度）排到 d_head 空间的前端，将"天然 sparse 的维度"（高方差、高辨别力）排到后端。该 permutation 对 attention 的数值计算**完全等价**。在此基础上，chunked prefill 的 block selection 路由仅使用 K 的**后端 sparse 维度**参与估计，去除 dense 维度带来的常数偏置噪声，提升路由质量，从而减少实际加载的 KV block 数。

**可行性评分**：8/10

**关键优势**：
- 对 attention 数值输出**完全数学等价**（无需验证 accuracy drop）
- 无需修改模型权重的"语义"，仅对列顺序做一次性物理排列
- 路由质量提升，dense head 的 KV block 加载量可大幅减少
- 不改变 GQA head-to-group 映射关系，代码侵入小

**主要风险**：
- Dense/sparse 维度的分界点 `d_dense` 需要 profiling 确定
- Block selection 路由从"全维度估计"改为"sparse 维度估计"是近似操作，routing recall 需验证
- 不同序列长度、任务类型下，维度分界的稳定性需验证

---

## 1. 问题分析：为什么 Dense 维度会破坏路由？

### 1.1 Chunked Prefill 的 Block Selection 原理

在 chunked prefill + CPU offload 场景下，推理系统需要在加载完整 KV cache 之前，**估计哪些 K blocks 对当前 Q 是重要的**。当前 xattn 的估计方法（以 block-level max score 为例）：

$$
\text{score}(q, b) = \max_{k \in \text{block } b} \frac{Q_h[q, :] \cdot K_g[k, :]}{\sqrt{d_{\text{head}}}}
$$

选择 score 最高的 top-B 个 block 加载，跳过其余 block。

### 1.2 Dense 维度如何污染路由信号

将 K 的 d_head 维度分为两类：

**Dense 维度** $i \in \mathcal{D}$：对应的 $K_g[:, i]$ 在所有 token 位置上数值差异极小，近似为常数：
$$K_g[k, i] \approx c_i \quad \forall k \in [0, S)$$

**Sparse 维度** $i \in \mathcal{S}$：对应的 $K_g[:, i]$ 在不同 token 位置上差异显著：
$$\text{Var}_k(K_g[k, i]) \gg 0$$

对于 dense 维度 $i \in \mathcal{D}$，其对 block score 的贡献为：

$$\Delta\text{score}_{i}(q, b) = Q_h[q, i] \cdot \frac{1}{|b|} \sum_{k \in b} K_g[k, i] \approx Q_h[q, i] \cdot c_i$$

这个值**对所有 block $b$ 相同**——它为每个 block 加上了相同的偏置，完全不携带任何 block 判别信息，只会将所有 block 的分数整体抬高（或降低）。

对于 sparse 维度 $i \in \mathcal{S}$，其贡献：

$$\Delta\text{score}_{i}(q, b) = Q_h[q, i] \cdot \frac{1}{|b|} \sum_{k \in b} K_g[k, i]$$

由于 $K_g[:, i]$ 在不同 token 有差异，**不同 block $b$ 的得分不同**，这才是真正的判别信号。

**结论**：Dense 维度对 block selection 的贡献是纯噪声（常数偏置），会掩盖 sparse 维度提供的有效判别信号，导致：
- 所有 block 的 score 被 dense 维度均匀抬高 → top-B 选择接近随机
- Dense head 的实际 block 需求被高估 → 加载过多 KV block

---

## 2. 核心机制：等价 Permutation + 路由改进

### 2.1 维度重排的数学等价性

**命题**：对于 KV group $g$ 内的所有 Q head $h$，同时对 $Q_h$ 和 $K_g$ 施加相同的列置换 $P$，attention 输出完全不变。

**证明**：

设原始 attention score 矩阵（shape $[S_q, S_k]$）：
$$A_h = \frac{Q_h K_g^\top}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{i=0}^{d-1} Q_h[:, i] \otimes K_g[:, i]$$

施加 permutation $P \in \mathbb{R}^{d \times d}$（置换矩阵，$P^\top P = I$）后：
$$Q_h^{\text{new}} = Q_h P, \quad K_g^{\text{new}} = K_g P$$

新的 attention score：
$$A_h^{\text{new}} = \frac{Q_h^{\text{new}} (K_g^{\text{new}})^\top}{\sqrt{d}} = \frac{(Q_h P)(K_g P)^\top}{\sqrt{d}} = \frac{Q_h P P^\top K_g^\top}{\sqrt{d}} = \frac{Q_h K_g^\top}{\sqrt{d}} = A_h \quad \square$$

由于 $A_h$ 不变，attention output $O_h = \text{softmax}(A_h) V_g$ 也完全不变（V 不需要调整）。

**Weight-level 实现**：

对权重矩阵执行一次性列排列：
$$W_q^{\text{new}}[:, h \cdot d : (h+1) \cdot d] = W_q[:, h \cdot d : (h+1) \cdot d] \cdot P, \quad \forall h \in \text{group } g$$
$$W_k^{\text{new}}[:, g \cdot d : (g+1) \cdot d] = W_k[:, g \cdot d : (g+1) \cdot d] \cdot P$$

$W_v$ 和 $W_o$ **无需修改**（因为 $O_h$ 不变，$W_o$ 所接收的输入不变）。

### 2.2 重排后的维度结构

选定置换 $P$ 使得：

$$K_g^{\text{new}}[:, i] = \begin{cases} \text{dense 维度（低方差）} & i < d_{\text{dense}} \\ \text{sparse 维度（高方差）} & i \geq d_{\text{dense}} \end{cases}$$

重排后，$K_g^{\text{new}}$ 的维度结构：

```
| <---- d_dense ----> | <-------- d_head - d_dense --------> |
  Dense dims（低方差）   Sparse dims（高方差，判别信号强）
  K_g[:, :d_dense]      K_g[:, d_dense:]
```

同样，所有 Q head 也得到了相同的维度排列：
$$Q_h^{\text{new}}[:, :d_\text{dense}] \leftrightarrow \text{与 dense K 维度对应}$$
$$Q_h^{\text{new}}[:, d_\text{dense}:] \leftrightarrow \text{与 sparse K 维度对应}$$

### 2.3 改进后的 Block Selection 路由

在 chunked prefill 的 block selection 阶段，仅使用 sparse 维度参与路由估计：

$$\text{score}^{\text{new}}(q, b) = \max_{k \in b} \frac{Q_h^{\text{new}}[q, d_\text{dense}:] \cdot K_g^{\text{new}}[k, d_\text{dense}:]^\top}{\sqrt{d_\text{head} - d_\text{dense}}}$$

与原始路由的对比：

| 方案 | 路由使用的维度 | 信噪比 |
|------|---------------|--------|
| 原始 | 全部 $d_\text{head}$ 维 | 低（dense 维度引入常数偏置） |
| 重排后 | 仅 $d_\text{head} - d_\text{dense}$ 维 sparse 部分 | 高（去除常数偏置噪声） |

实际 attention 计算（在已加载的 block 上）仍使用**所有 $d_\text{head}$ 维度**，保持数值等价。

---

## 3. Dense 维度的量化定义与 Profiling 算法

### 3.1 维度 Density Score 定义

对于 KV group $g$，其 K 矩阵在若干 calibration 样本上采集得到。第 $i$ 个维度的 density score 可由以下指标定义（从弱到强）：

**方法 A：归一化方差（Coefficient of Variation, CV）**

$$\text{CV}_i = \frac{\text{Std}_k(K_g[:, i])}{\text{Mean}_k(|K_g[:, i]|) + \epsilon}$$

低 CV → 各 token 的 K 值接近常数 → dense 维度

**方法 B：Block Discriminability Score**

对于 block $b$ 和维度 $i$，计算该维度下 block-level K 统计的离散度：

$$\text{DS}_i = \text{Var}_b\left(\text{Mean}_{k \in b}(K_g[k, i])\right)$$

低 DS → 不同 block 在该维度上的均值相似 → 对 block selection 无判别力 → dense 维度

**方法 C：Routing Recall 直接测量（最准确，开销最大）**

对于维度子集 $S$，定义 routing recall：

$$\text{Recall}(S) = \frac{|\text{top-B blocks selected using } S \cap \text{top-B blocks using all dims}|}{B}$$

目标：找到最小的 $|S|$ 使得 $\text{Recall}(S) \geq \tau$（如 $\tau = 0.9$）。

**推荐使用方法 B（Block Discriminability Score）**，计算高效且与 block selection 目标直接相关。

### 3.2 Profiling 算法

```python
def profile_kv_dim_density(K_g, block_size=128):
    """
    对单个 KV group 的 K 矩阵进行维度 density profiling。

    输入:
        K_g: [S, d_head]  —— 若干 calibration 样本的拼接
        block_size: int  —— K block 的 token 数量

    输出:
        density_scores: [d_head]  —— 越低越 dense
        sorted_perm: [d_head]  —— 按 density 升序的维度排列（dense 在前）
        d_dense: int  —— 建议的 dense/sparse 分界点
    """
    S, d = K_g.shape
    n_blocks = math.ceil(S / block_size)

    # 方法 B: Block Discriminability Score
    block_means = torch.zeros(n_blocks, d)
    for b in range(n_blocks):
        b_start = b * block_size
        b_end = min((b + 1) * block_size, S)
        block_means[b] = K_g[b_start:b_end].mean(dim=0)

    # 每个维度 i，block-level 均值的方差
    # 低方差 = dense 维度
    density_scores = block_means.var(dim=0)  # [d_head]

    # 排列：density_scores 升序（最 dense 排最前）
    sorted_perm = torch.argsort(density_scores)  # [d_head]

    # 确定 d_dense：density_scores 排序后，找到明显的"跃变点"
    sorted_scores = density_scores[sorted_perm]
    # 简单启发式：找到前缀均值和后缀均值差距最大的点
    best_d_dense = 0
    best_gap = -1
    for d_cut in range(1, d):
        prefix_mean = sorted_scores[:d_cut].mean()
        suffix_mean = sorted_scores[d_cut:].mean()
        gap = suffix_mean - prefix_mean
        if gap > best_gap:
            best_gap = gap
            best_d_dense = d_cut

    return density_scores, sorted_perm, best_d_dense


def build_reorder_config(model, calibration_data, block_size=128):
    """
    对所有 layer 的所有 KV group 做 density profiling，生成 permutation 配置。

    输出格式（JSON）:
    {
      "layer_0": {
        "group_0": {"perm": [3, 0, 7, ...], "d_dense": 12},
        "group_1": {...},
        ...
      },
      ...
    }
    """
    config = {}
    for layer_idx in range(model.num_layers):
        config[f"layer_{layer_idx}"] = {}
        for g in range(model.num_kv_heads):
            # 收集 calibration 样本下该 group 的 K 矩阵
            K_g_samples = collect_k_samples(model, calibration_data, layer_idx, g)

            scores, perm, d_dense = profile_kv_dim_density(K_g_samples, block_size)

            config[f"layer_{layer_idx}"][f"group_{g}"] = {
                "perm": perm.tolist(),
                "d_dense": d_dense,
                "density_scores_stats": {
                    "min": scores.min().item(),
                    "max": scores.max().item(),
                    "gap": (scores[perm[d_dense:]].mean() - scores[perm[:d_dense]].mean()).item()
                }
            }
    return config
```

### 3.3 权重矩阵的一次性物理重排

```python
def apply_reorder_to_weights(model, reorder_config):
    """
    根据 profiling 配置，一次性修改模型权重（W_q, W_k）。
    W_v, W_o 不需要修改。
    """
    for layer_idx, layer_config in reorder_config.items():
        layer = model.layers[int(layer_idx.split('_')[1])]

        for g_str, group_config in layer_config.items():
            g = int(g_str.split('_')[1])
            perm = torch.tensor(group_config['perm'])
            d = model.head_dim  # d_head

            # 1. 修改 W_k（KV group g 对应的列段）
            # W_k: [d_model, num_kv_heads * d_head]
            k_start = g * d
            k_end = (g + 1) * d
            layer.W_k[:, k_start:k_end] = layer.W_k[:, k_start:k_end][:, perm]

            # 2. 修改同一 group 内所有 Q head 的 W_q 对应列段
            # W_q: [d_model, num_heads * d_head]
            n = model.num_heads // model.num_kv_heads  # Q heads per group
            for h_in_group in range(n):
                h = g * n + h_in_group
                q_start = h * d
                q_end = (h + 1) * d
                layer.W_q[:, q_start:q_end] = layer.W_q[:, q_start:q_end][:, perm]

            # W_v, W_o 不变 ✓
```

---

## 4. 改进路由的数学分析

### 4.1 为什么去掉 Dense 维度可以提升路由质量

设重排后 $K_g^{\text{new}} = [K_D \mid K_S]$，其中：
- $K_D \in \mathbb{R}^{S \times d_D}$：dense 维度，每列方差低
- $K_S \in \mathbb{R}^{S \times d_S}$：sparse 维度，每列方差高（$d_D + d_S = d_\text{head}$）

**原始路由 score**（对 block $b$ 内的某个 key position $k$）：

$$s(q, k) = \underbrace{\frac{Q_D[q] \cdot K_D[k]}{\sqrt{d}}}_{\text{常数偏置（近似）}} + \underbrace{\frac{Q_S[q] \cdot K_S[k]}{\sqrt{d}}}_{\text{判别信号}}$$

其中 $Q_D = Q_h^{\text{new}}[:, :d_D]$，$Q_S = Q_h^{\text{new}}[:, d_D:]$。

由于 $K_D[k] \approx c$（对所有 $k$ 近似相同），第一项 $\approx Q_D[q] \cdot c / \sqrt{d}$，是一个仅依赖于 $q$ 而不依赖于 $k$ 的常数。

Block-level max score：
$$\text{score}(q, b) = \max_{k \in b}\left[Q_D[q] \cdot c / \sqrt{d} + Q_S[q] \cdot K_S[k] / \sqrt{d}\right]$$
$$= \underbrace{Q_D[q] \cdot c / \sqrt{d}}_{\text{对所有 block 相同，无判别力}} + \max_{k \in b}\underbrace{Q_S[q] \cdot K_S[k] / \sqrt{d}}_{\text{实际判别信号}}$$

**改进路由 score**（仅使用 sparse 维度）：
$$\text{score}^{\text{new}}(q, b) = \max_{k \in b} \frac{Q_S[q] \cdot K_S[k]}{\sqrt{d_S}}$$

去除了常数偏置 $Q_D[q] \cdot c / \sqrt{d}$，路由完全由有效判别信号决定。

### 4.2 Routing Recall 的理论上界提升

设真实最优 top-B block 集合为 $\mathcal{B}^*$（由完整 KV 计算确定）。

**原始路由的 recall 降级原因**：
- Dense 维度为所有 block 添加相同偏置 $\delta_b \approx \text{const}$
- Block 排名主要由偏置决定，而非真实 attention mass
- 设 dense 维度的偏置方差为 $\sigma_D^2$，sparse 信号的方差为 $\sigma_S^2$
- 信噪比（SNR）$\approx \sigma_S^2 / (\sigma_D^2 + \sigma_S^2)$

**改进路由的 SNR**：$= \sigma_S^2 / \sigma_S^2 = 1$（无噪声）

路由质量改善幅度正比于 $\sigma_D^2 / \sigma_S^2$（原始 dense 噪声与 sparse 信号的比值）。该比值越大（即 dense 维度越"纯粹"），改善越显著。

### 4.3 对 Dense Head 的特殊分析

对于原来的 dense head（整体 attention 均匀分布），有两种可能：

**情形 A：Dense 是因为 dense 维度主导**
- $\sigma_D \gg \sigma_S$：dense 维度的影响掩盖了 sparse 维度的局部性
- 去除 dense 维度后，routing 会变得 sparse（发现真实的局部注意力模式）
- **结论：Dense head 可以被救回为 sparse head，大幅减少 KV 加载**

**情形 B：Dense 是因为 sparse 维度本身也很均匀**
- $\sigma_S$ 也很小：该 head 真的需要 attend 到所有 position
- 去除 dense 维度后，routing 仍然 dense
- **结论：True dense head，无法改善，需要完整加载**

通过 profiling 可以区分这两种情形。对于情形 A 的 head（数量多），改善显著；对于情形 B（真正的 dense head），至少不会变差。

---

## 5. 算法完整流程

```
阶段 0：Profiling（一次性，离线）
  ├── 收集 calibration 数据的 K 矩阵（每层每 group）
  ├── 计算每个 d_head 维度的 Block Discriminability Score
  ├── 排序：dense 维度（低分）→ 前，sparse 维度（高分）→ 后
  ├── 确定分界点 d_dense（elbow detection）
  └── 生成 reorder_config.json

阶段 1：权重重排（一次性，离线）
  ├── 对每个 layer 的每个 KV group：
  │   ├── 应用 perm 到 W_k 的对应列段
  │   └── 应用相同 perm 到同 group 内所有 Q head 的 W_q 列段
  ├── W_v, W_o 不变
  └── 保存新的 checkpoint（或存 delta）

阶段 2：改进的 Chunked Prefill 推理
  ├── 对每个 KV group：
  │   ├── Block Selection（路由）：
  │   │   使用 Q_h[:, d_dense:] @ K_g[:, d_dense:]^T / sqrt(d_head - d_dense)
  │   │   选 top-B 个 block 加载
  │   ├── KV 加载：从 CPU 加载 full K_g（所有 d_head 维）for selected blocks
  │   └── Attention 计算：使用全维度 Q_h @ K_g^T（等价于原始）
  └── 输出与原始模型完全相同（若路由完美）
```

---

## 6. COMPASS 项目集成点

### 6.1 代码结构

```
compass/
  src/
    kv_dim_reorder.py          ← 新增：profiling + 权重重排工具
3rdparty/
  nanovllm/
    nanovllm/
      kvcache/
        sparse/
          xattn_bsa.py         ← 修改：block selection 使用 sparse 维度
      ops/
        xattn.py               ← 修改：支持传入 d_dense 参数
```

### 6.2 xattn_bsa.py 的修改点

当前 block selection（伪代码）：
```python
# 当前：全维度估计
for g in range(num_kv_groups):
    score_estimate = Q_heads[g] @ K_g.T  # [n_heads, S_q, S_k]
    block_scores = max_pool(score_estimate, block_size)
    selected_blocks = top_k(block_scores, B)
```

改进后：
```python
# 改进：仅使用 sparse 维度估计
for g in range(num_kv_groups):
    d_dense = reorder_config[layer][g]['d_dense']
    # 路由仅用 sparse 维度
    Q_sparse = Q_heads[g][:, :, d_dense:]   # [n_heads, S_q, d_sparse]
    K_sparse = K_g[:, d_dense:]             # [S_k, d_sparse]
    score_estimate = Q_sparse @ K_sparse.T  # [n_heads, S_q, S_k]
    block_scores = max_pool(score_estimate, block_size)
    selected_blocks = top_k(block_scores, B)

    # Attention 仍然使用全维度（等价）
    K_selected = load_from_cpu(K_g_cpu, selected_blocks)   # 全 d_head
    O[g] = flash_attn(Q_heads[g], K_selected, V_selected)  # 全维度计算
```

### 6.3 新增 Metric

在 RULER 评估框架中新增：

```bash
# 使用 dim-reordered 的 xattn chunked prefill
./scripts/run_ruler.sh glm-4-9b synthetic xattn_chunked_reordered \
  --reorder-config results/reorder_config/glm-4-9b.json
```

---

## 7. 与 Solution 4（Q Head Regrouping）的对比

| 维度 | Solution 4 (Head Regrouping) | Solution 5 (KV Dim Reorder) |
|------|------------------------------|-----------------------------|
| 数学等价性 | ❌ 改变 Q-K 配对关系，cos_sim ≈ 0 | ✅ 完全等价（dot product 不变） |
| 实现原理 | 重分配 Q head 到不同 KV group | 同一 group 内重排 d_head 维度 |
| 对 W_o 的修改 | 需要（重排 Q head 输出顺序） | **不需要** |
| 对 W_v 的修改 | 需要（随 KV group 移动） | **不需要** |
| Accuracy 风险 | 高（模型语义改变） | 无（路由近似，attention 等价） |
| 路由改善机制 | 改变 head 间的负载分配 | 去除路由中的 dense-dim 噪声 |
| Profiling 数据 | 需要（calibration 样本） | 需要（K 矩阵统计） |
| 收益来源 | Dense head 不污染 sparse group | 路由质量提升 → 加载更少 blocks |

### 解决的根本问题不同

- **Solution 4** 尝试从"group 级别"解决问题：将 dense Q head 和 sparse Q head 分离到不同 group，但代价是破坏 Q-K 训练配对。
- **Solution 5** 从"维度级别"解决问题：保持所有 Q-K 配对不变，仅改变 block selection 的信号质量。

---

## 8. 潜在风险与缓解

### 8.1 Dense/Sparse 维度分界的稳定性

**风险**：不同序列、不同任务下，哪些维度是 dense 可能不固定。

**分析**：Dense 维度的本质是 $K_g[:, i]$ 的统计特性（不同 token 的 key embedding 在该维度上的分布）。这是模型权重 $W_k$ 的固有性质，与输入无关——即 $W_k$ 本身的第 $i$ 列方向如果对所有输入产生的 key 都集中在某个数值附近，则该维度 "结构性地" dense。

**验证方法**：
1. 在多个不同 calibration 数据集（PG19, C4, RULER tasks）上分别 profile
2. 计算 dense 维度集合的 Jaccard similarity（跨数据集）
3. 如果 Jaccard > 0.8，则分界稳定，可以使用统一配置

### 8.2 改进路由的 Routing Recall

**风险**：去掉 dense 维度后，routing 使用 $d_S = d_\text{head} - d_\text{dense}$ 个维度，有效信息减少，recall 可能下降。

**缓解**：
- 从 $d_S / d_\text{head}$ 的角度，如果 $d_\text{dense}$ 很小（如 10/128 = 8%），信息损失极小
- 如果 dense 维度确实提供零判别力，去掉它们反而提升 SNR
- 通过离线实验测量 recall@k 的变化，设定下限（如 recall > 0.9）

### 8.3 不同 Layer 的差异

**风险**：浅层 layer 的 K 统计特性可能与深层不同，d_dense 需要 per-layer 配置。

**缓解**：Profiling 本身就是 per-layer 的，配置文件记录每一层的 perm 和 d_dense。

---

## 9. 量化预期收益（GLM-4-9B 案例）

以 64k 序列为例：

**当前情况**（来自测试数据）：
- KV Group 2：union coverage = 99.8%（几乎全部 block 必须加载）
- KV Group 3：union coverage = 100%

**如果情形 A 成立**（dense 维度主导了这些 group 的"伪密集"）：
- 去掉 dense 维度后，routing 可能降至 30-50% union coverage
- KV block 加载量减少约 50-70%

**如果部分为情形 A、部分为情形 B**（混合场景）：
- 情形 A 的 head 受益，情形 B 的 head 不变
- 综合改善：KV 加载量减少约 20-40%

**验证方法（Phase 1）**：
对单个 layer 的 K_g 进行 dim-level 的注意力可视化：
- 计算 $Q_h[:, i] \cdot K_g[:, i]^\top$ 对单个维度的 attention map
- 观察 dense 维度的 attention map 是否接近均匀分布
- 观察 sparse 维度的 attention map 是否具有明显的局部性或稀疏性

---

## 10. 实施路线图

### Phase 1：维度级别可视化验证（1-2 天）

目标：验证 K 维度确实存在 dense/sparse 的分化。

实现：
1. 加载 `results/kvcache-rope/glm-4-9b/64k/layer_05.pt`
2. 对每个 K_g 维度 $i$，计算 Block Discriminability Score
3. 可视化：
   - Score 排序曲线（是否有明显 elbow？）
   - Top-5 dense 维度和 Top-5 sparse 维度的 per-dim attention map
4. 验证：去掉 dense 维度后，routing recall 是否提升？

### Phase 2：权重重排实现与验证（2-3 天）

1. 实现 `compass/src/kv_dim_reorder.py`
2. 对 GLM-4-9B 的 Layer 5 应用 permutation
3. 验证：重排前后的完整 attention output 数值是否完全相同（fp32 误差 < 1e-5）
4. 测量：routing 使用 sparse 维度后，block selection recall 变化

### Phase 3：集成到 xattn_bsa 并基准测试（3-5 天）

1. 修改 nanovllm 的 block selection 逻辑
2. 在 RULER 上测试 `xattn_chunked_reordered`：
   - accuracy vs xattn_chunked baseline
   - KV block 加载量变化
   - 端到端 latency 变化

### Phase 4：全层 profiling 与多模型验证（5-7 天）

1. 对所有 40 层做 profiling，生成完整 reorder_config
2. 验证 d_dense 跨层的分布（是否有规律？）
3. 在 Llama-3.1-8B、Qwen2.5-7B 上验证（通用性）

---

## 11. 结论

### 11.1 可行性评分：8/10

- ✅ 数学上完全等价：permutation 不改变 attention 数值输出
- ✅ 无需修改 W_v、W_o：实现简洁
- ✅ 路由信号质量理论上严格提升（去除 dense 偏置噪声）
- ✅ 实现风险低：主体是一次性权重排列 + routing 逻辑微改
- ⚠️ Dense/sparse 分界的稳定性需 Phase 1 验证
- ⚠️ 实际 routing recall 提升幅度依赖于 K 维度的统计特性（需实测）
- ⚠️ 对"真正 dense 的 head"（情形 B）无改善

### 11.2 Next Steps

1. **立即执行 Phase 1**：在已有的 `results/kvcache-rope/glm-4-9b/64k/layer_05.pt` 上做 dim-level 可视化，验证 K 维度 dense/sparse 分化是否显著
2. **根据 Phase 1 结果**：
   - 如果分化显著（gap > 0.5）→ 进入 Phase 2
   - 如果不明显 → 重新评估，可能需要使用更强的维度分解方法（如 SVD）

---

## References

1. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
2. [Chunked Prefill and KV Cache Offloading (vLLM)](https://docs.vllm.ai/en/latest/configuration/optimization/)
3. [XAttention: Block-Sparse Attention for Long Context](../XATTN_CHUNKED_PREFILL.md)
4. GQA Dense Head Issue Analysis: `docs/GQA_DENSE_HEAD_ISSUE.md`
5. Solution 4 (Q Head Regrouping): `docs/04_q_head_regrouping.md`

---

**Document Status**: Draft v1.0
**Date**: 2026-02-18
**Next Review**: After Phase 1 visualization experiments
