# TriAttention Calibration Guide

本文档详细说明 `scripts/calibrate_triattention.py` 的作用、计算对象、公式来源、输入输出、为什么采用 offline 统计，以及在当前 COMPASS 工程中的使用方式。

---

## 1. 一句话总结

`scripts/calibrate_triattention.py` 不是推理脚本，也不是评测脚本。  
它是一个 **离线校准脚本**，用于为 TriAttention 生成：

- 每层（layer）
- 每个 head
- 在 **pre-RoPE 频域空间**
- 的 query 统计先验

最终产物是一个 `.pt` 文件，后续由：

- `compass/src/TriAttention.py`

在推理阶段加载，用于对 KV cache 中的 key 做打分和压缩。

---

## 2. 它到底在做什么？

从实现上看，这个脚本会：

1. 加载模型与 tokenizer
2. 读取一段纯文本
3. tokenize 成 `input_ids`
4. 在每层 attention 上挂 pre-hook，截获该层输入
5. 手动执行 `q_proj(hidden_states)` 得到 query
6. 按模型真实 rotary 配置重新施加 RoPE
7. 再用 `invert_rope(...)` 反解回 pre-RoPE 空间
8. 将 head 向量转成复数频率对
9. 对每层每头计算：
   - `q_mean_complex`
   - `q_abs_mean`
10. 把这些统计量和 metadata 保存成 `.pt`

也就是说：

> 这个脚本不是在计算某次请求的 attention，而是在为每个 head 生成一个“长期频域画像”。

---

## 3. 输入 / 输出

### 输入参数

脚本暴露的命令行参数见 `scripts/calibrate_triattention.py`：

```bash
python scripts/calibrate_triattention.py \
  --model <model_path_or_hf_id> \
  --input <plain_text_file> \
  --output <stats.pt> \
  --max-length 32768 \
  --device cuda \
  --attn-implementation flash_attention_2
```

其中：

- `--model`
  - HuggingFace model id 或本地模型目录
- `--input`
  - 一份纯文本文件，用于做一次 calibration forward
- `--output`
  - 输出的 stats 文件路径
- `--max-length`
  - calibration 文本的最大 token 长度
- `--device`
  - 一般是 `cuda`
- `--attn-implementation`
  - 默认 `flash_attention_2`

### 输出文件

输出是一个 `.pt` 文件，例如：

```bash
/tmp/triattention_llama31_smoke.pt
```

它包含：

- `metadata`
- `stats`

其中 `stats` 按 `(layer, head)` 存储对应频域统计量。

---

## 4. 代码入口与流程

### 4.1 `_find_attention_layers(model)`

文件：`scripts/calibrate_triattention.py`

作用：
- 定位模型所有 transformer block 中的 `self_attn`

它默认假设模型主干是：

```python
model.model.layers[i].self_attn
```

这对当前 Llama/Qwen 路径都是合理的。

---

### 4.2 `calibrate_triattention_stats(...)`

这是主函数，流程如下。

#### Step 1：加载 config / tokenizer / model

它先做：

\[
\\text{config} = \\text{AutoConfig.from\\_pretrained}(model)
\]

\[
\\text{tokenizer} = \\text{AutoTokenizer.from\\_pretrained}(model)
\]

\[
\\text{model} = \\text{AutoModelForCausalLM.from\\_pretrained}(model)
\]

这里会记录：

- `num_hidden_layers`
- `num_attention_heads`
- `head_dim`
- `rope_style`

用于后续统计。

---

#### Step 2：读取文本并 tokenize

脚本读取纯文本：

```python
text = Path(input_path).read_text(...)
input_ids = tokenizer.encode(text, return_tensors=\"pt\", truncation=True, max_length=max_length)
```

记 token 数为：

\[
T = \\text{seq\\_len}
\]

这个 \(T\) 决定了本次校准会统计多少个 token 的 query。

---

#### Step 3：构造 RoPE 表

脚本会构造：

- `cos_table`
- `sin_table`

也就是针对每个位置 \(p\) 的 rotary embedding 参数。

其目的不是直接做注意力，而是为后面：

- 施加 RoPE
- 反解 RoPE

提供精确的一致性支持。

---

#### Step 4：在每层挂 pre-hook 捕获 query

这是最关键的一步。

脚本对每层 attention 注册 pre-hook：

```python
attn.register_forward_pre_hook(...)
```

在 hook 内部，它手动做：

\[
Q = W_q H
\]

对应代码：

```python
q = attn.q_proj(hidden_states)
```

再 reshape 成：

\[
Q \\in \\mathbb{R}^{B \\times H \\times T \\times D}
\]

其中：

- \(B\)：batch size
- \(H\)：num_heads
- \(T\)：sequence length
- \(D\)：head_dim

---

#### Step 5：重新施加 RoPE

脚本不会直接统计裸 `q_proj` 输出，而是先按模型真实 rotary 规则计算：

\[
\\tilde q = \\text{RoPE}(q)
\]

代码中对应：

```python
q_rot = (q * cos) + (rotate_half(q) * sin)
q_rot = q_rot * attn_scale
```

这里 `attn_scale` 来自模型 rotary 模块本身。

也就是说：

> 它先把 query 放回“模型真实 attention 会看到的旋转空间”。

---

#### Step 6：再用 `invert_rope` 反解回 pre-RoPE 空间

脚本真正想统计的不是 post-RoPE query，而是：

\[
q = \\text{RoPE}^{-1}(\\tilde q)
\]

代码：

```python
q_base = invert_rope(q_rot, cos, sin, attn_scale, style=rope_style)
```

这一点非常关键：

> TriAttention 论文认为真正稳定的结构在 **pre-RoPE 空间**，  
> 所以统计必须建立在反解之后的 query 上。

---

#### Step 7：转成复数频率表示

对某个 head 的向量：

\[
q \\in \\mathbb{R}^{D}
\]

脚本会转成：

\[
q_f \\in \\mathbb{C}, \\quad f=1,2,\\dots,D/2
\]

代码：

```python
q_complex = to_complex_pairs(q_head, style=rope_style)
```

这一步的意义是：

> 把 head 的表示改写成频率分量，更适合 TriAttention 的 trig / frequency 打分公式。

---

## 5. 它真正统计的量是什么？

对每一层每一头、每个频率分量 \(f\)，脚本保存两个统计量。

### 5.1 复数均值 `q_mean_complex`

\[
\mu_f = \\mathbb{E}[q_f]
\]

实现上是：

\[
q\\_mean\\_complex[f] = \\frac{1}{T} \\sum_{t=1}^{T} q_f^{(t)}
\]

代码：

```python
q_mean_complex = q_complex.mean(dim=0)
```

它表示：

- 平均方向（phase）
- 平均复数中心（center）

这是 TriAttention 最核心的先验。

---

### 5.2 幅值均值 `q_abs_mean`

\[
a_f = \\mathbb{E}[|q_f|]
\]

实现上是：

\[
q\\_abs\\_mean[f] = \\frac{1}{T} \\sum_{t=1}^{T} |q_f^{(t)}|
\]

代码：

```python
q_abs_mean = q_complex.abs().mean(dim=0)
```

它表示：

- 这个频率分量平均有多强
- 不关心方向，只关心强度

---

## 6. 这些量后面怎么用于打分？

在推理阶段，TriAttention 会拿当前 key 的 pre-RoPE 频域表示 \(k_f\)，再结合 calibration 得到的：

- \(\mu_f = \\mathbb{E}[q_f]\)
- \(a_f = \\mathbb{E}[|q_f|]\)

去构造打分公式。

### 6.1 幅值项

\[
\\text{amp}_f = |\\mu_f|\\cdot|k_f|
\]

### 6.2 相位差

\[
\\phi_f = \\arg(\\mu_f) - \\arg(k_f)
\]

### 6.3 残余补偿项

\[
\\text{extra}_f = (a_f - |\\mu_f|)\\cdot|k_f|
\]

### 6.4 最终核心打分

\[
S(k, \\Delta)=\\sum_f \\text{amp}_f \\cos(\\omega_f \\Delta + \\phi_f) + \\sum_f \\text{extra}_f
\]

其中：

- \(\Delta\) 是 query/key 的相对距离
- \(\omega_f\) 是第 \(f\) 个 rotary frequency

也就是说：

> calibration 脚本负责准备 \(\mu_f\) 和 \(a_f\)，  
> 推理阶段负责把它们和当前 key \(k_f\) 组合起来算分数。

---

## 7. 为什么需要同时保存 `q_mean_complex` 和 `q_abs_mean`？

因为：

- `q_mean_complex = E[q_f]` 捕获了**中心方向**
- `q_abs_mean = E[|q_f|]` 捕获了**平均能量**

如果只保存 `E[q_f]`，你会丢掉：
- 围绕中心的离散程度

而 `E[|q_f|] - |E[q_f]|` 正好可以衡量“有多少能量没有集中在中心上”。

所以：

\[
E[|q_f|] - |E[q_f]|
\]

在推理时就成了一个很自然的 residual term。

---

## 8. metadata 存了什么，为什么要存？

脚本最后还会写入：

- `head_dim`
- `dtype`
- `attn_implementation`
- `rope_style`
- `rope_type`
- `num_traces`
- `sampled_heads`

其目的不是参与打分，而是为了：

### 8.1 兼容性校验
后续推理时会检查：
- 这个 stats 文件是不是给同类模型用的
- head_dim 是否一致
- rope_style / rope_type 是否一致

### 8.2 调试可追踪
你以后拿到一个 `.pt` 文件时，可以知道：
- 它是给什么模型做的
- 用什么 attention implementation 做的
- 当时统计了哪些 head

---

## 9. 为什么它是 offline，而不是 prefill 时现算？

这是方法设计最重要的一点之一。

### 9.1 它要的是“稳定先验”，不是当前 prompt 的瞬时统计
如果在当前 prompt 上现算，你得到的是：

\[
\hat{\mu}_f^{(prompt)} = \\frac{1}{T}\\sum_{t \\in prompt} q_f^{(t)}
\]

这只是当前 prompt 的估计。

但 TriAttention 想近似的是：

\[
\\mathbb{E}[q_f^{(future decode)}]
\]

也就是未来 decode query 的 head 级偏好。

当前 prompt 的统计：
- 噪声更大
- 更 prompt-specific
- 对未来 answer token 未必有代表性

### 9.2 online 现算会增加 TTFT
如果每个请求都在 prefill 里：
- 抓所有层 query
- 反解 RoPE
- 转 complex
- 算均值

那每次请求都要额外做一遍 calibration，TTFT 会明显变慢。

### 9.3 online 现算内存更重
因为你要在长 prompt 上临时保留大量中间 query 激活。

offline 的优势是：
- 一次统计
- 所有请求复用
- 运行时只加载一个小 `.pt` 文件

所以这个脚本天然是 **offline calibration**，而不是在线 prefill 统计。

---

## 10. 对不同模型意味着什么？

这也是你前面问到的重点。

因为 stats 文件依赖：

- `head_dim`
- `rope_style`
- `rope_type`
- 模型自身 query 分布

所以：

> **不同模型原则上都应该重新跑一次 calibration。**

例如：
- `Qwen3-8B` 要有自己的 stats
- `Llama-3.1-8B-Instruct` 也要有自己的 stats

不能期望一个模型的 stats 直接拿给另一个模型用还能完全可靠。

---

## 11. 它和当前 COMPASS TriAttention 的关系

你现在工程里：

- `scripts/calibrate_triattention.py`
  - 负责生成 calibration stats
- `compass/src/TriAttention.py`
  - 负责在推理时加载和使用这些 stats
- `compass/src/triattention_utils.py`
  - 负责：
    - `invert_rope`
    - `to_complex_pairs`
    - `save_head_frequency_stats`
    - `load_head_frequency_stats`

它们是同一条链路的不同阶段：

### 离线阶段
```text
plain text -> model forward -> capture Q -> invert RoPE -> complex stats -> stats.pt
```

### 在线阶段
```text
stats.pt + current K cache -> frequency scoring -> keep_indices -> KV compression
```

---

## 12. 最后用最短的话再总结一次

`scripts/calibrate_triattention.py` 的核心作用就是：

> **为每层每头估计 pre-RoPE query 的频域统计先验**

它实际算的核心公式是：

\[
q\_mean\_complex[f] = \\frac{1}{T}\sum_t q_f^{(t)}
\\]

\[
q\_abs\\_mean[f] = \\frac{1}{T}\sum_t |q_f^{(t)}|
\\]

然后推理阶段拿这些量去构造：

\[
S(k, \\Delta)=\\sum_f |E[q_f]||k_f|\\cos(\\omega_f\\Delta + \\phi_f)+\\sum_f (E[|q_f|]-|E[q_f]|)|k_f|
\\]

这就是它存在的数学意义。

---

如果你愿意，我下一步还可以继续写一份更深的说明，专门讲：

1. `invert_rope()` 的数学推导  
2. `to_complex_pairs()` 为什么可以理解成频域拆分  
3. `q_mean_complex / q_abs_mean` 如何一步步变成最终 `keep_indices` 选择结果。*** Update File: /home/zijie/Code/COMPASS/AGENTS.md
