# BLASST Split Kernel Design

## Architecture Overview

将当前的 fused kernel 拆分为两个独立的 kernel，实现更精确的 skip 统计和更好的模块化。

## Kernel 1: BLASST Mask Calculation

### Purpose
计算每个 (query, kv_block) 对是否需要跳过 PV 计算。

### Interface
```python
def blasst_mask_forward(
    q: torch.Tensor,          # [num_heads, q_len, head_dim]
    k: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    running_max: torch.Tensor, # [num_heads, q_len]
    ln_lambda: float,
    softmax_scale: float,
    granularity: int = 128,
) -> torch.Tensor:            # [num_heads, q_len, num_kv_blocks] (bool)
```

### Algorithm
```
For each query token:
    For each KV block at granularity:
        1. Compute QK scores for this block
        2. local_max = max(scores)
        3. skip = (local_max - running_max) < ln_lambda
        4. Store skip flag
        5. Update running_max = max(running_max, local_max)
```

### Memory Analysis
- Input Q: num_heads * q_len * head_dim * 2 bytes (fp16)
- Input K: num_kv_heads * kv_len * head_dim * 2 bytes (fp16)
- Output mask: num_heads * q_len * (kv_len / granularity) * 1 byte (bool)

Example: q_len=512, kv_len=32768, granularity=128
- mask size = num_heads * 512 * 256 = 131K * num_heads (very small)

## Kernel 2: BLASST Attention with Mask

### Purpose
根据 skip mask 计算 attention，只计算非跳过的 blocks。

### Interface
```python
def blasst_attn_forward(
    q: torch.Tensor,          # [num_heads, q_len, head_dim]
    k: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    v: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    skip_mask: torch.Tensor,  # [num_heads, q_len, num_kv_blocks] (bool)
    running_max: torch.Tensor, # [num_heads, q_len]
    softmax_scale: float,
    granularity: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # output: [num_heads, q_len, head_dim]
    # new_running_max: [num_heads, q_len]
    # lse: [num_heads, q_len]
```

### Algorithm
```
For each query token:
    Initialize: acc = 0, sum_exp = 0, running_max

    For each KV block at granularity:
        1. If skip_mask[query, block] == True:
               - Update running_max (but don't compute PV)
               - Continue

        2. Compute QK scores
        3. Online softmax update
           new_max = max(running_max, local_max)
           scale = exp(running_max - new_max)
           acc = acc * scale
           sum_exp = sum_exp * scale

        4. Compute PV and accumulate
           exp_scores = exp(scores - new_max)
           acc += exp_scores @ V
           sum_exp += sum(exp_scores)
           running_max = new_max

    5. Final: output = acc / sum_exp
              lse = running_max + log(sum_exp)
```

### Key Difference from Current Kernel
- Current: skip decision is internal, any query needing PV causes whole tile to compute
- New: skip_mask is external input, each query independently decides per block

## Integration Flow

```python
# In blasst.py compute_chunked_prefill()

# 1. Collect all KV blocks
k_all = torch.cat(k_blocks, dim=1)
v_all = torch.cat(v_blocks, dim=1)

# 2. Compute skip mask for all query sub-chunks
for q_sub_idx in range(num_q_subchunks):
    skip_mask = blasst_mask_forward(
        q=q_sub_kernel,
        k=k_all,
        running_max=q_sub_running_max[q_sub_idx],
        ln_lambda=ln_lambda,
        softmax_scale=softmax_scale,
        granularity=granularity,
    )

    # 3. Precise skip statistics
    skipped = skip_mask.sum().item()
    total = skip_mask.numel()

    # 4. Compute attention using mask
    fused_o, new_running_max, lse = blasst_attn_forward(
        q=q_sub_kernel,
        k=k_all,
        v=v_all,
        skip_mask=skip_mask,
        running_max=q_sub_running_max[q_sub_idx],
        softmax_scale=softmax_scale,
        granularity=granularity,
    )
```

## Advantages

1. **Precise Skip Statistics**: Direct counting from mask
2. **Modularity**: Mask can be reused for visualization/analysis
3. **Flexibility**: Easy to modify skip logic without changing attention kernel
4. **Debugging**: Can inspect mask to understand skip behavior

## Performance Considerations

1. **Memory**: Mask is small (~100KB for typical configs)
2. **Kernel Launch**: Two launches vs one, but both are compute-bound
3. **Occupancy**: Smaller kernels may have better occupancy

## Next Steps

1. Use Codex MCP to generate mask kernel
2. Use Codex MCP to generate attention kernel with mask
3. Update blasst.py integration
4. Test correctness and performance
