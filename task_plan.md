# Task Plan: XAttention BSA 真正的 Sparse 实现

## Goal

实现 XAttentionBSAPolicy 的真正 sparse attention，在 `select_blocks` 中使用 `xattn_estimate_chunked` 选择重要的 blocks，然后复用 FullAttentionPolicy 的 ring buffer pipeline。

**验收标准**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --datasets niah_single_1 \
    --sample-indices 0,1,2,3,4
# 期望: 5/5 PASS，并且真正使用 sparse selection
```

## 当前状态: Phase 1 - 代码分析完成

## 核心设计理解

### 1. Block Size 关系

| 参数 | 值 | 说明 |
|------|-----|------|
| BSA block_size | 128 tokens | XAttention 的 block 粒度 |
| kvcache_block_size | 1024 tokens | CPU offload 的 block 粒度 |
| 比例 | 1:8 | 1 CPU block = 8 BSA blocks |

### 2. 特化条件（用户要求）

- BSA chunk_size = 外部 chunk_size
- 这样 `xattn_estimate_chunked` 返回的 mask 可以直接映射到 CPU block selection
- 复用现有的 `flash_attn_with_lse` + `merge_attention_outputs`

### 3. select_blocks 设计

```
select_blocks(available_blocks, offload_engine, ctx) -> List[int]
    │
    ├─ 1. 从 metadata cache 获取下采样的 K
    │      (在 on_prefill_offload 中收集)
    │
    ├─ 2. 调用 xattn_estimate_chunked(Q, K_downsampled, q_start_pos)
    │      返回 mask: [B, H, q_blocks, k_blocks]
    │
    ├─ 3. 将 BSA k_blocks 映射到 CPU block IDs
    │      每 8 个 BSA blocks = 1 CPU block
    │      只要 8 个中有任意一个被选中，就保留该 CPU block
    │
    └─ 4. 返回 selected_cpu_blocks
```

### 4. Metadata 存储策略

**方案 A**: 存储下采样的 K（内存友好）
```python
# on_prefill_offload 中:
k_downsampled = k_cache[::stride]  # [block_size/stride, H, D]
self._k_cache[layer_id][cpu_block_id] = k_downsampled
```

**内存计算** (stride=8):
- 每 block: (1024/8) * 8 * 128 * 2 bytes = 256 KB
- 256 blocks * 32 layers = 2 GB (GPU 上用于快速估计)

**方案 B**: 存储 min/max metadata (更省内存)
```python
# on_prefill_offload 中:
k_min = k_cache[:num_valid].min(dim=0).values  # [H, D]
k_max = k_cache[:num_valid].max(dim=0).values  # [H, D]
```
- 但这需要不同的估计算法，不能直接用 xattn_estimate

**决定**: 使用方案 A（下采样 K），因为可以直接复用 xattn_estimate_chunked

## Phases

- [x] Phase 1: 代码分析，理解当前实现
- [ ] Phase 2: 实现 on_prefill_offload 收集 K metadata
- [ ] Phase 3: 实现 select_blocks 中的 xattn estimation
- [ ] Phase 4: 实现 BSA block → CPU block 的映射
- [ ] Phase 5: 测试验证

## Phase 2: on_prefill_offload 实现

### 需要修改的文件
- `nanovllm/kvcache/sparse/xattn_bsa.py`

### 实现细节

```python
class XAttentionBSAPolicy(SparsePolicy):
    def __init__(self, threshold=0.9, stride=8, ...):
        self.threshold = threshold
        self.stride = stride
        self._k_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        # _k_cache[layer_id][cpu_block_id] = k_downsampled

    def initialize(self, num_layers, num_kv_heads, head_dim, num_cpu_blocks, dtype, device):
        """初始化 K cache 结构"""
        self._k_cache = {layer_id: {} for layer_id in range(num_layers)}
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

    def on_prefill_offload(self, cpu_block_id, layer_id, k_cache, num_valid_tokens):
        """收集下采样的 K 用于后续估计"""
        # k_cache: [block_size, num_kv_heads, head_dim]
        k_downsampled = k_cache[:num_valid_tokens:self.stride].clone()
        # k_downsampled: [num_valid_tokens//stride, num_kv_heads, head_dim]
        self._k_cache[layer_id][cpu_block_id] = k_downsampled
```

## Phase 3: select_blocks 实现

### 关键问题

1. **Q 从哪里来？**
   - `ctx.query` 需要在调用 select_blocks 时传入
   - 当前 FullAttentionPolicy 传递 `query=None`
   - 需要修改 compute_chunked_prefill 传递真实的 Q

2. **Q 的格式转换**
   - 输入 Q: [seq_len, num_heads, head_dim]
   - xattn 需要: [B, H, q_len, D]
   - 转换: `q.unsqueeze(0).transpose(1, 2)`

3. **K 的组装**
   - 从 `_k_cache[layer_id]` 获取各 block 的下采样 K
   - 按 `available_blocks` 顺序 cat 起来
   - 结果: [B, H, total_k_downsampled, D]

### 实现草案

```python
def select_blocks(self, available_blocks, offload_engine, ctx):
    if not available_blocks or ctx.query is None:
        return available_blocks

    layer_id = ctx.layer_id

    # 1. 组装下采样的 K
    k_list = []
    for cpu_block_id in available_blocks:
        if cpu_block_id in self._k_cache[layer_id]:
            k_list.append(self._k_cache[layer_id][cpu_block_id])

    if not k_list:
        return available_blocks

    k_hist = torch.cat(k_list, dim=0)  # [total_tokens/stride, H, D]
    k_hist = k_hist.unsqueeze(0).transpose(1, 2)  # [1, H, k_len, D]

    # 2. 准备 Q
    q = ctx.query  # [seq_len, num_heads, head_dim]
    q = q.unsqueeze(0).transpose(1, 2)  # [1, H, q_len, D]

    # GQA 扩展（如果需要）
    if q.shape[1] != k_hist.shape[1]:
        num_groups = q.shape[1] // k_hist.shape[1]
        k_hist = k_hist.repeat_interleave(num_groups, dim=1)

    # 3. 计算 q_start_pos
    q_start_pos = len(available_blocks) * ctx.block_size

    # 4. 调用 xattn_estimate_chunked
    # 注意：K 已经是下采样的，需要调整参数
    attn_sum, mask = xattn_estimate_chunked(
        q, k_hist,
        q_start_pos=q_start_pos // self.stride,  # 调整到下采样空间
        block_size=self.BSA_BLOCK_SIZE // self.stride,  # 16
        stride=1,  # K 已经下采样
        threshold=self.threshold,
        chunk_size=q.shape[2],  # 与 Q 长度一致
        use_triton=self.use_triton,
    )

    # 5. 从 mask 提取 CPU block IDs
    # mask: [1, H, q_blocks, k_blocks]
    # 对所有 heads 取 OR
    selected_mask = mask.any(dim=1).squeeze(0)  # [q_blocks, k_blocks]
    # 对所有 q_blocks 取 OR（只要任意 Q 位置需要这个 K block）
    selected_k_mask = selected_mask.any(dim=0)  # [k_blocks]

    # 6. 映射 BSA blocks → CPU blocks
    # 每个 CPU block = 8 BSA blocks (block_size=1024, BSA_block=128)
    bsa_to_cpu_ratio = ctx.block_size // self.BSA_BLOCK_SIZE  # 8
    num_cpu_blocks = len(available_blocks)

    selected_cpu_indices = set()
    for bsa_idx in selected_k_mask.nonzero(as_tuple=True)[0].tolist():
        cpu_idx = bsa_idx // bsa_to_cpu_ratio
        if cpu_idx < num_cpu_blocks:
            selected_cpu_indices.add(cpu_idx)

    selected_blocks = [available_blocks[i] for i in sorted(selected_cpu_indices)]

    logger.info(f"[XAttn] select_blocks: {len(available_blocks)} -> {len(selected_blocks)} "
                f"({100*len(selected_blocks)/len(available_blocks):.1f}%)")

    return selected_blocks
```

## Phase 4: compute_chunked_prefill

### 关键修改

1. **传递真实的 Q 给 select_blocks**
   - 修改 PolicyContext 构造，设置 `query=q`

2. **复用 FullAttentionPolicy 的 pipeline**
   - 继承 FullAttentionPolicy 而不是 SparsePolicy
   - 或者直接调用父类方法

### 方案对比

**方案 A**: XAttentionBSAPolicy 继承 FullAttentionPolicy
```python
class XAttentionBSAPolicy(FullAttentionPolicy):
    # 只需要 override select_blocks 和 on_prefill_offload
    # compute_chunked_prefill 直接用父类的
```

**方案 B**: 独立实现，调用相同的 pipeline 代码
```python
class XAttentionBSAPolicy(SparsePolicy):
    def compute_chunked_prefill(self, q, k, v, ...):
        # 复制 FullAttentionPolicy 的代码
        # 但修改 PolicyContext 传递 query=q
```

**决定**: 使用方案 B，因为需要在 compute_chunked_prefill 中修改 PolicyContext

## Phase 5: 测试

### 单元测试
```bash
# 测试 select_blocks 的 sparsity
python -c "
from nanovllm.kvcache.sparse.xattn_bsa import XAttentionBSAPolicy
policy = XAttentionBSAPolicy(threshold=0.9)
# ... 测试代码
"
```

### 集成测试
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --datasets niah_single_1 \
    --sample-indices 0,1,2,3,4
```

## Key Decisions

| 决策 | 理由 |
|------|------|
| 使用下采样 K 作为 metadata | 可以直接复用 xattn_estimate_chunked |
| stride=8 | 平衡内存和精度 |
| BSA blocks → CPU blocks 映射用 OR | 只要有一个 BSA block 被选中就保留 |
| 继承 FullAttentionPolicy 的 pipeline | 复用已验证的 ring buffer 流程 |

## Files to Modify

| 文件 | 修改 |
|------|------|
| `nanovllm/kvcache/sparse/xattn_bsa.py` | 主要实现：initialize, on_prefill_offload, select_blocks |

## 注意事项

1. **GQA 处理**: Llama-3.1-8B 有 32 query heads, 8 kv heads，需要在估计时扩展 K
2. **内存管理**: `_k_cache` 存储在 GPU，需要在 reset() 时清理
3. **Triton 兼容性**: xattn_estimate_chunked 有 Triton bug，可能需要用 PyTorch fallback
4. **边界条件**: 第一个 chunk (available_blocks=[]) 时直接返回空列表

## Errors Encountered

(待填充)

## Status

**Currently in Phase 1** - 代码分析完成，准备开始 Phase 2 实现
