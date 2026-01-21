# Task Plan: XAttention BSA 集成到 nanovllm

## Goal

使用 `--sparse-policy XATTN_BSA` 运行 `test_ruler.py`，通过 `niah_single_1` 的前 5 个 sample。

**验收标准**:
```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --task niah_single_1 \
    --sample-ids 0,1,2,3,4
# 期望: 5/5 PASS
```

## 当前状态

- `XAttentionBSAPolicy.compute_chunked_prefill` 实现 = `FullAttentionPolicy`（无 sparse）
- `xattn_estimate_chunked` 已实现并验证
- BSA kernel (`block_sparse_attn`) 可用

## Phases

- [ ] Phase 1: 理解当前代码路径
- [ ] Phase 2: 实现 sparse mask 估计
- [ ] Phase 3: 实现 BSA sparse 计算
- [ ] Phase 4: 测试验证

## Phase 1: 理解当前代码路径

### 1.1 确认 XATTN_BSA policy 是否被正确加载
- [ ] 检查 `test_ruler.py` 如何解析 `--sparse-policy XATTN_BSA`
- [ ] 检查 `KVCacheManager` 如何实例化 sparse_policy
- [ ] 运行 baseline 测试（`--sparse-policy FULL`）确认基础功能正常

### 1.2 确认数据流
- [ ] `compute_chunked_prefill` 的输入参数含义
- [ ] `offload_engine` 提供的数据访问接口
- [ ] 当前 chunk 的 K/V 如何获取

## Phase 2: 实现 sparse mask 估计

### 2.1 调用 xattn_estimate_chunked
- [ ] 在 `compute_chunked_prefill` 中加载历史 K
- [ ] 拼接历史 K + 当前 K
- [ ] 调用 `xattn_estimate_chunked(q, k_full, q_start_pos=...)`
- [ ] 获取 block mask

### 2.2 处理参数对齐
- [ ] BSA block_size = 128
- [ ] chunk_size 与 kvcache_block_size 的关系
- [ ] q_start_pos 计算

## Phase 3: 实现 BSA sparse 计算

### 3.1 方案选择
- 选项 A: 历史 + 当前分开计算，然后 merge
- 选项 B: 全部一起用 BSA 计算

### 3.2 实现
- [ ] 构造 BSA 需要的输入格式
- [ ] 调用 `block_sparse_attn_func`
- [ ] 处理输出格式

## Phase 4: 测试验证

### 4.1 单元测试
- [ ] 验证 sparse mask 与 `test_xattn_estimate_chunked.py` 一致

### 4.2 集成测试
- [ ] 运行验收命令
- [ ] 5/5 PASS

## Key Questions

1. 历史 K 如何高效加载？（全量 vs 按需）
2. BSA causal mask 如何处理？（历史 non-causal + 当前 causal）

## Status

**Currently in Phase 1** - 等待用户确认后开始

## 待讨论

请确认：
1. 这个 goal 和验收标准是否正确？
2. 我使用哪个 GPU 运行测试？
