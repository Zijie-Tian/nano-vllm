# Progress: CUDA Graph for Offload Mode

## Session: 2026-01-22

### 调研阶段 ✅ 完成

**完成的调研**:

1. ✅ 分析 `model_runner.py` 中的 CUDA Graph 实现
   - `capture_cudagraph()`: 为不同 batch size 捕获完整 model forward
   - `run_model()`: 通过 `is_chunked_prefill` 决定 eager/graph

2. ✅ 分析 offload decode 流程
   - `run_chunked_offload_decode()` 设置 `is_chunked_prefill=True`
   - 导致永远使用 eager mode

3. ✅ 分析 ring buffer pipeline
   - `_decode_ring_buffer_pipeline()` 包含 H2D 传输 + attention 计算
   - H2D 不能 graph，attention 可以 graph

4. ✅ 验证 graph 复用策略
   - 创建 `test_chunk_attention_graph_reuse.py`
   - 确认 2 个 graph 可复用于所有层

### 计划编写 ✅ 完成

- ✅ 创建 `task_plan.md`
- ✅ 创建 `findings.md`
- ✅ 创建 `progress.md`

### 下一步: 实现

**Phase 1**: 添加 graph 捕获到 OffloadEngine
- [ ] 在 `offload_engine.py` 添加 `capture_attention_graphs()`
- [ ] 添加 `attention_graph_causal` 和 `attention_graph_non_causal` 属性

**Phase 2**: 修改 ring buffer pipeline
- [ ] 在 `_decode_ring_buffer_pipeline()` 使用 graph replay
- [ ] 保持 H2D 和 merge 为 eager

**Phase 3**: 测试
- [ ] 运行 needle test 验证正确性
- [ ] 对比性能

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `tests/test_chunk_attention_graph.py` | ✅ 已提交 | 预分配 chunk pair graphs 测试 |
| `tests/test_chunk_attention_graph_reuse.py` | 待提交 | Graph 复用验证 |
| `task_plan.md` | ✅ 创建 | 实现计划 |
| `findings.md` | ✅ 创建 | 调研发现 |
| `progress.md` | ✅ 创建 | 进度日志 |
