# CV 融合算子开发

CV 融合算子指同一个算子中同时使用 Cube Core 和 Vector Core：Cube Core 通常负责 `tl.dot`、矩阵乘或卷积式主计算，Vector Core 负责 bias、activation、softmax、归约、mask、layout 重排或跨块同步。CV 融合的目标是减少 kernel 边界和 GM 往返，但需要同时控制 Cube tile、Vector tile、UB/L1 占用和同步关系。

## CV 融合简单算子开发

简单 CV 融合建议先从本仓 [矩阵乘法样例](../examples/05_matrix_multiplication_example.md) 抽出稳定的 `tl.dot` 主计算，再在写回前加入 Vector 后处理；更复杂的切片更新可参考 [融合注意力样例](../examples/04_fused_attention_example.md)。最小路径如下：

1. 先实现稳定的 Cube 主计算，例如 `acc = tl.dot(a, b, acc)`。
2. 在 accumulator 写回前融合轻量 Vector 后处理，例如 bias、scale、activation 或 dtype cast。
3. 对较大的 accumulator，可使用 `range` 配合 `extension.extract_slice`/`extension.insert_slice` 做普通子块切分，避免 Vector 后处理阶段 UB overflow。
4. `extension.parallel(..., bind_sub_block=True)` 属于更强的显式多 Vector 子块绑定路径，目标硬件和编译配置存在差异时可能不可用，不建议作为简单示例的默认写法。

示例结构：

```python
# 在 matmul kernel 内部，K 循环完成后得到 fp32 accumulator。
acc = tl.dot(a, b, acc)  # 通常位于 K 维循环内，这里仅展示结构。

# 在写回前融合轻量 Vector 后处理。
acc = tl.where(acc >= 0, acc, 0.01 * acc)
c = acc.to(tl.float16)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

简单 CV 融合开发时要保持边界清晰：Cube 负责产生较大的二维 accumulator，Vector 负责同一 tile 内的逐元素或小规模归约。若 Vector 部分需要跨多个 Cube tile 共享状态，就需要引入同步、workspace 或拆分 kernel。

## CV 融合复杂算子开发

复杂 CV 融合可参考 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 中的 best practice：

- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py)：Decode attention 中 QK/PV 使用 Cube，softmax、mask、指数、归一化和离散 KV 访存重排使用 Vector。
- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md)：展示融合 cat、slice、conv1d update 时如何用 `extension.insert_slice`、转置和分核优化减少离散访存与 padding 开销。

复杂 CV 融合建议按数据流分层组织：

1. **主计算层**：识别哪些步骤必须走 Cube，例如 QK、PV、GEMM、batched matmul。
2. **Vector 后处理层**：识别 softmax、activation、mask、scale、normalization、cat/slice、layout transform 等是否能在同一 tile 内完成。
3. **访存重排层**：对离散 KV cache、MoE token 重排、短尾轴 tensor，优先在 UB 中用 `extension.insert_slice`、`extension.extract_slice`、转置或借轴转置形成硬件友好的连续访问。
4. **流水和同步层**：通过 `multibuffer`、`set_workspace_multibuffer`、`tile_mix_vector_loop`、`tile_mix_cube_loop` 等编译选项探索 Cube 与 Vector 的重叠执行。
5. **分核层**：CV 融合算子通常按 Cube Core 数量发射 grid；运行时会以约 1:2 的比例协同 Vector Core。不要简单沿用 GPU 上的大 grid。

对于 attention 类 CV 融合，推荐先让非 causal、短序列、小 head_dim 的 case 跑通，再逐步加入：

- causal mask 分阶段处理。
- 长序列 K/V block 循环。
- `m_i`/`l_i` 的数值稳定 softmax 更新。
- HEAD_DIM 较大时的 accumulator workspace 和子块切分。
- KV cache 离散索引下的 load 重排。

复杂 CV 融合调优时，优先观察 profiling 中 Cube、Vector、MTE2 的时间占比。如果 Cube 等待 Vector，考虑减少 Vector 后处理粒度或打开 CV balance 相关选项；如果 Vector 等待搬运，优先检查离散访存、tail-axis padding 和 multibuffer 配置。
