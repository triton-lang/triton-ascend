# CV融合算子开发

CV融合算子指同一个算子中同时使用Cube Core和Vector Core：Cube Core通常负责 `tl.dot`、矩阵乘或卷积式主计算，Vector Core负责bias、activation、softmax、归约、mask、layout重排或跨块同步。CV融合的目标是减少kernel边界和GM往返，但需要同时控制Cube tile、Vector tile、UB/L1占用和同步关系。

## CV融合简单算子开发

简单CV融合建议先从本仓 [矩阵乘法样例](../examples/05_matrix_multiplication_example.md) 抽出稳定的 `tl.dot` 主计算，再在写回前加入Vector后处理；更复杂的切片更新可参考 [融合注意力样例](../examples/04_fused_attention_example.md)。最小路径如下：

1. 先实现稳定的Cube主计算，例如 `acc = tl.dot(a, b, acc)`。
2. 在accumulator写回前融合轻量Vector后处理，例如bias、scale、activation或dtype cast。
3. 对较大的accumulator，可使用 `range` 配合 `extension.extract_slice`/`extension.insert_slice` 做普通子块切分，避免Vector后处理阶段UB overflow。
4. `extension.parallel(..., bind_sub_block=True)` 属于更强的显式多Vector子块绑定路径，目标硬件和编译配置存在差异时可能不可用，不建议作为简单示例的默认写法。

示例结构：

```python
# 在matmul kernel内部，K循环完成后得到fp32 accumulator。
acc = tl.dot(a, b, acc)  # 通常位于K维循环内，这里仅展示结构。

# 在写回前融合轻量Vector后处理。
acc = tl.where(acc >= 0, acc, 0.01 * acc)
c = acc.to(tl.float16)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

简单CV融合开发时要保持边界清晰：Cube负责产生较大的二维accumulator，Vector负责同一tile内的逐元素或小规模归约。若Vector部分需要跨多个Cube tile共享状态，就需要引入同步、workspace或拆分kernel。

## CV融合复杂算子开发

复杂CV融合可参考 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 中的best practice：

- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py)：Decode attention中QK/PV使用Cube，softmax、mask、指数、归一化和离散KV访存重排使用Vector。
- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md)：展示融合cat、slice、conv1d update时如何用 `extension.insert_slice`、转置和分核优化减少离散访存与padding开销。

复杂CV融合建议按数据流分层组织：

1. **主计算层**：识别哪些步骤必须走Cube，例如QK、PV、GEMM、batched matmul。
2. **Vector后处理层**：识别softmax、activation、mask、scale、normalization、cat/slice、layout transform等是否能在同一tile内完成。
3. **访存重排层**：对离散KV cache、MoE token重排、短尾轴（最后一个维度）tensor，优先在UB中用 `extension.insert_slice`、`extension.extract_slice`、转置或借轴转置形成硬件友好的连续访问。
4. **流水和同步层**：通过 `multibuffer`、`set_workspace_multibuffer`、`tile_mix_vector_loop`、`tile_mix_cube_loop` 等编译选项探索Cube与Vector的重叠执行。
5. **分核层**：CV融合算子通常按Cube Core数量发射grid；运行时会以约1:2的比例协同Vector Core。不要简单沿用GPU上的大grid。

对于attention类CV融合，推荐先让非causal、短序列、小head_dim的case跑通，再逐步加入：

- causal mask分阶段处理。
- 长序列K/V block循环。
- `m_i`/`l_i` 的数值稳定softmax更新。
- HEAD_DIM较大时的accumulator workspace和子块切分。
- KV cache离散索引下的load重排。

复杂CV融合调优时，优先观察profiling中Cube、Vector、MTE2的时间占比。如果Cube等待Vector，考虑减少Vector后处理粒度或打开CV balance相关选项；如果Vector等待搬运，优先检查离散访存、tail-axis padding和multibuffer配置。
