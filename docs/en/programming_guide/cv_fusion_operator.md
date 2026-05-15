# CV Fusion Operator Development

CV fusion operators use Cube Cores and Vector Cores in the same operator. Cube Cores usually handle `tl.dot`, matrix multiplication, or convolution-like main computation, while Vector Cores handle bias, activation, softmax, reductions, masks, layout reorganization, or cross-block synchronization. The goal is to reduce kernel boundaries and GM round trips while controlling Cube tiles, Vector tiles, UB/L1 usage, and synchronization.

## Simple CV Fusion Operator Development

For simple CV fusion, start with matmul plus activation in `third_party/ascend/tutorials/03-matrix-multiplication.py` or the [Fused Attention example](../examples/04_fused_attention_example.md).

1. Implement a stable Cube main computation such as `acc = tl.dot(a, b, acc)`.
2. Fuse lightweight Vector post-processing before storing the accumulator, such as bias, scale, activation, or dtype cast.
3. Split large accumulators into sub-blocks to avoid UB overflow in the Vector post-processing stage.
4. If one Cube output block needs to be split across Vector sub-blocks, use `extension.parallel(..., bind_sub_block=True)` and `extension.extract_slice`.

```python
acc = tl.dot(a, b, acc)

SUB_M: tl.constexpr = BLOCK_M // 2
for s in extension.parallel(0, 2, bind_sub_block=True):
    acc_sub = extension.extract_slice(acc, (s * SUB_M, 0), (SUB_M, BLOCK_N), (1, 1))
    acc_sub = tl.where(acc_sub >= 0, acc_sub, 0.01 * acc_sub)
    c_sub = acc_sub.to(tl.float16)
    tl.store(c_ptrs_for_sub_block, c_sub, mask=c_mask_for_sub_block)
```

Keep the boundary simple: Cube produces a 2D accumulator, and Vector performs element-wise or small reductions within the same tile. If Vector logic needs state shared across multiple Cube tiles, introduce synchronization, workspace, or split the kernel.

## Complex CV Fusion Operator Development

Useful best-practice references in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops):

- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py): QK/PV use Cube, while softmax, masks, exponentiation, normalization, and discrete KV-cache reorganization use Vector.
- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md): shows how to reduce irregular memory access and padding overhead with `insert_slice`, transpose, and core allocation optimization.

Organize complex CV fusion by data flow:

1. **Main compute layer**: identify the steps that must use Cube, such as QK, PV, GEMM, or batched matmul.
2. **Vector post-processing layer**: identify softmax, activation, mask, scale, normalization, cat/slice, and layout transforms that can finish within the same tile.
3. **Memory reorganization layer**: for discrete KV cache, MoE token reordering, or short tail-axis tensors, use `insert_slice`, `extract_slice`, transpose, or axis borrowing in UB to form hardware-friendly continuous access.
4. **Pipeline and synchronization layer**: explore Cube/Vector overlap with options such as `multibuffer`, `set_workspace_multibuffer`, `tile_mix_vector_loop`, and `tile_mix_cube_loop`.
5. **Core allocation layer**: CV fusion operators usually launch by Cube Core count, while Vector Cores cooperate at roughly a 1:2 ratio. Do not directly reuse large GPU grids.

For attention-style CV fusion, start with non-causal, short-sequence, small-head-dimension cases, and then add causal mask stages, long-sequence K/V loops, numerically stable `m_i`/`l_i` softmax updates, accumulator workspace for large `HEAD_DIM`, and load reorganization for discrete KV-cache indices.

When tuning complex CV fusion, inspect the Cube, Vector, and MTE2 time ratios in profiling. If Cube waits for Vector, reduce the Vector post-processing granularity or enable CV balance options. If Vector waits for data movement, check irregular access, tail-axis padding, and multibuffer settings first.
