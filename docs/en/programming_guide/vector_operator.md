# Vector Operator Development

Vector operators are mainly executed by Vector Cores. Typical examples include element-wise computation, row-wise reduction, type conversion, gather/scatter, masked update, and small fused operators without `tl.dot`. The key is not to create as many grid programs as possible, but to keep the launch close to the number of physical Vector Cores and let each program process multiple tiles in an inner loop.

## Simple Vector Operator Development

For a simple Vector operator, start with the [Vector Addition example](../examples/01_vector_add_example.md) or `third_party/ascend/tutorials/01-vector-add.py`. The basic pattern is:

1. Build contiguous offsets for the current tile with `tl.arange`.
2. Use a tail mask to guard load/store.
3. Compute the element-wise expression and store the result.
4. If the grid is much larger than the physical core count, set the grid to `num_vectorcore` and process tiles with `range(pid, num_blocks, num_core)` inside the kernel.

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, num_core):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
```

Check these items first:

- **Data type**: Ascend Vector units have different performance for integer types. Prefer `int32` for indices, lengths, and offsets when precision allows. See `triton-ascend-ops/tutorial/basic/001-vector_add.zh.md` and `002-vector_cmp.zh.md`.
- **BLOCK_SIZE**: Keep it as large as possible without exceeding UB capacity. If UB overflow occurs, reduce the tile size or split it into sub-blocks.
- **Core count**: GPU-style small tiles with very large grids often cause repeated dispatch overhead on NPUs.

## Complex Vector Operator Development

Complex Vector operators usually combine irregular memory access, token reordering, multiple outputs, or long hidden dimensions. Useful references in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) include:

- [`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py)
- [`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py)
- [`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py)

Use this structure:

1. Split outer tasks by physical Vector Core count.
2. Split the hidden dimension by UB capacity with `BLOCK_X`.
3. Use `SUB_BLOCK_SIZE` to batch small irregular tasks.
4. Use `tl.insert_slice` to assemble UB-local blocks and `tl.extract_slice` to scatter sub-blocks.
5. Keep index masks, column masks, and expert/bin boundary masks separate, and combine them only at load/store sites.

Typical UB budgeting:

```python
num_core = get_npu_properties()["num_vectorcore"]
block_size = triton.cdiv(indices_length, num_core)
block_x = round_up(min(num_columns, max_block_x), 16)
sub_block_size = max((ub_budget - block_x * element_bytes) //
                     (block_x * element_bytes + index_bytes), 1)
```

When performance is poor, first check whether the grid is much larger than the physical Vector Core count, whether irregular GM access can be converted into "bulk load to UB and select in UB", whether the tail axis is 32B aligned, and whether `BLOCK_X` or `SUB_BLOCK_SIZE` causes UB overflow or too-small transfer granularity.
