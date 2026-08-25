# gemm_swizzle2d_Nz

将 GEMM 迭代序号转换为输出矩阵的二维 tile 坐标。

tile 按列分组，相邻组反向遍历行，形成蛇形调度。最后一组不足
`swizzle_offset` 列时会自动处理。

**签名**

```python
gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
)
```

**参数**

| 参数 | 含义 |
| --- | --- |
| `iter_id` | 当前迭代序号。范围为 `[0, ceil(data_row_shape / tile_row_shape) * ceil(data_col_shape / tile_col_shape))`。 |
| `data_row_shape` | 输出矩阵的行数，通常为 `M`。 |
| `data_col_shape` | 输出矩阵的列数，通常为 `N`。 |
| `tile_row_shape` | tile 行数，通常为 `BLOCK_SIZE_M`。 |
| `tile_col_shape` | tile 列数，通常为 `BLOCK_SIZE_N`。 |
| `swizzle_offset` | 每组包含的列 tile 数量，默认为 `7`。 |

**返回值**

返回 `(data_row_idx, data_col_idx)`，即 tile 的行索引和列索引。

**调度顺序**

例如，3 行、10 列的 tile 网格在 `swizzle_offset=7` 时：前 7 列按
`0 -> 1 -> 2` 遍历行，后 3 列按 `2 -> 1 -> 0` 遍历。

**使用场景**

用于分布式融合 kernel 的 GEMM 阶段。

- 将相邻列的 tile 分组，有助于提高矩阵数据的缓存局部性。
- 蛇形遍历让相邻分组在同一行衔接，减少 tile 坐标的大跨度跳转。
- 返回的 `(block_id_m, block_id_n)` 可直接用于计算 A、B、C 的分块地址。

**注意事项**

- shape、tile 和 `swizzle_offset` 参数必须大于 `0`。
- 调用方仍需为矩阵边界生成 mask。
- 接口只调整调度顺序，不改变数据布局。

**示例**

```python
import triton
import triton.language as tl
from triton_dist.language.extra.ascend.algorithm import gemm_swizzle2d_Nz


@triton.jit
def gemm_tile_kernel(c_ptr, M, N, stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    for iter_id in range(pid, num_tiles, ncore):
        block_m, block_n = gemm_swizzle2d_Nz(
            iter_id, M, N, BLOCK_M, BLOCK_N,
        )
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # 使用 offs_m 和 offs_n 计算 A、B、C 的分块地址。
```
