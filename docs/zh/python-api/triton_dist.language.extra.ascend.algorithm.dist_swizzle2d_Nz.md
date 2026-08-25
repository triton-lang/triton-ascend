# dist_swizzle2d_Nz

将通信迭代序号转换为数据 tile、目标 rank 和尾块大小。

不同 tile 会从不同 rank 开始通信，避免多个 AI Core 集中访问同一 rank。

**签名**

```python
dist_swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
)
```

**参数**

| 参数 | 含义 |
| --- | --- |
| `iter_id` | 当前迭代序号。范围为 `[0, rank_size * ceil(data_row_shape / tile_row_shape) * ceil(data_col_shape / tile_col_shape))`。 |
| `rank_size` | 参与通信的 rank 数量。 |
| `data_row_shape` | 数据行数。 |
| `data_col_shape` | 数据列数。 |
| `tile_row_shape` | 每个 tile 的最大行数。 |
| `tile_col_shape` | 每个 tile 的最大列数。 |
| `comm_npu_split` | 每组交错调度的 rank 数量，默认为 `1`。必须是 `rank_size` 的因数。 |

**返回值**

返回 `(data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size)`：

- `data_row_idx`、`data_col_idx`：tile 的行、列索引。
- `rank_idx`：目标 rank。
- `comm_row_size`、`comm_col_size`：当前 tile 的实际行数和列数。

**使用场景**

用于 AllGather-GEMM、GEMM-ReduceScatter 和 GEMM-AllReduce 等融合 kernel。

- 轮转不同 tile 的目标 rank，有助于减少通信热点和 rank 间的访问竞争。
- 同时返回尾块大小，可直接生成通信 mask，省去额外的边界计算。

**注意事项**

- `comm_npu_split` 必须满足 `rank_size % comm_npu_split == 0`。
- 接口只调整调度顺序，不改变数据布局。

**示例**

```python
import triton
import triton.language as tl
from triton_dist.language.extra.ascend.algorithm import dist_swizzle2d_Nz


@triton.jit
def communication_kernel(rank_size, rows, cols,
                         TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    num_iters = rank_size * tl.cdiv(rows, TILE_M) * tl.cdiv(cols, TILE_N)

    for iter_id in range(pid, num_iters, ncore):
        tile_m, tile_n, target_rank, tile_rows, tile_cols = dist_swizzle2d_Nz(
            iter_id, rank_size, rows, cols, TILE_M, TILE_N,
        )

        # target_rank 用于选择远端 rank；tile_rows 和 tile_cols 用于生成尾块 mask。
```
