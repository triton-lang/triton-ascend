# Vector 算子开发

Vector 算子主要由 Vector Core 执行，典型形态包括逐元素计算、行级归约、类型转换、Gather/Scatter、Mask 更新以及不含 `tl.dot` 的小型融合算子。开发重点不是把 grid 切得越细越好，而是在固定物理 Vector Core 数量的前提下，让每个 program 在核内循环处理多个 tile。

## Vector 简单算子开发

简单 Vector 算子可以从本仓的 [向量相加样例](../examples/01_vector_add_example.md) 或 `third_party/ascend/tutorials/01-vector-add.py` 入手。该类算子的基本步骤如下：

1. 用 `tl.arange` 构造当前 tile 的连续偏移。
2. 用 `mask` 保护尾块，避免越界 load/store。
3. 完成逐元素计算后写回结果。
4. 当 grid 数远大于物理核数时，将 grid 固定为 `num_vectorcore`，在 kernel 内用 `range(pid, num_blocks, num_core)` 分批处理。

基础 kernel 结构如下：

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

开发时优先检查三类问题：

- **数据类型**：Ascend Vector 单元对不同整数类型的支持和性能不同。对于不影响精度的索引、长度、偏移类数据，优先使用 `int32`，可参考 `triton-ascend-ops/tutorial/basic/001-vector_add.zh.md` 和 `002-vector_cmp.zh.md`。
- **BLOCK_SIZE**：BLOCK_SIZE 需要在 UB 容量内尽量大。若出现 UB overflow，先降低单次处理元素数，再考虑拆分子块。
- **分核数**：NPU 物理 Vector Core 数量通常为几十个。小 tile 大 grid 的 GPU 写法迁移到 NPU 时，容易因多轮下发带来明显开销。

## Vector 复杂算子开发

复杂 Vector 算子通常不是单个逐元素表达式，而是带有离散访存、批量重排、多个输出或长 hidden size 的组合逻辑。可参考 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 中的以下案例：

- [`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py)：Megablocks gather/scatter/scatter_wgrad 的 Ascend 亲和实现。
- [`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py)：按 expert/bin 分组后的 gather/scatter。
- [`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py)：带 padding 的 MoE gather/scatter。

这类算子的组织方式通常是：

1. **按物理核切分外层任务**：用 `num_vectorcore` 作为 grid，每个 program 负责一段 indices 或 token。
2. **按 UB 容量切分 hidden 维**：对 `NUM_COLUMNS` 使用 `BLOCK_X` 分块，并预留 double buffer、索引和临时张量的空间。
3. **用 `SUB_BLOCK_SIZE` 合并小粒度离散任务**：一次加载一组 indices，在 UB 中组织成连续临时块，减少 GM 标量访存和多次 store。
4. **用扩展语义管理 UB 内局部数据**：使用 `tl.insert_slice` 合并多行数据，使用 `tl.extract_slice` 取出子块后再分散写回。
5. **为尾块保留统一 mask**：复杂 gather/scatter 中同时存在 index mask、column mask 和 expert/bin 边界，建议分别命名并只在 load/store 处组合。

典型的 UB 预算思路如下：

```python
num_core = get_npu_properties()["num_vectorcore"]
block_size = triton.cdiv(indices_length, num_core)
block_x = round_up(min(num_columns, max_block_x), 16)
sub_block_size = max((ub_budget - block_x * element_bytes) //
                     (block_x * element_bytes + index_bytes), 1)
```

当复杂 Vector 算子性能不达预期时，优先从以下方向排查：

- grid 是否远大于物理 Vector Core 数，导致多轮下发。
- 离散访存是否可转化为“批量搬入 UB 后在 UB 内选择”。
- 尾轴是否满足 32B 对齐；不满足时是否可用转置或借轴转置规避自动 padding。
- `BLOCK_X` 和 `SUB_BLOCK_SIZE` 是否造成 UB overflow 或过小的搬运粒度。
