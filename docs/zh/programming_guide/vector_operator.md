# Vector算子开发

Vector算子主要由Vector Core执行，典型形态包括逐元素计算、行级归约、类型转换、Gather/Scatter、Mask更新以及不含 `tl.dot` 的小型融合算子。开发重点不是把grid切得越细越好，而是在固定物理Vector Core数量的前提下，让每个program在核内循环处理多个tile。

## Vector简单算子开发

简单Vector算子可以从本仓的 [向量相加样例](../examples/01_vector_add_example.md) 或 [`third_party/ascend/tutorials/01-vector-add.py`](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py) 入手。该类算子的基本步骤如下：

1. 用 `tl.arange` 构造当前tile的连续偏移。
2. 用 `mask` 保护尾块，避免越界load/store。
3. 完成逐元素计算后写回结果。
4. 当grid数远大于物理核数时，将grid固定为 `num_vectorcore`，在kernel内用 `range(pid, num_blocks, num_core)` 分批处理。

基础kernel结构如下：

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

- **数据类型**：Ascend Vector单元对不同整数类型的支持和性能不同。对于不影响精度的索引、长度、偏移类数据，优先使用 `int32`，可参考 [`001-vector_add.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/001-vector_add.zh.md) 和 [`002-vector_cmp.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/002-vector_cmp.zh.md)。
- **BLOCK_SIZE**：BLOCK_SIZE需要在UB容量内尽量大。若出现UB overflow，先降低单次处理元素数，再考虑拆分子块。
- **分核数**：NPU物理Vector Core数量通常为几十个。小tile大grid的GPU写法迁移到NPU时，容易因多轮下发带来明显开销。

## Vector复杂算子开发

复杂Vector算子通常不是单个逐元素表达式，而是带有离散访存、批量重排、多个输出或长hidden size的组合逻辑。可参考 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 中的以下案例：

- [`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py)：Megablocks gather/scatter/scatter_wgrad的Ascend亲和实现。
- [`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py)：按expert/bin分组后的gather/scatter。
- [`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py)：带padding的MoE gather/scatter。

这类算子的组织方式通常是：

1. **按物理核切分外层任务**：用 `num_vectorcore` 作为grid，每个program负责一段indices或token。
2. **按UB容量切分hidden维**：对 `NUM_COLUMNS` 使用 `BLOCK_X` 分块，并预留double buffer、索引和临时张量的空间。
3. **用 `SUB_BLOCK_SIZE` 合并小粒度离散任务**：一次加载一组indices，在UB中组织成连续临时块，减少GM标量访存和多次store。
4. **用扩展语义管理UB内局部数据**：先通过 `import triton.language.extra.cann.extension as extension` 引入扩展模块，再使用 `extension.insert_slice` 合并多行数据，使用 `extension.extract_slice` 取出子块后再分散写回。
5. **为尾块保留统一mask**：复杂gather/scatter中同时存在index mask、column mask和expert/bin边界，建议分别命名并只在load/store处组合。

典型的UB预算思路如下：

```python
# 获取NPU硬件属性，详见“通用多核任务并行”一节
import torch_npu
import triton.runtime.driver as driver
device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
num_core = properties["num_vectorcore"]

block_size = triton.cdiv(indices_length, num_core)
align_elems = 16
block_x = triton.cdiv(min(num_columns, max_block_x), align_elems) * align_elems
sub_block_size = max((ub_budget - block_x * element_bytes) //
                     (block_x * element_bytes + index_bytes), 1)
```

当复杂Vector算子性能不达预期时，优先从以下方向排查：

- grid是否远大于物理Vector Core数，导致多轮下发。
- 离散访存是否可转化为“批量搬入UB后在UB内选择”。
- 尾轴（最后一个维度）是否满足32B对齐；不满足时是否可用转置或借轴转置规避自动padding。
- `BLOCK_X` 和 `SUB_BLOCK_SIZE` 是否造成UB overflow或过小的搬运粒度。
