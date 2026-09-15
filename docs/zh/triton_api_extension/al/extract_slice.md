# al.extract_slice

## 1. 函数概述

从输入张量中按照操作指定的偏移量、大小和步幅参数提取一个张量。

```python
al.extract_slice(
    ful,
    offsets,
    sizes,
    strides,
    _semantic=None,
    _generator=None
) -> tensor
```

可以作为 tensor 的成员函数调用，如 `x.extract_slice(...)`，与 `extract_slice(x, ...)` 等效。

## 2. 规格

### 2.1 参数说明

| 参数名       | 类型            | 说明                         |
| ------------ | --------------- | ---------------------------- |
| `ful`        | `tensor`        | 要提取切片的源张量           |
| `offsets`    | `tuple of ints` | 切片在各个维度上的起始偏移量 |
| `sizes`      | `tuple of ints` | 切片在各个维度上的大小       |
| `strides`    | `tuple of ints` | 切片在各个维度上的步长       |
| `_semantic`  | -               | 保留参数，暂不支持外部调用   |
| `_generator` | -               | 保留参数，暂不支持外部调用   |

返回值：

`tensor`：提取的切片张量

### 2.2 说明

Python 前端实际做了以下断言校验（`vec_ops.py::extract_slice`）：

1. `ful` 的 rank 必须大于 0；
2. `offsets`、`sizes`、`strides` 的长度必须与 `ful` 的维度数相同；
3. `sizes` 中的元素需要大于等于 1，`strides` 中的元素需要大于等于 0（注意 `strides` 允许为 0）。

### 2.3 DataType 支持

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| --- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | ×    | √    | √    |
| Ascend 950 | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | ×    | √    | √    |

`extract_slice` 对所有列出的整型/浮点类型都能正常编译、运行，切片结果与直接下标索引一致，A2/A3 与 Ascend 950 结论一致，只有 `fp64` 编译报错；`bool=√`，kernel 内直接构造的 `tl.int1`（如比较运算结果，不经内存往返）也能正确处理。

## 3. 使用方法

以下示例实现了从计算结果中提取前 32 个元素：

```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 提取前32个元素
    out_sub = al.extract_slice(output, [block_start], [32], [1])
    out_idx = block_start + tl.arange(0, 32)
    out_msk = out_idx < n_elements
    tl.store(output_ptr + out_idx, out_sub, mask=out_msk)
```
