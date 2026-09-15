# al.insert_slice

## 1. 函数概述

将一个张量（子张量）插入到另一个张量的指定位置，即将一个张量按照操作指定的偏移量、大小和步幅参数插入到另一个张量中。

```python
al.insert_slice(
    ful,
    sub,
    offsets,
    sizes,
    strides,
    _semantic=None,
    _generator=None
) -> tensor
```

可以作为 tensor 的成员函数调用，如 `x.insert_slice(...)`，与 `insert_slice(x, ...)` 等效。

## 2. 规格

### 2.1 参数说明

| 参数名       | 类型            | 说明                                                   |
| ------------ | --------------- | ------------------------------------------------------- |
| `ful`        | `tensor`        | 接收插入的目标张量                                       |
| `sub`        | `tensor`        | 要插入的子张量，其形状应与 `sizes` 参数指定的形状匹配     |
| `offsets`    | `tuple of ints` | 指定在 `ful` 张量中插入的起始偏移量（每个维度）           |
| `sizes`      | `tuple of ints` | 指定插入区域的大小（每个维度）                           |
| `strides`    | `tuple of ints` | 指定插入区域的步长（每个维度）                           |
| `_semantic`  | -               | 保留参数，暂不支持外部调用                               |
| `_generator` | -               | 保留参数，暂不支持外部调用                               |

返回值：

`tensor`：插入子张量后的新张量

### 2.2 说明

Python 前端实际做了以下断言校验（`vec_ops.py::insert_slice`）：

1. `ful` 的 rank 必须大于 0，且 `ful` 和 `sub` 的维度数（rank）必须相同；
2. `offsets`、`sizes`、`strides` 的长度必须与 `ful` 的维度数相同；
3. `sizes` 中的元素需要大于等于 1，`strides` 中的元素需要大于等于 0（注意 `strides` 允许为 0）。

### 2.3 DataType 支持

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| --- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | ×    | √    | √    |
| Ascend 950 | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | ×    | √    | √    |

整型/浮点类型在 A2/A3 与 Ascend 950 上均能正常编译、运行，结果与直接下标赋值一致；`fp64` 编译报错；`bool`（含 kernel 内直接构造的 `tl.int1`）在两代架构上均能正确插入（需要 CANN ≥ 9.1.0）。

## 3. 使用方法

以下示例实现了将切片计算结果插入回原张量：

```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # 提取切片
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    # 加载原始输出张量
    output = tl.load(output_ptr + offsets, mask=mask)
    # 将计算结果插入回原张量
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```
