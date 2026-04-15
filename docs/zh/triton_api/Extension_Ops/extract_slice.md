# triton.language.extract_slice
## 1. OP 概述

简介：从输入张量中按照操作指定的偏移量、大小和步幅参数提取一个张量。
原型：

```python
triton.language.extract_slice(
	ful,
	offsets,
	sizes,
	strides,
	_builder=None,
	_generator=None
)→ tensor
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `ful`        | `tensor`          | 要提取切片的源张量                                                     |
| `offsets`       | `tuple of ints`    | 切片在各个维度上的起始偏移量                                                        |
| `sizes`     | `tuple of ints`    | 切片在各个维度上的大小 |
| `strides` | `tuple of ints` | 切片在各个维度上的步长                                             |
| `_builder` |- | 保留参数，暂不支持外部调用                                            |
| `_generator`   | -               | 保留参数，暂不支持外部调用                                                |

返回值：
`tensor`：提取的切片张量

### 2.2 支持规格

#### 2.2.1 DataType 支持


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 |  bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |



#### 2.2.2 Shape 支持

支持任意形状的张量，但切片尺寸不能超过源张量对应维度的尺寸

### 2.3 特殊限制说明

无特殊限制

### 2.4 使用方法

以下示例实现了从计算结果中提取前32个元素：

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
    out_sub = tl.extract_slice(output, [block_start], [32], [1])
    out_idx = block_start + tl.arange(0, 32)
    out_msk = out_idx < n_elements
    tl.store(output_ptr + out_idx, out_sub, mask=out_msk)
```
