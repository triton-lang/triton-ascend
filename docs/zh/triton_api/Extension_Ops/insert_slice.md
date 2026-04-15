# triton.language.insert_slice
## 1. OP 概述

简介：将一个张量（子张量）插入到另一个张量的指定位置，即将一个张量按照操作指定的偏移量、大小和步幅参数插入到另一个张量中。
原型：

```python
triton.language.insert_slice(
	ful,
	sub,
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
| `ful`        | `tensor`          | 接收插入的目标张量                                                     |
| `sub`       | `tensor`    | 要插入的子张量，其形状必须与`sizes`参数指定的形状匹配                                                        |
| `offsets`     | `tuple of ints`    | 指定在`ful`张量中插入的起始偏移量（每个维度） |
| `sizes` | `tuple of ints` | 指定插入区域的大小（每个维度）                                             |
| `strides` | `tuple of ints` | 指定插入区域的步长（每个维度）                                            |
| `_builder` |- | 保留参数，暂不支持外部调用                                            |
| `_generator`   | -               | 保留参数，暂不支持外部调用                                                |

返回值：
`tensor`：插入子张量后的新张量

### 2.2 支持规格

#### 2.2.1 DataType 支持


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |


#### 2.2.2 Shape 支持

支持任意形状的张量，但需满足：

1. `ful`和`sub`的维度数必须相同
2. `offsets`、`sizes`、`strides`的长度必须与张量维度数相同
3. 插入区域不能超出`ful`张量的边界

### 2.3 特殊限制说明

无特殊限制

### 2.4 使用方法

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
    x_sub = tl.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = tl.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    # 加载原始输出张量
    output = tl.load(output_ptr + offsets, mask=mask)
    # 将计算结果插入回原张量
    output = tl.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 3. 语义GAP

无语义差异
