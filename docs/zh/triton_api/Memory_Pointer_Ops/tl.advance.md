# triton.language.advance
## 1. OP 概述

简介：将`tl.make_block_ptr`的offset增加一个偏移量
原型：

```python
triton.language.advance(
	base: triton.PointerType,
	offsets: tuple(int | constexpr),
	_semantic=None
)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `base`        | `triton.PointerType`          |需要被更新的指针，`tl.make_block_ptr`的结果                                |
| `offsets`     | `tuple(int \| constexpr)`    | 张量各维度的基址偏移量列表，`len(offsets)`需要与`len(base.offsets)`相等 |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用|

返回值：`pointer_type<blocked<shape, element_type>>`： 指向tensor的指针

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    | ×    |

结论：Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

- Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。
- Ascend只允许通过调整`order`参数的顺序来表达转置语义，不能通过调整`stride`参数的顺序实现转置语义。
- 当前`tl.make_tensor_ptr`，如果与较复杂的循环和分支语句搭配使用，可能会出现编译问题

### 2.4 使用方法

参考以下示例：

```python
@triton.jit
def fn_npu_3d(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(3, 1, 2),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    bbptr = tl.advance(block_ptr_in, (-3, -1, -2))
    # XB,YB,1
    X = tl.load(bbptr)

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, X)
```
