# triton.language.zeros

## 1. OP 概述

简介：`triton.language.zeros`返回用给定形状和dtype的标量值0填充的张量。

```python
triton.language.zeros(shape, dtype)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                  | 说明                                   |
| ------------- | ----------------- | ---------------------------- |
| `shape`           | `tuple of ints`               | 新数组的形状，例如 (8, 16) 或 (8, ) |
| `dtype`            | `tl.dtype`               |  新数组的数据类型，例如 tl.float16  |

返回值：
`tensor`：返回用给定形状和dtype的标量值0填充的张量。

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |

#### 2.2.2 Shape 支持

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现
> 无

### 2.4 使用方法

以下示例实现了返回填充了0数值的(XB,YB,ZB)shape的tensor：

```python
@triton.jit
def fn_f32(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.zeros((XB, YB, ZB), dtype=tl.float32)

    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)
```
