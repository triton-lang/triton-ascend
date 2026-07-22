# triton.language.reduce

## 1. OP 概述

简介：`triton.language.reduce` 将输入tensor根据提供轴axis，应用combine_fn计算，返回计算完的tensor。

```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `input` | `Tensor` 或 `tuple of Tensor` | 输入tensor，可以是单个tensor或tensor元组 |
| `axis` | `int` 或 `None` | 沿着哪个维度进行reduce操作。如果为None，则reduce所有维度 |
| `combine_fn` | `Callable` | 用于组合两个标量tensor组的函数（必须用@triton.jit标记） |
| `keep_dims` | `bool` | 如果为True，保持被reduce的维度为长度1 |
| `_semantic` | `Optional[str]` | （保留参数，暂不支持外部调用 |
| `_generator` | `Optional[Generator]` | 保留参数，暂不支持外部调用 |

**注意**：此函数也可以作为tensor的成员函数调用，如 `x.reduce(...)` 而不是 `reduce(x, ...)`

返回值：
`tensor`：将输入tensor根据提供的轴axis，应用combine_fn计算，返回计算完的tensor。

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ | 
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape 支持

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现
> keep_dims=True需要测试更多规格，来确定是否全面支持。目前已测3D dim=2情况下，支持 keep_dims=True。

### 2.4 使用方法

以下示例实现了对2Dshape的tensor进行reduce计算，其中的combine_fn使用简单加法：

```python
@triton.jit
def _reduce_combine(a, b):
    return a + b

@triton.jit
def tt_reduce_2d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)

```
