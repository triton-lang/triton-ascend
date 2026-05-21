# triton.language.cumsum

## 1. OP 概述

简介：`triton.language.cumsum` 计算输入tensor沿指定轴的累积和，返回累积求和结果。

```python
triton.language.cumsum(input, axis=0, reverse=False)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `input` | `Tensor` | 输入tensor |
| `axis` | `int` | 沿着哪个维度进行累积和操作，默认为0 |
| `reverse` | `bool` | 如果为True，沿反方向进行累积和操作 |

`cumsum` 函数计算沿指定轴的累积和（前缀和）。例如，对于输入 `[a, b, c, d]`，累积和结果为 `[a, a+b, a+b+c, a+b+c+d]`。

当 `reverse=True` 时，计算反向累积和：`[a+b+c+d, b+c+d, c+d, d]`。

返回值：
`tensor`：计算输入tensor沿指定轴的累积和，返回累积求和结果。

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape 支持

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

### 2.4 使用方法

以下示例实现了对2Dshape的tensor进行cumsum运算：

```python
@triton.jit
def triton_kernel_2d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)
```
