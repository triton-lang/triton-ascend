# triton.language.sum

## 1. OP 概述

简介：`triton.language.sum` 计算输入tensor沿指定轴的元素和，返回求和结果。

```python
triton.language.sum(input, axis=None, keep_dims=False)
```

> **版本差异说明**
>
> `dtype` 参数为社区 Triton 3.5.0 引入的功能。当前发布的 Triton-Ascend 基于社区 Triton 3.2.0 开发，没有`dtype` 参数。后续升级至社区 Triton 3.5.0 版本时，将完整支持 `dtype` 参数的功能。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `input` | `Tensor` | 输入tensor |
| `axis` | `int` 或 `None` | 沿着哪个维度进行求和操作。如果为None，则对所有维度求和 |
| `keep_dims` | `bool` | 如果为True，保持被求和的维度为长度1 |

返回值：
`tensor`：计算输入tensor沿指定轴的元素和，返回求和结果。

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
> `dtype` 参数当前版本暂未支持。社区 Triton 3.5.0 中，`dtype` 参数用于控制求和运算的累加数据类型：未指定时，位宽小于 32 的整数类型会自动提升为 `int32`/`uint32` 以避免溢出；显式指定时，输入会先转换为指定类型再执行求和。当前 Triton-Ascend 基于社区 Triton 3.2.0，该类型提升逻辑尚未支持，将在后续升级至 3.5.0 版本时完整支持。

### 2.4 使用方法

以下示例实现了对2Dshape的tensor进行sum运算：

```python
@triton.jit
def tt_sum_2d(in_ptr, out_ptr,
              xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
              XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.sum(x, dim)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)

```
