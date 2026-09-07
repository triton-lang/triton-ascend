# triton.language.ceil

## 1. OP 概述

简介：计算张量中每个元素的向上取整值

```python
triton.language.ceil(x)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `x` | `tensor` | 张量数据 |

返回值：
`out`：同`x`的shape的张量

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | × | × | × | × | × | × | × | × | × | √ | √ | × | × | × | × |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | √ | × | × | × | × | × |
| Ascend 950 | × | × | × | × | × | × | × | × | × | √ | × | × | × | × | × |

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| :---: | :---: |
| GPU    | 无限制 |
| Ascend | 无限制 |

结论：在 Shape 方面, GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

结论：
- Ascend A2/A3/950 对比 GPU 缺失 fp64 的支持能力。


### 2.4 使用方法

以下示例实现了对输入张量 `x` 做向上取整运算：

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.ceil(X)

    tl.store(output_ptr + idx, ret)
```
