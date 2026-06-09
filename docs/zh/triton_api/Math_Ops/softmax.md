# triton.language.softmax

## 1. 函数概述

简介：计算x的逐元素softmax。

```python
triton.language.softmax(x, dim=None, keep_dims=False, ieee_rounding=False)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `dim`        | `int`          | 指定在哪个维度上计算 softmax                                                      |
| `keep_dims`        | `bool`          | 控制计算后是否保留原维度的形状                                                  |
| `ieee_rounding`   | `bool`                 | 控制浮点数运算是否遵循 IEEE 754 标准的舍入规则          |

返回值：
`x`：与x相同的shape的张量

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     |  √    | √    | √    |  √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

结论：Ascend 比 GPU 少了fp64的支持。
torch_npu不支持u8。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

无。

### 2.4 使用方法

以下示例实现了对输入张量 `x` 做逐元素softmax：

```python
@triton.jit
def tt_softmax_3d(in_ptr, out_ptr,
                  xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                  XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a)

    tl.store(out_ptr + idx, ret)
```
