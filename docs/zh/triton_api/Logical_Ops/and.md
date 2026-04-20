# triton.language.and

## 1. 函数概述

简介：计算两个元素的与值。

```python
triton.language.and(x, y, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `y`        | `tensor`          | 张量数据                                                      |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用

返回值：
`out`：同x、y的shape的张量

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √     | √      | √     | √      | ×      | ×       | ×       | √      | ×    | ×    | ×    | ×    | √    |
| Ascend A2/A3 | √    | √     | √     | √     | ×     | ×      | ×      | √     | ×    | ×    | ×    | ×    | √    |

结论：Ascend 比 GPU 少了uint类型的支持。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 比 GPU 少了uint类型的支持。

### 2.4 使用方法

以下示例实现了对输入张量 `x` 做逐元素指数（以2为底）：

```python@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
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
    Y = tl.load(y_ptr + idx)

    ret = X & Y

    tl.store(output_ptr + idx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
y = test_common.generate_tensor(shape, dtype).npu()
```
