# triton.language.swizzle2d

## 1. 函数概述

简介：**将一个大小为 size_i × size_j 的行优先矩阵的索引，按每 size_g 行一组，分别转换为列优先矩阵的索引。**。

```python
triton.language.swizzle2d(i, j, size_i, size_j, size_g)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `i`        | `tensor`          | index索引值 ，最大值为size(i)-1                                                     |
| `j`        | `tensor`          | index索引值 ，最大值为size(j)-1                                                     |
| `size_i`        | `int`          | 整型，表示索引值i的长度                                                     |
| `size_j`        | `int`          | 整型，表示索引值j的长度                                                          |
| `size_g`        | `int`          | 整型                                                      |

返回值：
`out0，out1`：同i，j shape的张量

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×     | ×      | √     | ×      |  ×      |  ×       |  ×       | √      | ×    | ×   | ×    | ×    | ×    |
| Ascend A2/A3 | ×    | ×     | √     | ×     | ×     | ×      | ×      | √     | ×    | ×   | ×    | ×   | ×

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 2维 tensor |
| Ascend A2/A3 | 仅支持 2维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 2 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

暂无。

### 2.4 使用方法

以下示例实现了对输入张量 `x` 做逐元素指数（以2为底）：

```python@triton.jit@triton.jit@triton.jit@triton.jit
def fn_npu_(out0, out1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    i = tl.arange(0, XB)[:, None]
    j = tl.arange(0, YB)[None, :]
    ij = i * YB + j
    xx, yy = tl.swizzle2d(i, j, size_i=XB, size_j=YB, size_g=ZB)

    ptr = tl.load(out0)
    xx = tl.cast(xx, dtype=ptr.dtype)
    yy = tl.cast(yy, dtype=ptr.dtype)
    tl.store(out0 + ij, xx)
    tl.store(out1 + ij, yy)
```
