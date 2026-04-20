# div

## 1. OP 概述

简介：除法，四则运算 ‘/’，无tl.div方法

底层实现与fdiv算子相同，只不过fdiv明确限制入参必须是float类型，‘/’无此限制，它会把非浮点型转换为浮点型再做计算

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `tensor or Number`     |     第一个入参 ，被除数    |                                                       |
| `other`       | `tensor or Number`     |     第二个入参 ，除数    |                                                   |

返回值：
`tl.tensor`：除法结果
返回结果类型：总是返回浮点型

| 输入类型            | 处理方式                 | 结果类型      |
| --------------------- | -------------------------- | --------------- |
| `int / int`     | 两个都转成 `float32` | `float32` |
| `int / float`   | int 转 float             | float 类型    |
| `float / float` | 统一到更高精度 float     | 高精度 float  |
| `float / int`   | int 转 float             | float 类型    |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| × | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 无限制 |
| Ascend A2/A3 |无限制  |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

Ascend A3 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持

### 2.4 使用方法

以下示例实现了对输入张量 `in_ptr0, in_ptr1` 做除法计算：

```python
@triton.jit
def triton_div(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):     
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 / tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```
