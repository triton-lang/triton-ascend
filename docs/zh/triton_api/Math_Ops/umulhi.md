# triton.language.umulhi

## 1. 函数概述

简介：计算x和y的2N位乘积中每个元素的最显著N位。

```python
triton.language.umulhi(x, y)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `y`        | `tensor`          | 张量数据                                                      |

返回值：
输出张量的shape与输入x的shape相同

### 2.2 OP 规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | × | × | × | × | × | √ | × | √ | × | × | × | × | × | × | × |
| Ascend A2/A3 | × | × | × | × | × | √ | × | × | × | × | × | × | × | × | × |
| Ascend 950 | × | × | × | × | × | √ | × | × | × | × | × | × | × | × | × |

结论：
- Ascend A2/A3/950 对比 GPU 缺失 int64 的支持能力。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~8维 tensor |
| Ascend A2/A3 | 仅支持 1~8维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 8 维张量。

### 2.3 特殊限制说明

int64不支持

### 2.4 使用方法

以下示例实现了对输入张量 `x` 和 `y` 的乘积取最显著N位：

```python
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)
```
