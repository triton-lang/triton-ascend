# triton.language.umulhi
## 1. 函数概述

简介：计算x和y的2N位乘积中每个元素的最显著N位。

```python
triton.language.umulhi(x, y, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `y`        | `tensor`          | 张量数据                                                      |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用           |

返回值：
`x`：输出张量的shape与输入x的shape相同

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×     | ×      | ×      | √     | ×    | ×    | ×    | ×    | ×    |
| Ascend A2/A3 | ×    | ×     | √     | ×     | ×     | ×      | ×      | ×     | ×    | ×    | ×    | ×    | ×    |

结论：Ascend 比 GPU 少了int64的支持。
torch_npu对u8的支持。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

int64不支持



### 2.4 使用方法

以下示例实现了对输入张量 `x` 做显著N位：

```python
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)
```
