# triton.language.topk

## 1. 函数概述

简介：返回输入张量 `x` 沿指定维度的前 `k` 个最大元素，返回结果按从大到小排序。

```python
triton.language.topk(x, k, dim: constexpr | None = None)
```

## 2. 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| ------ | ---- | ---- |
| `x` | `tensor` | 输入张量 |
| `k` | `int` | 要返回的top元素数量，必须是 2 的幂 |
| `dim` | `constexpr int` 或 `None` | 要查找 top k 元素的维度；该参数需要在编译期确定；如果为 `None`，则使用最后一个维度；当前仅支持最后一个维度 |

返回值：
`out`：输出张量的 shape 与输入张量一致，但指定维度长度变为 `k`

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | ×      | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √ | √ | × | × | × | × | × | × | √ | √ | × | √ | × |

结论：Ascend 相比 GPU 缺失 int32、uint8、int64、float64、bool 支持。
torch_npu 支持 uint8。

#### 2.2.2 Shape 支持

|        | 支持维度范围 |
| ------ | ------------ |
| GPU    | 仅支持 1~5 维 tensor |
| Ascend A2/A3 | 仅支持 1~5 维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

毕升编译器限制，int32、uint8、int64、float64、bool 无法实现。

当前 `topk` 仅返回最大值，不支持通过参数切换为返回最小值。
`dim` 仅支持最后一个维度。
`k` 必须为 2 的幂。

### 2.4 使用方法

以下示例实现了对输入张量 `x` 沿最后一个维度取前 `k` 个最大元素：

```python
@triton.jit
def topk_kernel_2d(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid
    offs_n = tl.arange(0, N)
    offs = offs_m * N + offs_n
    x = tl.load(X + offs)
    z = tl.topk(x, K, dim=0)
    tl.store(Z + offs_m * K + tl.arange(0, K), z)
```
