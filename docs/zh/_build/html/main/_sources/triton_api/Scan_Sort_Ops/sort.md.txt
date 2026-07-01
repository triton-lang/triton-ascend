# triton.language.sort

## 1. 函数概述

简介：对输入张量`x`按维度进行升序或者降序的排序。

```python
triton.language.sort(x, dim: constexpr | None = None, descending: constexpr = constexpr[0])
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `dim`        | `int`          | 排序维度                                                      |
| `descending`        | `bool`          | 是否降序                                                      |

返回值：
`x`：输出张量的shape与输入x的shape相同

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √      | √     | ×     | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √     | √      | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

结论：Ascend 比 GPU 少了int32，uint8，int64，fp64，bool的支持。
torch_npu支持u8。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

毕升编译器限制，int32，uint8，int64，fp64，bool无法实现。

### 2.4 使用方法

以下示例实现了对输入张量 `x` 做排序：

```python
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    offy = pid * M
    off2d = offx + offy
    x = tl.load(X + off2d)
    x = tl.sort(x, descending=descending, dim=0)
    tl.store(Z + off2d, x)
```
