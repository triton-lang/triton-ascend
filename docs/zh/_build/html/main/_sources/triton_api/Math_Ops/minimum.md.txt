# triton.language.minimum

## 1. 函数概述

简介：计算x和y的逐元素最小值。

```python
triton.language.minimum(x, y, propagate_nan: ~triton.language.core.constexpr = <PROPAGATE_NAN.NONE: 0>, _semantic=None)¶
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `y`       | `tensor`    | 张量数据                                                        |
| `propagate_nan`       | `tl.PropagateNan`    | 是否传播NaN值                                                        |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用

返回值：
`x`：与输入x的shape相同的张量

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | ×     | ×     | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | √     | ×     | ×      | ×      | √     | √    | √    | ×    | √    | √    |

结论：Ascend 相比 GPU 缺失 fp64 支持。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

无。

#### 2.3.1 propagate_nan 参数限制

**注意：当 `propagate_nan=tl.PropagateNAN.NONE` 时，系统会自动添加 NaN 值处理逻辑，这会导致：**

1. **UB 空间使用增加**：额外的 NaN 检测和处理需要占用更多的 UB 空间
2. **可能的性能下降**：由于增加了额外的计算逻辑，可能导致算子执行性能下降

**建议：**

- 如果输入数据不包含 NaN 值，或不需要严格的 NaN 处理语义，建议使用默认值或根据实际需求选择合适的 `propagate_nan` 参数值
- 在 UB 空间紧张的场景下，应特别注意此参数的选择，避免因 UB 空间不足导致编译失败

### 2.4 使用方法

以下示例实现了对输入张量 `x` 和`y`的逐元素最小值：

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr,
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

    ret = tl.minimum(X, Y)

    tl.store(output_ptr + idx, ret)

```
