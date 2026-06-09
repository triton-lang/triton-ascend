# triton.language.argmin

## 1. OP 概述

简介：在指定维度上返回最小值所在的下标

```python
triton.language.argmin(input, axis, tie_break_left=True, keep_dims=False)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | --- |
| `input` | `tensor` | 张量数据，左操作数 |
| `axis`   | `int` | 指定在哪个维度上进行规约 |
| `keep_dims` | `bool` | 保持规约轴规约后的维度 |
| `tie_break_left` | `bool` | 如果多个元素有相同的最小值，返回最左侧最小值的下标 |

返回值：
`tl.tensor`：同`input`的shape的张量

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2A3系列| √ | √ | x | √ | × | √ | × | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| -------- | ---------------------- |
| GPU    | 无限制 |
| Ascend | 无限制 |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend A3对比 GPU 缺失uint16、uint32、uint64、fp64的支持能力。

### 2.4 使用方法

更多示例参考triton-ascend代码仓，ascend/examples/generalization_cases/test_argmin.py

```python
@triton.jit
def triton_argmin_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmin(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```
