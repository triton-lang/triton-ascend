# triton.language.semantic.greater_equal

## 1. OP 概述

简介：用于比较两个张量的元素, 与`>=`等价。
函数原型：

```python
triton.language.semantic.greater_equal(
 input: tl.tensor, 
 other: tl.tensor, 
 builder: ir.builder
) -> tl.tensor
```

作为`tensor`的内置运算符使用, 如`x>=y`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `input` | `tensor` | 张量数据, 左操作数, 代表要进行比较的主数据 |
| `other`   | `tensor` | 张量数据, 右操作数, 与`input`逐元素进行比较 |
| `_builder` | - | 保留参数，暂不支持外部调用 |

返回值：
`tl.tensor`：同`input`的shape的张量

### 2.2 支持规格

#### 2.2.1 DataType 支持

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | √ | √ | × | √ | √ |

结论：Triton-Ascend 对比 GPU 缺失uint, fp64,  fp8的支持能力。

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| -------- | ---------------------- |
| GPU    | 无限制 |
| Ascend A2/A3 | 无限制 |

结论：在 Shape 方面, GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Triton-Ascend 对比 GPU 缺失fp64的支持能力。uint, fp8支持中。

### 2.4 使用方法

以下示例实现了对三维张量`x0`、`x1`做大于等于运算：

```python
@triton.jit
def triton_ge_3d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 >= x1
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```
