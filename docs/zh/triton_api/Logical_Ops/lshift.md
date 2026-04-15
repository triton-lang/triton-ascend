# triton.language.core.__lshift__
## 1. OP 概述

简介：根据给定值 将tensor张量进行左移位。

```
triton.language.core.__lshift__(
	input: tl.tensor,
	other: tl.tensor,
	builder: ir.builder
) -> tl.tensor
```

作为`tensor`的内置运算符使用，如`x<<y`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | --- |
| `input` | `tensor` | 张量数据，左操作数，代表要进行移位的主数据 |
| `other`   | `tensor or scalar` | 张量数据，右操作数，进行移位的数值 |
| `_builder` | - | 保留参数，暂不支持外部调用 |

返回值：
`tl.tensor`：同`input`的shape的张量

### 2.2 支持规格

#### 2.2.1 DataType 支持

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU      | √ | √ | √ | √ | √ | √ | √ | √ | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | × | × | × | × | √ |

结论：Ascend 对比 GPU 缺失uint的支持能力。

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| -------- | ---------------------- |
| GPU    | 无限制 |
| Ascend A2/A3 | 无限制 |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint的支持能力，并且输入右参数为tensor时不支持。

### 2.4 使用方法

以下示例实现了对三维张量`x0`、`x1`做左移位运算：

```@triton.jit
def triton_lshift_3d(in_ptr0, out_ptr0, L : tl.constexpr, M : tl.constexpr, N : tl.constexpr):
    loffs = tl.program_id(0) * L
    lblk_idx = tl.arange(0,L) + loffs
    mblk_idx = tl.arange(0,M)
    nblk_idx = tl.arange(0,N)
    idx = lblk_idx[:,None,None]*N*M+mblk_idx[None,:,None]*N+nblk_idx[None,None,:]
    x0=tl.load(in_ptr0+idx)
    ret = x0 << 2
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0+odx, ret)
```
