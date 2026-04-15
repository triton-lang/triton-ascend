# triton.language.xor_sum
## 1. OP 概述

简介：`triton.language.xor_sum` 计算输入tensor沿指定轴的异或和，返回异或操作结果。

```
triton.language.xor_sum(input, axis=None, keep_dims=False)
```

## 2. OP 规格

### 2.1 参数说明

```
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `input` | `Tensor` | 输入tensor |
| `axis` | `int` 或 `None` | 沿着哪个维度进行异或和操作。如果为None，则对所有维度进行异或操作 |
| `keep_dims` | `bool` | 如果为True，保持被操作的维度为长度1 |

```

返回值：
`tensor`：输入tensor沿指定轴的异或和，返回异或操作结果

### 2.2 支持规格

#### 2.2.1 DataType 支持


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |



#### 2.2.2 Shape 支持

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现
> keep_dims=True需要测试更多规格，来确定是否全面支持。目前已测3D dim=2情况下，支持 keep_dims=True。

### 2.4 使用方法

以下示例实现了对2Dshape的tensor进行xor_sum运算：

```python
@triton.jit
def triton_xorsum_2d(in_ptr0, out_ptr0, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
                     NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=-float('inf'))
    tmp4 = tl.xor_sum(x, dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0, N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0, M), tmp4, None)

```
