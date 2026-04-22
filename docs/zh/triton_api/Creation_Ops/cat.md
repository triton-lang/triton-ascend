# triton.language.cat

## 1. OP 概述

简介：`triton.language.cat`函数用于将指定的tensor进行拼接。

```python
triton.language.cat(input, other, can_reorder=False, _semantic=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                  | 说明                                   |
| ------------- | ----------------- | ---------------------------- |
| `input`           | `Tensor`               | 拼接的第一个tensor |
| `other`            | `Tensor`               | 拼接的第二个tensor |
| `can_reorder`            | `Bool`               | 重新排序 – 编译器提示。如果为真，编译器在连接输入时允许重新排序元素。仅支持can_reorder=True。  |
| `_semantic`            | `Optional[str]`               | 保留参数，暂不支持外部调用 |

返回值：
`tensor`：完成拼接之后的tensor

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape 支持

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。cat 只支持1D shape 的拼接。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

1.ASCEND和CUDA都只支持 can_reorder=True，即拼接tensor后重新排序。
2.cat 只支持1D shape 的拼接。

### 2.4 使用方法

以下示例实现了对1D shape的两个tensor进行的拼接：

```python
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
import math

@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):

    idx = tl.arange(0, XB)
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cat(X, Y, can_reorder=True)

    oidx = tl.arange(0, XB * 2)

    tl.store(output_ptr + oidx, ret)


## The CAT operator in the Triton community also does not support boolean types.
@pytest.mark.parametrize('shape', [(32,), (741,)]) #triton only support 1D cat
@pytest.mark.parametrize('dtype', ['float32',])
def test_cat(shape, dtype):
    m = shape[0]
    x = torch.full((m, ), 100, dtype=eval("torch." + dtype)).npu()
    y = torch.full((m, ), 30, dtype=eval("torch." + dtype)).npu()

    output = torch.randint(1, (m * 2, ), dtype=eval("torch." + dtype)).npu()

    ans = torch.cat((x, y), dim=0)

    fn_npu_[1, 1, 1](output, x, y, m)

    test_common.validate_cmp(dtype, ans, output)
```
