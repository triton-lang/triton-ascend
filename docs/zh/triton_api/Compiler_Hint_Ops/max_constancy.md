# triton.language.max_constancy
## 1. 函数概述

`max_constancy` 用于向编译器声明输入张量中值的常量性模式，告知编译器输入数据中每组连续的值都是相等的。

```python
triton.language.max_constancy(input, values, _builder=None, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `input` | `Tensor` | 必需 | 输入张量，其值具有特定的常量性模式 |
| `values` | `constexpr[int]` | 必需 | 描述常量性模式的编译时常量整数（或整数序列） |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

**`values`描述着每个维度的恒定性特征，所以`values` 的维度要与`input` 的维度相同。
注意当`shape`的最后一维为`1`时出现的降维情况。**

如：二维 `input`对应通用`values`入参为`[1,1]`。

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |



### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

### 2.4 使用方法

```python
import triton.language as tl

@triton.jit
def basic_constancy_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # 使用constexpr声明常量性模式：每4个连续的值都是相等的
    # 例如输入模式：[0,0,0,0,1,1,1,1,2,2,2,2,...]
    input_data = tl.max_constancy(input_data, 4)

    # 编译器可以基于这个信息进行优化
    result = input_data * 2
    tl.store(B + offsets, result)
```
