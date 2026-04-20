# triton.language.min

## 1. OP 概述

简介：在指定维度上返回最小值

```python
triton.language.min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | --- |
| `input` | `tensor` | 输入的张量数据 |
| `axis`   | `int` | 指定在哪个维度上进行规约 |
| `keep_dims` | `bool` | 保持规约后的维度 |
| `return_indices` | `bool` | True返回最小值的同时返回最小值相关所在下标 |
| `return_indices_tie_break_left` | `bool` | True如果多个元素有相同的最小值，返回最左侧最小值的下标。|

返回值：
`tl.tensor`：同`input`的shape的张量

参数组合支持：

| axis | keep_dims | return_indices | return_indices_tie_break_left  |  规格 |
| ------ | ------------ | ----------------- | ----------------------------------- | ---|
|    1 |    TRUE    |      TRUE      |               TRUE                |  支持
|    1 |    TRUE    |      TRUE      |               FALSE                | 支持
|    1 |    TRUE    |      FALSE      |               TRUE              | 支持
|    1 |    TRUE    |      FALSE      |               FALSE             | 支持
|    1 |   FALSE   |      TRUE      |               TRUE               | 支持
|    1 |   FALSE   |      TRUE      |               FALSE                | 支持
|    1 |   FALSE   |      FALSE      |               TRUE                | 支持
|    1 |   FALSE   |      FALSE      |               FALSE                | 支持
| None |    TRUE    |      TRUE      |               TRUE                | 不支持
| None |    TRUE    |      TRUE      |               FALSE                | 不支持
|  None |    TRUE    |      FALSE      |               TRUE               | 支持
| None |    TRUE    |      FALSE      |               FALSE                | 支持
| None |   FALSE   |      TRUE      |               TRUE               | 不支持
| None |   FALSE   |      TRUE      |               FALSE              | 不支持
| None |   FALSE   |      FALSE      |               TRUE               | 支持
| None |   FALSE   |      FALSE      |               FALSE               | 支持

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| √ | √ | x | √ | × | √ | × | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| -------- | ---------------------- |
| GPU    | 无限制 |
| Ascend A2/A3 | 无限制 |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 使用方法

更多示例参考triton-ascend代码仓，ascend/examples/generalization_cases/test_min.py

```python
@triton.jit
def triton_min_1d(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.min(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```

### 2.4. 特殊限制

Ascend A3对比 GPU 缺失uint16、uint32、uint64、fp64的支持。
