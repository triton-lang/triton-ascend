# triton.language.arange

## 1. OP 概述

简介：`triton.language.arange`函数用于生成一个从`start`到`end`（不包括`end`）的连续整数序列。

```python
triton.language.arange(start,end, _semantic=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                  | 说明                                   |
| ------------- | ----------------- | ---------------------------- |
| `start`           | `scalar`               | 创建连续整数序列的起始数值，必须是编译时常量（tl.constexpr） |
| `end`            | `scalar`               | 创建连续整数序列的结束数值 |

返回值：
`tensor`：连续整数序列的tensor

### 2.2 支持规格

#### 2.2.1 DataType 支持

结论：要求arange的参数start、end必须是constant，因此无类型，支持类型对应的值范围，最大到int32，硬件指令也只支持到int32。

|                | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| --------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| GPU            | ×    | ×     | ×     | ×     | ×      | √      | ×      | ×     | ×    | ×    | ×    | ×    | ×    |
| Ascend  A2/A3 | ×    | ×     | √     | ×     | ×      | ×      | ×      | ×     | ×    | ×    | ×    | ×    | ×    |

#### 2.2.2 Shape 支持

0 =< (end - start) <1048576
end >= 0,  start  >= 0

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

1.函数用于生成一个[start, end) 的连续整数序列，CUDA要求range=(end-start)必须为2的幂次方。Triton-ascend并无此要求。
2.NV和Triton-ascend都限制end的最大值TRITON_MAX_TENSOR_NUMEL = 1048576
3.arange的输入必须是constant常量，即已支持uint、int类型的小于1048576（最大值TRITON_MAX_TENSOR_NUMEL ）的数值。int64不支持。
4.arange的start 和 end 应大于等于0。

### 2.4 使用方法

以下示例实现了生成一个[0, 128) 的连续整数序列：

```python
@triton.jit
def triton_arange(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)
@pytest.mark.parametrize('param_list',[[0, 128],])
def test_case_access(param_list):
    start, end = param_list
    shape = [end]
    block = end - start
    dtype = 'int32'
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()
    triton_arange_access[(1, )](y_cal, START = start, END = end, BLOCK = block)
```
