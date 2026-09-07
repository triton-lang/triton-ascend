# triton.language.sub

## 1. OP 概述

简介：减法  ，四则运算 ‘-’，无tl.sub方法

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `input` | tensor或Number | 被减数 |
| `other` | tensor或Number | 减数 |

返回值：
`tl.tensor`：减法结果

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | × | √ |
| Ascend A2/A3 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ | √ | √ |


结论：
- Ascend A2/A3 对比 GPU 缺少uint16, uint32, uint64, fp64数据类型支持。
- Ascend 950 对比 GPU 缺少 fp64 数据类型支持。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 无限制 |
| Ascend |无限制  |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 使用方法

以下示例实现了对输入张量 `in_ptr0, in_ptr1` 做减法计算：

```python
@triton.jit
def triton_sub(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 - tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

## 3. 特殊说明

> Ascend 950 对比 GPU 缺失 fp64 的支持能力。
