# triton.language.histogram

## 1. OP 概述

简介：基于 input 计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0。
原型：

```python
triton.language.histogram(
 input,
 num_bins,
 mask=None,
 _semantic=None,
 _generator=None
)
```

可以作为tensor的成员函数调用，如`x.histogram(...)`，与`histogram(x, ...)`等效。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `input`        | `tensor`          | 输入数据，包含需要统计分布的所有数值点                                                    |
| `num_bins`       | `int`    | 定义要将整个数据范围划分成多少个等宽的区间                       |
| `mask`     | `int1`或`tensor<int1>`，可选    | 指定数据范围，防止访问越界 |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                                |
| `_generator` |-                 | 保留参数，暂不支持外部调用

返回值：
用tensor表示的直方图
注：当前triton3.2版本暂未支持mask，待版本更新后支持；input输入范围限制在[0,num_bins-1]中，待版本更新后支持全范围

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×      | √      | ×      | ×     | ×    | ×    | ×    | ×    | ×       |
| Ascend A2/A3 | ×    | ×     | √    | ×     | ×      | √      | √      | √     | ×    | ×    | ×    | ×    | ×    |

#### 2.2.2 Shape 支持

目前仅支持一维

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

### 2.4 使用方法

以下示例实现了histogram的调用：

```python
@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)

x = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
z = torch.empty(N, dtype=torch.int32, device=device)
histogram_kernel[(1, )](x, z, M=M, N=N)
```

## 3. 语义GAP

> 相对社区能力缺失但能开发支持
