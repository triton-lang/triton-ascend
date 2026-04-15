# floordiv
## 1. OP 概述

简介：取整除法，返回向零取整后的除法结果，四则运算 ‘//’，无tl.floordiv方法

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `tensor or Number`     |     第一个入参 ，被除数    |                                                       |
| `other`       | `tensor or Number`     |     第二个入参 ，除数    |                                                   |



### 2.2 支持规格

#### 2.2.1 DataType 支持


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √| √    | ×    | ×    | ×    | √    |
|Ascend A2/A3| × | √ | × | √ | × | √ | × | √  | ×    | ×    | ×    | √    |



#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 无限制 |
| Ascend A2/A3 |无限制  |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。



### 2.3 使用方法

以下示例实现了对输入张量 `in_ptr0, in_ptr1` 做整除计算：

```
@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx)
    y = tl.load(in_ptr1 + idx)
    ret = x // y
    tl.store(out_ptr0 + idx, ret)
```



### 2.4. 特殊限制

Ascend A3 相比 GPU 缺失uint8、uint16、uint32、uint64的支持
