# triton.language.add

## 1. OP 概述

简介：加法  ，与四则运算 ‘+’等价
原型：

```python
triton.language.add(x, y, sanitize_overflow: constexpr = True, _builder=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor or Number`     |     第一个入参     |                                                       |
| `y`       | `tensor or Number`     |     第二个入参     |                                                   |
| `sanitize_overflow`     | `bool`    | 是否对整数加法做溢出检查，默认值为True,无需显示指定 |
| `_builder`   | -                 | 保留参数，暂不支持外部调用                                                |

返回值：
`tl.tensor`：加法结果

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 无限制 |
| Ascend |无限制  |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 使用方法

以下示例实现了对输入张量 `x_ptr, y_ptr` 做加法计算：

```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.             
               ):  
    pid = tl.program_id(axis=0)    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements 
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y # 等价于 output = tl.add(x,y)   
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 3. 特殊说明

> Ascend A3 对比 GPU 不支持fp64
