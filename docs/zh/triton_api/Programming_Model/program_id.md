# triton.language.program_id

## 1. OP 概述

简介：返回当前程序实例沿给定 axis 的 ID。
函数原型：

```python
triton.language.program_id(axis)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `axis` | `int` |  3D 启动网格的轴。必须是 0、1 或 2 |

返回值：
由轴的值组成的tl.tensor

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | × | × | × | × | × | √ | × | × | × | × | × | × | × | × | × |
| Ascend A2/A3 | × | × | × | × | × | √ | × | × | × | × | × | × | × | × | × |
| Ascend 950 | × | × | × | × | × | √ | × | × | × | × | × | × | × | × | × |


结论：
- Ascend A2/A3/950 对比 GPU 无缺失的数据类型。


#### 2.2.2 Shape 支持

无相关设置

### 2.3 特殊限制说明

无

### 2.4 使用方法

在triton kernel中会用到，用于获取PID

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ...

    tl.store(output_ptr + idx, ret)
```
