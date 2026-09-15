# triton.language.flip

## 1. 函数概述

简介：将tensor沿某一维度进行翻转。

```python
triton.language.flip(x, dim=None)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `dim`        | `int`          | 整型                                                      |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用

返回值：
`out`：输出张量的shape与输入x的shape相同

### 2.2 OP 规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | × | √ |
| Ascend A2/A3 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |

结论：
- Ascend A2/A3 对比 GPU 缺失 uint16、uint32、uint64、fp64 的支持能力。
- Ascend 950 对比 GPU 缺失 fp64 的支持能力。

#### 2.2.2 Shape 支持

|            | 支持维度范围        |
| ---------- | ------------------- |
| GPU        | 仅支持 1~8维 tensor |
| Ascend A2/A3 | 仅支持 1~8维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 8 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 相比 GPU Ascend 950 对比 GPU 缺失 fp64 的支持能力。。

### 2.4 使用方法

以下示例将输入张量 `x` 沿指定维度进行翻转：

```python
@triton.jit
def fn_npu_3d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
            YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    zidx = tl.arange(0, ZB) + tl.program_id(2) * ZB
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = tl.flip(X, 2)
    oidx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
```
