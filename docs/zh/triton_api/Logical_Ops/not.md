# triton.language.tensor.__not__

## 1. 函数概述

简介：对 tensor 做逐元素逻辑非（0 变 1，非零变 0）。对应 Python 的 `not` 关键字 —— Triton 通过 AST visitor 特殊处理，把 `not X` 重写为 `X.__not__()`。与按位取反 `~X`（见 [invert](./invert.md)）不同：前者是逻辑非，后者是按位翻转。

```python
# 通过 not 关键字（Triton AST 拦截处理）
not x

# 或直接调用 dunder 方法
x.__not__()
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用

返回值：
`out`：输出张量的shape与输入x的shape相同

### 2.2 OP 规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | × | √ | × | √ | × | √ | × | × | × | × | × | × | √ |
| Ascend 950 | × | × | × | × | × | × | × | × | × | × | × | × | × | × | √ |

结论：Ascend 相比 GPU 额外支持非 bool 类型。


结论：
- Ascend A2/A3/950 对比 GPU 无缺失的数据类型。


#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~8维 tensor |
| Ascend A2/A3 | 仅支持 1~8维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 8 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

暂无。

### 2.4 使用方法

以下示例对输入张量 `x` 做逐元素按位取反：

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

    ret = not(X)

    tl.store(output_ptr + idx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
```
