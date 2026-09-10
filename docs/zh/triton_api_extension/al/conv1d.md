# al.conv1d 接口文档

## 1. 背景

al.conv1d 在输入信号上执行一维卷积，支持可选偏置（bias）与分组卷积（groups），padding 支持标量或元组形式，接口语义对齐 torch.nn.functional.conv1d。

## 2. 接口说明

<table>
  <tr>
    <td>Python<br>output = al.conv1d(<br>    input,<br>    weight,<br>    bias,<br>    groups=1,<br>    padding=0,<br>    stride=1,<br>    dilation=1,<br>) -&gt; tensor :</td>
  </tr>
</table>

### 参数

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>input</td>
    <td>tensor</td>
    <td>是</td>
    <td>输入张量，形状 [N, iC, iW] 或 [iC, iW]，N 为 batch size，iC 为输入通道数，iW 为输入宽度</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>tensor</td>
    <td>是</td>
    <td>权重张量，形状 [oC, iC / groups, wW]，oC 为输出通道数，wW 为卷积核宽度，要求 oC % groups == 0</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>tensor</td>
    <td>否</td>
    <td>偏置张量，形状 [oC]，默认 None</td>
  </tr>
  <tr>
    <td>groups</td>
    <td>int</td>
    <td>否</td>
    <td>输入到输出通道的分组数，默认 1</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>int / tuple</td>
    <td>否</td>
    <td>输入两侧的填充，支持 int（两侧对称）或 2 元组 (paddingLeft, paddingRight)（非对称），默认 0</td>
  </tr>
  <tr>
    <td>stride</td>
    <td>int</td>
    <td>否</td>
    <td>卷积核的步长，默认 1</td>
  </tr>
  <tr>
    <td>dilation</td>
    <td>int</td>
    <td>否</td>
    <td>卷积核元素之间的间距，暂未支持非 1，默认 1</td>
  </tr>
</table>

### 返回值

输出张量，形状 [N, oC, oW] 或 [oC, oW]。

### 2.3 支持规格

#### 2.3.1 DataType 支持

| 输入类型 | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |
| Ascend A5 | ×    | ×     | ×     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

结论：al.conv1d 支持 fp16、bf16、fp32 三种浮点数据类型。

### 2.4 约束说明

- groups 必须同时整除 iC 与 oC（oC % groups == 0）。

- bias 为可选参数，形状必须为 [oC]。

- dilation 暂未支持非 1 取值。

- padding 支持 int 或 2 元组 (paddingLeft, paddingRight)。

- 默认值：groups=1、padding=0、stride=1、dilation=1。

## 3. 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def conv1d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    L_in: tl.constexpr,
    C_out: tl.constexpr,
    L_out: tl.constexpr,
    K: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    groups: tl.constexpr,
):
    # Load input: (N, C_in, L_in)
    n_offs = tl.arange(0, N)[:, None, None]
    c_offs = tl.arange(0, C_in)[None, :, None]
    l_offs = tl.arange(0, L_in)[None, None, :]
    input_tile = tl.load(input_ptr + n_offs * (C_in * L_in) + c_offs * L_in + l_offs)

    # Load weight: (C_out, C_in // groups, K)
    co_offs = tl.arange(0, C_out)[:, None, None]
    ci_offs = tl.arange(0, C_in // groups)[None, :, None]
    k_offs = tl.arange(0, K)[None, None, :]
    weight_tile = tl.load(weight_ptr + co_offs * ((C_in // groups) * K) + ci_offs * K + k_offs)

    # Load bias: (C_out,)
    bias_tile = tl.load(bias_ptr + tl.arange(0, C_out))

    output = al.conv1d(
        input_tile,
        weight_tile,
        bias_tile,
        groups=groups,
        padding=padding,
        stride=stride,
        dilation=1,
    )

    # Store output: (N, C_out, L_out)
    no_offs = tl.arange(0, N)[:, None, None]
    co_offs = tl.arange(0, C_out)[None, :, None]
    lo_offs = tl.arange(0, L_out)[None, None, :]
    tl.store(output_ptr + no_offs * (C_out * L_out) + co_offs * L_out + lo_offs, output)
```
