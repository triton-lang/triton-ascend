# triton.language.dot

## 1. OP 概述

简介：对两个tensor进行矩阵乘操作。tensor需要是二维或三维并且维度需一致。对于三维块，tl.dot执行批量矩阵乘法，其中每个块的第一维代表批量维度。
原型：

```python
triton.language.dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=triton.language.float32, _semantic=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `input`        | `int8 fp16 bf16 fp32`     |     第一个输入，2D or 3D 张量， 为了避免溢出 取值范围限制为-5-5     |                                                       |
| `other`       | `int8 fp16 bf16 fp32`     |     第二个输入,  2D or 3D 张量，为了避免溢出 取值范围限制为-5-5    |                                                   |
| `acc`           | `int32  float32`    | 存累加结果的张量, accumulator tensor. If not None, the result is added to this tensor, acc_dtype支持 {:code:`float16`, :code:`float32`, :code:`int32`} |
| `input_precision`   | -                 |  Available options for nvidia 通过选择精度模式来决定是否启用 Tensor Cores 加速    |
| `max_num_imprecise_acc`     | `int`    | 多少次低精度的累加数（当前昇腾不支持低精度累加） |
| `out_dtype`     | `fp32  int32`    | 输出结果类型|

返回值：
`tl.tensor`：矩阵乘结果

### 2.2 支持规格

#### 2.2.1 DataType 支持

|   输入类型     | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ✓    |

结论：Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 无限制 |
| Ascend A2/A3 |无限制  |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

- Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

- acc 不能支持FP16，为了精度硬件默认就是FP32

- max_num_imprecise_acc 暂时不支持

- out_dtype对比GPU 缺乏int8和FP16的类型支持

### 2.4 使用方法

以下示例实现了对输入张量 `x_ptr, y_ptr` 做矩阵乘计算，参考  ascend/examples/generalization_cases/test_matmul.py：

```@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        acc_dtype: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```
