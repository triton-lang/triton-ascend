# triton.language.cast

## 1 功能作用说明

将张量转换为指定的数据类型，支持数值类型转换、位级别重解释（bitcast），以及浮点降精度舍入模式。

**语法：**

- `triton.language.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)` - 函数调用形式
- `input.cast(dtype, fp_downcast_rounding=None, bitcast=False)` - 成员函数形式

**功能：**

- 数值类型转换：整型<->整型、浮点<->浮点、整型<->浮点
- 位级别重解释（bitcast）：不改变比特，只改变解释类型
- 浮点降精度支持舍入模式：`rtne`（默认，四舍六入五成双）、`rtz`（向零）

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| dtype | tl.dtype | 是 | 目标数据类型 |
| fp_downcast_rounding | str | 否 | 仅对浮点降精度有效，`rtne` 或 `rtz` |
| bitcast | bool | 否 | 是否执行位级别重解释，默认 False |

**返回值：**

- **类型：** tensor
- **形状：** 与输入张量相同
- **数据类型：** 与 dtype 参数指定的目标类型相同
- **内存布局：** 根据 bitcast 参数决定是否进行位级别重解释

**约束条件：**

- `fp_downcast_rounding` 仅在浮点降精度时可设置，否则将报错
- `bitcast=True` 时不进行数值转换，忽略舍入模式

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape支持表

支持任意维度数、任意形状大小。

### 2.4 特殊限制说明

无

### 2.5 使用方法

**基本用法：**

将 `float32` 输入转换为 `int32`，并通过 host 侧启动 kernel 查看输出类型：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cast_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # float32 -> int32
    y = tl.cast(x, tl.int32)
    tl.store(y_ptr + offs, y, mask=mask)

n = 8
x = torch.randn(n, device="npu", dtype=torch.float32)
y = torch.empty(n, device="npu", dtype=torch.int32)
cast_kernel[(1,)](x, y, n, BLOCK=8)

print(x.dtype)  # torch.float32
print(y.dtype)  # torch.int32
```

**高级用法：**

成员函数形式同样支持 `bitcast` 与浮点降精度舍入：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cast_advanced_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)

    # Bitcast reinterpret: float32 -> int32
    y = x.cast(tl.int32, bitcast=True)
    # FP downcast with round-toward-zero: float32 -> float16
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(z_ptr + offs, z, mask=mask)

n = 8
x = torch.randn(n, device="npu", dtype=torch.float32)
y = torch.empty(n, device="npu", dtype=torch.int32)
z = torch.empty(n, device="npu", dtype=torch.float16)
cast_advanced_kernel[(1,)](x, y, z, n, BLOCK=8)

print(y.dtype)  # torch.int32
print(z.dtype)  # torch.float16
```

**实际应用场景：**

量化场景中将缩放后的浮点值转为 `int8`：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def quantization_kernel(x_ptr, output_ptr, scale, zero_point, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offsets, mask=mask)
    x_quantized = tl.cast(x * scale + zero_point, tl.int8)
    tl.store(output_ptr + offsets, x_quantized, mask=mask)

M, N = 16, 16
BLOCK_M, BLOCK_N = 8, 8
x = torch.randn((M, N), device="npu", dtype=torch.float32)
out = torch.empty((M, N), device="npu", dtype=torch.int8)
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
quantization_kernel[grid](x, out, 1.0, 0.0, M, N, BLOCK_M, BLOCK_N)

print(out.dtype)  # torch.int8
```
