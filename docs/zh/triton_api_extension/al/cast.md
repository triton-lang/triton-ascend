# al.cast

## 1. 函数概述

将张量转换为指定的数据类型，支持数值类型转换、位级别重解释（bitcast）、浮点降精度舍入模式，以及 Ascend 扩展的整数溢出处理模式。

`al.cast(input, dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None, _semantic=None)`

可以作为 tensor 的成员函数调用，如 `x.cast(...)`，与 `cast(x, ...)` 等效。

> 注意：`triton.language` 本身也有一个同名的 `tl.cast`（`python/triton/language/core.py`），签名里**没有 `overflow_mode`** 参数。`al.cast` 是 Ascend 扩展版本，在 `tl.cast` 的基础上增加了 `overflow_mode`（整数溢出处理），两者不是同一个函数，写代码时不要混用/混淆。

## 2. 参数规格

### 2.1 参数说明

| 参数名                  | 类型     | 必需   | 说明                                             |
| -------------------- | -------- | ------ | ------------------------------------------------ |
| input                | tensor   | 是     | 输入张量                                         |
| dtype                | tl.dtype | 是     | 目标数据类型                                     |
| fp_downcast_rounding | str      | 否     | 仅对浮点降精度有效，`rtne` 或 `rtz`              |
| bitcast              | bool     | 否     | 是否执行位级别重解释，默认 False                 |
| overflow_mode        | str      | 否     | Ascend 扩展：整数溢出处理，`trunc` 或 `saturate` |
| `_semantic`          | -        | `None` | 保留参数，暂不支持外部调用                       |

**返回值：**

- **类型：** tensor
- **形状：** 与输入张量相同
- **数据类型：** 与 dtype 参数指定的目标类型相同
- **内存布局：** 根据 bitcast 参数决定是否进行位级别重解释

**约束条件：**

- `fp_downcast_rounding` 仅在浮点降精度（源类型是更宽的浮点类型、目标类型是更窄的浮点类型）时可设置，否则 `raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions...")`。
- `bitcast=True` 时不进行数值转换，忽略舍入/溢出模式（直接调用 `_semantic.bitcast`，函数提前返回）。
- `overflow_mode` 未设置或为 `"trunc"` 时，用截断方式处理溢出；不是这两个合法值之一（即不是 `"trunc"`/`"saturate"`）会 `raise ValueError(f"Unknown overflow_mode:{overflow_mode} is found.")`。

### 2.2 特殊限制说明

- **fp8 / fp64 只在 910_95（对应 Ascend 950 系列）架构上支持转换**：源码里有一条硬性检查——
  ```python
  if not is_compile_on_910_95(_semantic.builder.options.arch):
      if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
          raise ValueError("[fp8, fp64] is unsupported on Ascend for now. ...")
  ```

## 3. 使用方法

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def cast_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)
    # 转换为int32
    y = al.cast(x, tl.int32)
    return y
```

**高级用法：**

```python
@triton.jit
def cast_advanced_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)
    # 位级别重解释
    y = x.cast(tl.int32, bitcast=True)
    # 浮点降精度，向零舍入
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")
    # float32 → int8，传入 overflow_mode="saturate"（具体饱和效果见 2.2 节说明，建议实测确认）
    w = x.cast(tl.int8, overflow_mode="saturate")
    return y, z, w
```

**实际应用场景：**

```python
@triton.jit
def quantization_kernel(x_ptr, output_ptr, scale, zero_point, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 加载float32数据
    x = tl.load(x_ptr + offsets, mask=mask)
    # 量化：转换为int8
    x_quantized = al.cast(x * scale + zero_point, tl.int8, overflow_mode="saturate")
    # 存储量化结果
    tl.store(output_ptr + offsets, x_quantized, mask=mask)
```
