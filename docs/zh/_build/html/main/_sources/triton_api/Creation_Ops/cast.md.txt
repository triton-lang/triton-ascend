# triton.language.cast

## 1 功能作用说明

将张量转换为指定的数据类型，支持数值类型转换、位级别重解释（bitcast）、浮点降精度舍入模式，以及Ascend扩展的整数溢出处理模式。

**语法：**

- `triton.language.cast(input, dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None)` - 函数调用形式
- `input.cast(dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None)` - 成员函数形式

**功能：**

- 数值类型转换：整型<->整型、浮点<->浮点、整型<->浮点
- 位级别重解释（bitcast）：不改变比特，只改变解释类型
- 浮点降精度支持舍入模式：`rtne`（默认，四舍六入五成双）、`rtz`（向零）
- 整数转换（Ascend 扩展）支持溢出模式：`trunc`（截断，默认）、`saturate`（饱和）

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| dtype | tl.dtype | 是 | 目标数据类型 |
| fp_downcast_rounding | str | 否 | 仅对浮点降精度有效，`rtne` 或 `rtz` |
| bitcast | bool | 否 | 是否执行位级别重解释，默认 False |
| overflow_mode | str | 否 | Ascend 扩展：整数溢出处理，`trunc` 或 `saturate` |

**返回值：**

- **类型：** tensor
- **形状：** 与输入张量相同
- **数据类型：** 与dtype参数指定的目标类型相同
- **内存布局：** 根据bitcast参数决定是否进行位级别重解释

**约束条件：**

- `fp_downcast_rounding` 仅在浮点降精度时可设置，否则将报错
- `bitcast=True` 时不进行数值转换，忽略舍入/溢出模式
- `overflow_mode` 仅对整型有意义（Ascend 扩展）

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

```python
import triton
import triton.language as tl

@triton.jit
def cast_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 转换为int32
    y = tl.cast(x, tl.int32)

    return y

## 调用示例
result = cast_example()
print(result.dtype)  # 输出: int32
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

    # 整数转换，饱和模式（Ascend扩展）
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
    x_quantized = tl.cast(x * scale + zero_point, tl.int8, overflow_mode="saturate")

    # 存储量化结果
    tl.store(output_ptr + offsets, x_quantized, mask=mask)
```
