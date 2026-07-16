# Libdevice 开发者手册

## SIMT 编译示例

使用 SIMT 编译的 triton kernel 示例

```python
# Enable libdevice SIMT compilation
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch

@triton.jit
def triton_kernel(input, output, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        x = tl.load(input + (x0), None)
        y = libdevice.abs(x)
        tl.store(output + (x0), y, None)

dtype, shape, ncore, xblock, xblock_sub = ['int32', (128, 4096), 512, 1024, 1024]
input = torch.randn(shape, dtype=eval('torch.' + dtype)).npu()
output = torch.zeros_like(input)
# Enable SIMT compilation with option "force_simt_only=True"
triton_kernel[ncore, 1, 1](input, output, xblock, xblock_sub, force_simt_only=True)
```

## 1. triton.language.extra.cann.libdevice.abs

### OP概述

计算输入参数的绝对值。

原型:

```python
triton.language.extra.cann.libdevice.abs(x, _builder=None)
```

输入类型：

- x: `int32`, `float32`

返回值: `tl.tensor`, 返回输入参数的绝对值。

返回类型：`int32`, `float32`

支持编译模式：SIMT

## 2. triton.language.extra.cann.libdevice.acos

### OP概述

计算输入参数的反余弦值。

原型:

```python
triton.language.extra.cann.libdevice.acos(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的反余弦值，取值范围 \[0, π] 弧度。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 3. triton.language.extra.cann.libdevice.acosh

### OP概述

计算输入参数的反双曲余弦值。

原型:

```python
triton.language.extra.cann.libdevice.acosh(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的反双曲余弦值，取值范围 \[0, +∞]。

返回类型：`float32`

支持编译模式：SIMT

## 4. triton.language.extra.cann.libdevice.add_rd

### OP概述

向下舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_rd(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回向下舍入的加法结果。

返回类型：`float32`

支持编译模式：SIMT

## 5. triton.language.extra.cann.libdevice.add_rn

### OP概述

最近偶数舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_rn(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回最近偶数舍入的加法结果。

返回类型：`float32`

支持编译模式：SIMT

## 6. triton.language.extra.cann.libdevice.add_ru

### OP概述

向上舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_ru(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回向上舍入的加法结果。

返回类型：`float32`

支持编译模式：SIMT

## 7. triton.language.extra.cann.libdevice.add_rz

### OP概述

向零舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_rz(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回向零舍入的加法结果。

返回类型：`float32`

支持编译模式：SIMT

## 8. triton.language.extra.cann.libdevice.asin

### OP概述

计算输入参数的反正弦值。

原型:

```python
triton.language.extra.cann.libdevice.asin(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的反正弦值，取值范围 \[-π/2, π/2] 弧度。

返回类型：`float32`

支持编译模式：SIMT

## 9. triton.language.extra.cann.libdevice.asinh

### OP概述

计算输入参数的反双曲正弦值。

原型:

```python
triton.language.extra.cann.libdevice.asinh(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的反双曲正弦值。

返回类型：`float32`

支持编译模式：SIMT

## 10. triton.language.extra.cann.libdevice.atan

### OP概述

计算输入参数的反正切值。

原型:

```python
triton.language.extra.cann.libdevice.atan(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的反正切值，取值范围 \[-π/2, π/2] 弧度。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 11. triton.language.extra.cann.libdevice.atan2

### OP概述

反正切函数，计算 x / y 的反正切值。

原型:

```python
triton.language.extra.cann.libdevice.atan2(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回 x / y 的反正切值，取值范围 \[-π, π] 弧度。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 12. triton.language.extra.cann.libdevice.atanh

### OP概述

反双曲正切函数，计算输入参数的反双曲正切值。

原型:

```python
triton.language.extra.cann.libdevice.atanh(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的反双曲正切值，取值范围 \[-1, 1]。

返回类型：`float32`

支持编译模式：SIMT

## 13. triton.language.extra.cann.libdevice.brev

### OP概述

位反转函数，反转32位整数的位顺序。

原型:

```python
triton.language.extra.cann.libdevice.brev(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回位反转后的32位整数。

返回类型：`int32`

支持编译模式：SIMT

## 14. triton.language.extra.cann.libdevice.byte_perm

### OP概述

字节排列操作，从两个32位整数中选择字节组成新整数。输入整数 x 和 y 的字节顺序如下。

```cpp
input[0] = x<7:0>     input[1] = x<15:8>
input[2] = x<23:16>   input[3] = x<31:24>
input[4] = y<7:0>     input[5] = y<15:8>
input[6] = y<23:16>   input[7] = y<31:24>
```

字节选择参数 s 为32位整数，各比特位与字节选择对应关系如下。

```cpp
selector[0] = s<2:0>    selector[1] = s<6:4>
selector[2] = s<10:8>   selector[3] = s<14:12>
```

原型:

```python
triton.language.extra.cann.libdevice.byte_perm(x, y, s, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`
- s: `int32`

返回值: `tl.tensor`, 返回值 return\[n] := input\[selector\[n]]，n 表示输出整数的第 n 个字节。

返回类型：`int32`

支持编译模式：SIMT

## 15. triton.language.extra.cann.libdevice.cbrt

### OP概述

计算输入参数的立方根值。

原型:

```python
triton.language.extra.cann.libdevice.cbrt(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的立方根值。

返回类型：`float32`

支持编译模式：SIMT

## 16. triton.language.extra.cann.libdevice.ceil

### OP概述

向上取整，返回大于或等于 x 的最小整数。

原型:

```python
triton.language.extra.cann.libdevice.ceil(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回向上取整的结果。

返回类型：`float32`

支持编译模式：SIMT

## 17. triton.language.extra.cann.libdevice.clz

### OP概述

计算32位整数的前导零数量。

原型:

```python
triton.language.extra.cann.libdevice.clz(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回输入参数的前导零数量。范围 \[0, 32]。

返回类型：`int32`

支持编译模式：SIMT

## 18. triton.language.extra.cann.libdevice.copysign

### OP概述

生成一个浮点数，其绝对值等于 x 的绝对值，符号与 y 相同。

原型:

```python
triton.language.extra.cann.libdevice.copysign(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回一个浮点数，其绝对值等于 x 的绝对值，符号与 y 相同。

返回类型：`float32`

支持编译模式：SIMT

## 19. triton.language.extra.cann.libdevice.cos

### OP概述

计算输入参数（弧度）的余弦值。

原型:

```python
triton.language.extra.cann.libdevice.cos(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的余弦值。

返回类型：`float32`

支持编译模式：SIMT

## 20. triton.language.extra.cann.libdevice.cosh

### OP概述

计算输入参数的双曲余弦值。

原型:

```python
triton.language.extra.cann.libdevice.cosh(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的双曲余弦值。

返回类型：`float32`

支持编译模式：SIMT

## 21. triton.language.extra.cann.libdevice.cospi

### OP概述

计算 cos(π × x) 的值。

原型:

```python
triton.language.extra.cann.libdevice.cospi(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 cos(π × x) 的值。

返回类型：`float32`

支持编译模式：SIMT

## 22. triton.language.extra.cann.libdevice.cyl_bessel_i0

### OP概述

计算输入参数的修正零阶贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i0(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的修正零阶贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 23. triton.language.extra.cann.libdevice.cyl_bessel_i1

### OP概述

计算输入参数的修正一阶贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i1(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的修正一阶贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 24. triton.language.extra.cann.libdevice.div_rd

### OP概述

向下舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_rd(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回除法结果。

返回类型：`float32`

支持编译模式：SIMT

## 25. triton.language.extra.cann.libdevice.div_rn

### OP概述

最近偶数舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_rn(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回除法结果。

返回类型：`float32`

支持编译模式：SIMT

## 26. triton.language.extra.cann.libdevice.div_ru

### OP概述

向上舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_ru(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回除法结果。

返回类型：`float32`

支持编译模式：SIMT

## 27. triton.language.extra.cann.libdevice.div_rz

### OP概述

向零舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_rz(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回除法结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 28. triton.language.extra.cann.libdevice.erf

### OP概述

计算输入参数的误差函数值。

原型:

```python
triton.language.extra.cann.libdevice.erf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的误差函数值。

返回类型：`float32`

支持编译模式：SIMT

## 29. triton.language.extra.cann.libdevice.erfc

### OP概述

计算输入参数的互补误差函数值，即 1 - erf(x)。

原型:

```python
triton.language.extra.cann.libdevice.erfc(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的互补误差函数值。

返回类型：`float32`

支持编译模式：SIMT

## 30. triton.language.extra.cann.libdevice.erfcinv

### OP概述

逆互补误差函数，找到满足 x = erfc(y) 的值 y。

原型:

```python
triton.language.extra.cann.libdevice.erfcinv(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的逆互补误差函数值。

返回类型：`float32`

支持编译模式：SIMT

## 31. triton.language.extra.cann.libdevice.erfcx

### OP概述

计算输入参数的缩放互补误差函数值，即 exp(x²) × erfc(x)。

原型:

```python
triton.language.extra.cann.libdevice.erfcx(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的缩放互补误差函数值。

返回类型：`float32`

支持编译模式：SIMT

## 32. triton.language.extra.cann.libdevice.erfinv

### OP概述

逆误差函数，找到满足 x = erf(y) 的值 y。

原型:

```python
triton.language.extra.cann.libdevice.erfinv(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的逆误差函数值。

返回类型：`float32`

支持编译模式：SIMT

## 33. triton.language.extra.cann.libdevice.exp

### OP概述

指数函数，计算 e 的 x 次方。

原型:

```python
triton.language.extra.cann.libdevice.exp(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 e 的 x 次方的计算结果。

返回类型：`float32`

支持编译模式：SIMT

## 34. triton.language.extra.cann.libdevice.exp10

### OP概述

以 10 为底的指数函数，计算 10 的 x 次方。

原型:

```python
triton.language.extra.cann.libdevice.exp10(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 10 的 x 次方的计算结果。

返回类型：`float32`

支持编译模式：SIMT

## 35. triton.language.extra.cann.libdevice.exp2

### OP概述

以 2 为底的指数函数，计算 2 的 x 次方。

原型:

```python
triton.language.extra.cann.libdevice.exp2(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 2 的 x 次方的计算结果。

返回类型：`float32`

支持编译模式：SIMT

## 36. triton.language.extra.cann.libdevice.expm1

### OP概述

计算 e 的 x 次方减 1 的结果。

原型:

```python
triton.language.extra.cann.libdevice.expm1(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 e 的 x 次方减 1 的计算结果。

返回类型：`float32`

支持编译模式：SIMT

## 37. triton.language.extra.cann.libdevice.fast_cosf

### OP概述

快速近似余弦函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_cosf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似余弦函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 38. triton.language.extra.cann.libdevice.fast_dividef

### OP概述

快速近似除法。

原型:

```python
triton.language.extra.cann.libdevice.fast_dividef(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回快速近似除法的结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 39. triton.language.extra.cann.libdevice.fast_exp10f

### OP概述

快速近似以 10 为底的指数函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_exp10f(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似以 10 为底的指数函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 40. triton.language.extra.cann.libdevice.fast_expf

### OP概述

快速近似指数函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_expf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似指数函数的结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 41. triton.language.extra.cann.libdevice.fast_log10f

### OP概述

快速近似以 10 为底的对数函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_log10f(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似以 10 为底的对数函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 42. triton.language.extra.cann.libdevice.fast_log2f

### OP概述

快速近似以 2 为底的对数函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_log2f(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似以 2 为底的对数函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 43. triton.language.extra.cann.libdevice.fast_logf

### OP概述

快速近似自然对数函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_logf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似自然对数函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 44. triton.language.extra.cann.libdevice.fast_powf

### OP概述

快速近似幂函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_powf(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回快速近似幂函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 45. triton.language.extra.cann.libdevice.fast_sinf

### OP概述

快速近似正弦函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_sinf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似正弦函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 46. triton.language.extra.cann.libdevice.fast_tanf

### OP概述

快速近似正切函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_tanf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回快速近似正切函数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 47. triton.language.extra.cann.libdevice.fdim

### OP概述

计算 x 与 y 的正差。当 x > y 时，返回 x - y，否则返回 0。

原型:

```python
triton.language.extra.cann.libdevice.fdim(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回 x 与 y 之间的正差。

返回类型：`float32`

支持编译模式：SIMT

## 48. triton.language.extra.cann.libdevice.fdiv

### OP概述

浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.fdiv(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回浮点数除法结果。

返回类型：`float32`

支持编译模式：SIMT

## 49. triton.language.extra.cann.libdevice.ffs

### OP概述

查找第一个被置为1的位，返回最低被置为1的位的索引。

原型:

```python
triton.language.extra.cann.libdevice.ffs(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回最低被置为1的位的索引，取值范围 \[0, 32]。

返回类型：`int32`

支持编译模式：SIMT

## 50. triton.language.extra.cann.libdevice.finitef

### OP概述

判断输入是否为有限浮点数。

原型:

```python
triton.language.extra.cann.libdevice.finitef(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 若输入为有限值返回 True，否则返回 False。

返回类型：`bool`

支持编译模式：SIMT

## 51. triton.language.extra.cann.libdevice.flip

### OP概述

沿指定维度反转张量元素顺序。

输入类型：

- ptr: `tensor`
- dim: `int32`

返回值: `tl.tensor`, 返回沿指定维度反转后的张量。

返回类型：`tensor`

支持编译模式：SIMT

## 52. triton.language.extra.cann.libdevice.float2int_rd

### OP概述

向下舍入模式将浮点数转换为32位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2int_rd(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位整数。

返回类型：`int32`

支持编译模式：SIMT

## 53. triton.language.extra.cann.libdevice.float2int_rn

### OP概述

最近偶数舍入模式将浮点数转换为32位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2int_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位整数。

返回类型：`int32`

支持编译模式：SIMT

## 54. triton.language.extra.cann.libdevice.float2int_ru

### OP概述

向上舍入模式将浮点数转换为32位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2int_ru(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位整数。

返回类型：`int32`

支持编译模式：SIMT

## 55. triton.language.extra.cann.libdevice.float2int_rz

### OP概述

向零舍入模式将浮点数转换为32位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2int_rz(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位整数。

返回类型：`int32`

支持编译模式：SIMT

## 56. triton.language.extra.cann.libdevice.float2ll_rd

### OP概述

向下舍入模式将浮点数转换为64位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ll_rd(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位整数。

返回类型：`int64`

支持编译模式：SIMT

## 57. triton.language.extra.cann.libdevice.float2ll_rn

### OP概述

最近偶数舍入模式将浮点数转换为64位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ll_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位整数。

返回类型：`int64`

支持编译模式：SIMT

## 58. triton.language.extra.cann.libdevice.float2ll_ru

### OP概述

向上舍入模式将浮点数转换为64位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ll_ru(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位整数。

返回类型：`int64`

支持编译模式：SIMT

## 59. triton.language.extra.cann.libdevice.float2ll_rz

### OP概述

向零舍入模式将浮点数转换为64位整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ll_rz(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位整数。

返回类型：`int64`

支持编译模式：SIMT

## 60. triton.language.extra.cann.libdevice.float2uint_rd

### OP概述

向下舍入模式将浮点数转换为32位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2uint_rd(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位无符号整数。

返回类型：`uint32`

支持编译模式：SIMT

## 61. triton.language.extra.cann.libdevice.float2uint_rn

### OP概述

最近偶数舍入模式将浮点数转换为32位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2uint_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位无符号整数。

返回类型：`uint32`

支持编译模式：SIMT

## 62. triton.language.extra.cann.libdevice.float2uint_ru

### OP概述

向上舍入模式将浮点数转换为32位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2uint_ru(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位无符号整数。

返回类型：`uint32`

支持编译模式：SIMT

## 63. triton.language.extra.cann.libdevice.float2uint_rz

### OP概述

向零舍入模式将浮点数转换为32位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2uint_rz(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的32位无符号整数。

返回类型：`uint32`

支持编译模式：SIMT

## 64. triton.language.extra.cann.libdevice.float2ull_rd

### OP概述

向下舍入模式将浮点数转换为64位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ull_rd(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位无符号整数。

返回类型：`uint64`

支持编译模式：SIMT

## 65. triton.language.extra.cann.libdevice.float2ull_rn

### OP概述

最近偶数舍入模式将浮点数转换为64位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ull_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位无符号整数。

返回类型：`uint64`

支持编译模式：SIMT

## 66. triton.language.extra.cann.libdevice.float2ull_ru

### OP概述

向上舍入模式将浮点数转换为64位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ull_ru(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位无符号整数。

返回类型：`uint64`

支持编译模式：SIMT

## 67. triton.language.extra.cann.libdevice.float2ull_rz

### OP概述

向零舍入模式将浮点数转换为64位无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.float2ull_rz(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回转换后的64位无符号整数。

返回类型：`uint64`

支持编译模式：SIMT

## 68. triton.language.extra.cann.libdevice.float_as_int

### OP概述

将浮点数的比特位重新解释为32位整数。不进行数值转换。

原型:

```python
triton.language.extra.cann.libdevice.float_as_int(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回将浮点数的比特位重新解释为32位整数的结果。

返回类型：`int32`

支持编译模式：SIMT, SIMD

## 69. triton.language.extra.cann.libdevice.float_as_uint

### OP概述

将浮点数的比特位重新解释为32位无符号整数。不进行数值转换。

原型:

```python
triton.language.extra.cann.libdevice.float_as_uint(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回将浮点数的比特位重新解释为32位无符号整数的结果。

返回类型：`uint32`

支持编译模式：SIMT

## 70. triton.language.extra.cann.libdevice.floor

### OP概述

向下取整，返回小于或等于 x 的最大整数。

原型:

```python
triton.language.extra.cann.libdevice.floor(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回向下取整的结果。

返回类型：`float32`

支持编译模式：SIMT

## 71. triton.language.extra.cann.libdevice.fma

### OP概述

融合乘加，计算 x × y + z。

原型:

```python
triton.language.extra.cann.libdevice.fma(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回融合乘加的结果。

返回类型：`float32`

支持编译模式：SIMT

## 72. triton.language.extra.cann.libdevice.fma_rd

### OP概述

向下舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_rd(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回融合乘加的结果。

返回类型：`float32`

支持编译模式：SIMT

## 73. triton.language.extra.cann.libdevice.fma_rn

### OP概述

最近偶数舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_rn(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回融合乘加的结果。

返回类型：`float32`

支持编译模式：SIMT

## 74. triton.language.extra.cann.libdevice.fma_ru

### OP概述

向上舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_ru(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回融合乘加的结果。

返回类型：`float32`

支持编译模式：SIMT

## 75. triton.language.extra.cann.libdevice.fma_rz

### OP概述

向零舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_rz(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回融合乘加的结果。

返回类型：`float32`

支持编译模式：SIMT

## 76. triton.language.extra.cann.libdevice.fmod

### OP概述

浮点数取模，计算 x / y 的余数，结果与 x 同号。

原型:

```python
triton.language.extra.cann.libdevice.fmod(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回浮点数取模的结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 77. triton.language.extra.cann.libdevice.gamma

### OP概述

计算输入参数的伽马函数值。

原型:

```python
triton.language.extra.cann.libdevice.gamma(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的伽马函数值。

返回类型：`float32`

支持编译模式：SIMT

## 78. triton.language.extra.cann.libdevice.hadd

### OP概述

计算 x 和 y 的平均值。

原型:

```python
triton.language.extra.cann.libdevice.hadd(x, y, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`

返回值: `tl.tensor`, 返回 x 和 y 的平均值。

返回类型：`int32`

支持编译模式：SIMT

## 79. triton.language.extra.cann.libdevice.hypot

### OP概述

计算 x 和 y 之间的欧几里得距离。

原型:

```python
triton.language.extra.cann.libdevice.hypot(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回 x 和 y 之间的欧几里得距离。

返回类型：`float32`

支持编译模式：SIMT

## 80. triton.language.extra.cann.libdevice.ilogb

### OP概述

提取浮点数的无偏指数（以2为底的整数对数）。

原型:

```python
triton.language.extra.cann.libdevice.ilogb(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的无偏指数值。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 81. triton.language.extra.cann.libdevice.int2float_rd

### OP概述

向下舍入模式将32位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.int2float_rd(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 82. triton.language.extra.cann.libdevice.int2float_rn

### OP概述

最近偶数舍入模式将32位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.int2float_rn(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 83. triton.language.extra.cann.libdevice.int2float_ru

### OP概述

向上舍入模式将32位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.int2float_ru(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 84. triton.language.extra.cann.libdevice.int2float_rz

### OP概述

向零舍入模式将32位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.int2float_rz(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 85. triton.language.extra.cann.libdevice.int_as_float

### OP概述

将32位整数的比特位重新解释为浮点数。不进行数值转换。

原型:

```python
triton.language.extra.cann.libdevice.int_as_float(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回将32位整数的比特位重新解释为浮点数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 86. triton.language.extra.cann.libdevice.isfinited

### OP概述

判断输入是否为有限值。

原型:

```python
triton.language.extra.cann.libdevice.isfinited(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 若输入为有限值返回 True，否则返回 False。

返回类型：`bool`

支持编译模式：SIMT

## 87. triton.language.extra.cann.libdevice.isinf

### OP概述

判断输入是否为无穷大。

原型:

```python
triton.language.extra.cann.libdevice.isinf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 若输入为无穷大返回 True，否则返回 False。

返回类型：`bool`

支持编译模式：SIMT, SIMD

## 88. triton.language.extra.cann.libdevice.isnan

### OP概述

判断输入是否为 NaN（非数值）。

原型:

```python
triton.language.extra.cann.libdevice.isnan(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 若输入为 NaN 返回 True，否则返回 False。

返回类型：`bool`

支持编译模式：SIMT, SIMD

## 89. triton.language.extra.cann.libdevice.j0

### OP概述

计算输入参数的零阶第一类贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.j0(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的零阶第一类贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 90. triton.language.extra.cann.libdevice.j1

### OP概述

计算输入参数的一阶第一类贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.j1(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的一阶第一类贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 91. triton.language.extra.cann.libdevice.jn

### OP概述

计算输入参数的 n 阶第一类贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.jn(n, x, _builder=None)
```

输入类型：

- n: `int32`
- x: `float32`

返回值: `tl.tensor`, 返回输入参数的 n 阶第一类贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 92. triton.language.extra.cann.libdevice.ldexp

### OP概述

计算 x × 2^exp 的值。

原型:

```python
triton.language.extra.cann.libdevice.ldexp(x, exp, _builder=None)
```

输入类型：

- x: `float32`
- exp: `int32`

返回值: `tl.tensor`, 返回 x × 2^exp 的计算结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 93. triton.language.extra.cann.libdevice.lgamma

### OP概述

计算输入为 x 的伽马函数绝对值的自然对数。

原型:

```python
triton.language.extra.cann.libdevice.lgamma(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入为 x 的伽马函数绝对值的自然对数。

返回类型：`float32`

支持编译模式：SIMT

## 94. triton.language.extra.cann.libdevice.ll2float_rd

### OP概述

向下舍入模式将64位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ll2float_rd(x, _builder=None)
```

输入类型：

- x: `int64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 95. triton.language.extra.cann.libdevice.ll2float_rn

### OP概述

最近偶数舍入模式将64位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ll2float_rn(x, _builder=None)
```

输入类型：

- x: `int64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 96. triton.language.extra.cann.libdevice.ll2float_ru

### OP概述

向上舍入模式将64位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ll2float_ru(x, _builder=None)
```

输入类型：

- x: `int64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 97. triton.language.extra.cann.libdevice.ll2float_rz

### OP概述

向零舍入模式将64位整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ll2float_rz(x, _builder=None)
```

输入类型：

- x: `int64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 98. triton.language.extra.cann.libdevice.llrint

### OP概述

将浮点数舍入为最接近的64位整数值。

原型:

```python
triton.language.extra.cann.libdevice.llrint(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回舍入后的64位整数。

返回类型：`int64`

支持编译模式：SIMT

## 99. triton.language.extra.cann.libdevice.llround

### OP概述

将浮点数四舍五入为最接近的64位整数值。

原型:

```python
triton.language.extra.cann.libdevice.llround(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回舍入后的64位整数。

返回类型：`int64`

支持编译模式：SIMT

## 100. triton.language.extra.cann.libdevice.log

### OP概述

计算输入为 x 的以 e 为底的对数。

原型:

```python
triton.language.extra.cann.libdevice.log(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入为 x 的以 e 为底的对数。

返回类型：`float32`

支持编译模式：SIMT

## 101. triton.language.extra.cann.libdevice.log10

### OP概述

计算输入为 x 的以 10 为底的对数。

原型:

```python
triton.language.extra.cann.libdevice.log10(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入为 x 的以 10 为底的对数。

返回类型：`float32`

支持编译模式：SIMT

## 102. triton.language.extra.cann.libdevice.log1p

### OP概述

计算 log(1 + x) 的值。

原型:

```python
triton.language.extra.cann.libdevice.log1p(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 log(1 + x) 的计算结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 103. triton.language.extra.cann.libdevice.log2

### OP概述

计算输入为 x 的以 2 为底的对数。

原型:

```python
triton.language.extra.cann.libdevice.log2(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入为 x 的以 2 为底的对数。

返回类型：`float32`

支持编译模式：SIMT

## 104. triton.language.extra.cann.libdevice.logb

### OP概述

提取浮点数的指数值。

原型:

```python
triton.language.extra.cann.libdevice.logb(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的指数值。

返回类型：`float32`

支持编译模式：SIMT

## 105. triton.language.extra.cann.libdevice.mul24

### OP概述

计算 x 和 y 的低24位乘法结果。

原型:

```python
triton.language.extra.cann.libdevice.mul24(x, y, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`

返回值: `tl.tensor`, 返回 x 和 y 的低24位乘法结果。

返回类型：`int32`

支持编译模式：SIMT

## 106. triton.language.extra.cann.libdevice.mul_rd

### OP概述

向下舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_rd(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回浮点数乘法的结果。

返回类型：`float32`

支持编译模式：SIMT

## 107. triton.language.extra.cann.libdevice.mul_rn

### OP概述

最近偶数舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_rn(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回浮点数乘法的结果。

返回类型：`float32`

支持编译模式：SIMT

## 108. triton.language.extra.cann.libdevice.mul_ru

### OP概述

向上舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_ru(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回浮点数乘法的结果。

返回类型：`float32`

支持编译模式：SIMT

## 109. triton.language.extra.cann.libdevice.mul_rz

### OP概述

向零舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_rz(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回浮点数乘法的结果。

返回类型：`float32`

支持编译模式：SIMT

## 110. triton.language.extra.cann.libdevice.mulhi

### OP概述

计算 x 和 y 的乘法结果的高 32 位。

原型:

```python
triton.language.extra.cann.libdevice.mulhi(x, y, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`

返回值: `tl.tensor`, 返回 x 和 y 的乘法结果的高 32 位。

返回类型：`int32`

支持编译模式：SIMT

## 111. triton.language.extra.cann.libdevice.nearbyint

### OP概述

将 x 转换为最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.nearbyint(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回最近邻整数。

返回类型：`float32`

支持编译模式：SIMT

## 112. triton.language.extra.cann.libdevice.nextafter

### OP概述

计算从 x 方向朝 y 的下一个可表示浮点数。

原型:

```python
triton.language.extra.cann.libdevice.nextafter(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回下一个可表示浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 113. triton.language.extra.cann.libdevice.norm3d

### OP概述

计算三维向量的欧几里得范数，即 sqrt(x² + y² + z²)。

原型:

```python
triton.language.extra.cann.libdevice.norm3d(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回三维向量的欧几里得范数。

返回类型：`float32`

支持编译模式：SIMT

## 114. triton.language.extra.cann.libdevice.norm4d

### OP概述

计算四维向量的欧几里得范数，即 sqrt(x² + y² + z² + w²)。

原型:

```python
triton.language.extra.cann.libdevice.norm4d(x, y, z, w, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`
- w: `float32`

返回值: `tl.tensor`, 返回四维向量的欧几里得范数。

返回类型：`float32`

支持编译模式：SIMT

## 115. triton.language.extra.cann.libdevice.normcdf

### OP概述

计算标准正态分布的累积分布函数值。

原型:

```python
triton.language.extra.cann.libdevice.normcdf(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回标准正态分布的累积分布函数值。

返回类型：`float32`

支持编译模式：SIMT

## 116. triton.language.extra.cann.libdevice.normcdfinv

### OP概述

计算标准正态分布累积分布函数的逆函数值。

原型:

```python
triton.language.extra.cann.libdevice.normcdfinv(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回标准正态分布累积分布函数的逆函数值。

返回类型：`float32`

支持编译模式：SIMT

## 117. triton.language.extra.cann.libdevice.popc

### OP概述

计算 x 中置位为 1 的数量。

原型:

```python
triton.language.extra.cann.libdevice.popc(x, _builder=None)
```

输入类型：

- x: `int32`

返回值: `tl.tensor`, 返回 x 中置位为 1 的数量，取值范围 \[0, 32]。

返回类型：`int32`

支持编译模式：SIMT

## 118. triton.language.extra.cann.libdevice.pow

### OP概述

幂函数，计算 x 的 y 次方。

原型:

```python
triton.language.extra.cann.libdevice.pow(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回 x 的 y 次方。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 119. triton.language.extra.cann.libdevice.rcbrt

### OP概述

计算 x 的立方根倒数。

原型:

```python
triton.language.extra.cann.libdevice.rcbrt(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的立方根倒数。

返回类型：`float32`

支持编译模式：SIMT

## 120. triton.language.extra.cann.libdevice.rcp_rd

### OP概述

向下舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_rd(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 1 / x。

返回类型：`float32`

支持编译模式：SIMT

## 121. triton.language.extra.cann.libdevice.rcp_rn

### OP概述

最近偶数舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 1 / x。

返回类型：`float32`

支持编译模式：SIMT

## 122. triton.language.extra.cann.libdevice.rcp_ru

### OP概述

向上舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_ru(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 1 / x。

返回类型：`float32`

支持编译模式：SIMT

## 123. triton.language.extra.cann.libdevice.rcp_rz

### OP概述

向零舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_rz(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 1 / x。

返回类型：`float32`

支持编译模式：SIMT

## 124. triton.language.extra.cann.libdevice.reciprocal

### OP概述

计算 1 / x 的倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.reciprocal(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 1 / x。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 125. triton.language.extra.cann.libdevice.relu

### OP概述

修正线性单元函数，当 x > 0 时返回 x，否则返回 0。

原型:

```python
triton.language.extra.cann.libdevice.relu(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回修正线性单元的结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 126. triton.language.extra.cann.libdevice.remainder

### OP概述

计算 x 对 y 的余数，满足 r = x - ny，其中 n 是 x / y 的最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.remainder(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回 x 对 y 的余数。

返回类型：`float32`

支持编译模式：SIMT

## 127. triton.language.extra.cann.libdevice.rhadd

### OP概述

计算 x 和 y 平均值的取整结果。

原型:

```python
triton.language.extra.cann.libdevice.rhadd(x, y, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`

返回值: `tl.tensor`, 返回 x 和 y 平均值的取整结果。

返回类型：`int32`

支持编译模式：SIMT

## 128. triton.language.extra.cann.libdevice.rhypot

### OP概述

计算 x 和 y 之间的欧几里得距离的倒数。

原型:

```python
triton.language.extra.cann.libdevice.rhypot(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回 x 和 y 之间的欧几里得距离的倒数。

返回类型：`float32`

支持编译模式：SIMT

## 129. triton.language.extra.cann.libdevice.rint

### OP概述

按最近偶数舍入模式计算 x 的最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.rint(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的最近邻整数。

返回类型：`float32`

支持编译模式：SIMT

## 130. triton.language.extra.cann.libdevice.rnorm3d

### OP概述

计算三维向量的欧几里得范数的倒数。

原型:

```python
triton.language.extra.cann.libdevice.rnorm3d(x, y, z, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`

返回值: `tl.tensor`, 返回三维向量的欧几里得范数的倒数。

返回类型：`float32`

支持编译模式：SIMT

## 131. triton.language.extra.cann.libdevice.rnorm4d

### OP概述

计算四维向量的欧几里得范数的倒数。

原型:

```python
triton.language.extra.cann.libdevice.rnorm4d(x, y, z, w, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`
- z: `float32`
- w: `float32`

返回值: `tl.tensor`, 返回四维向量的欧几里得范数的倒数。

返回类型：`float32`

支持编译模式：SIMT

## 132. triton.language.extra.cann.libdevice.round

### OP概述

按最近偶数舍入模式计算 x 的最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.round(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的最近邻整数。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 133. triton.language.extra.cann.libdevice.rsqrt

### OP概述

计算 x 的平方根倒数。

原型:

```python
triton.language.extra.cann.libdevice.rsqrt(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根倒数。

返回类型：`float32`

支持编译模式：SIMT

## 134. triton.language.extra.cann.libdevice.rsqrt_rn

### OP概述

按最近偶数舍入模式计算 x 的平方根倒数。

原型:

```python
triton.language.extra.cann.libdevice.rsqrt_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根倒数。

返回类型：`float32`

支持编译模式：SIMT

## 135. triton.language.extra.cann.libdevice.sad

### OP概述

计算 |x-y|+z，其中 x 和 y 是有符号整数，z 是无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.sad(x, y, z, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`
- z: `int32`

返回值: `tl.tensor`, 返回 |x-y|+z。

返回类型：`int32`

支持编译模式：SIMT

## 136. triton.language.extra.cann.libdevice.saturatef

### OP概述

将 x 限制在 \[+0.0, 1.0] 范围内。

原型:

```python
triton.language.extra.cann.libdevice.saturatef(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的饱和值，取值范围 \[+0.0, 1.0]。

返回类型：`float32`

支持编译模式：SIMT

## 137. triton.language.extra.cann.libdevice.scalbn

### OP概述

计算 x × 2^n 的值。

原型:

```python
triton.language.extra.cann.libdevice.scalbn(x, n, _builder=None)
```

输入类型：

- x: `float32`
- n: `int32`

返回值: `tl.tensor`, 返回 x × 2^n 的计算结果。

返回类型：`float32`

支持编译模式：SIMT

## 138. triton.language.extra.cann.libdevice.signbit

### OP概述

获取 x 的符号位。

原型:

```python
triton.language.extra.cann.libdevice.signbit(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的符号位。

返回类型：`int32`

支持编译模式：SIMT

## 139. triton.language.extra.cann.libdevice.sin

### OP概述

计算输入参数 x （弧度）的正弦值。

原型:

```python
triton.language.extra.cann.libdevice.sin(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入 x 的正弦值。

返回类型：`float32`

支持编译模式：SIMT

## 140. triton.language.extra.cann.libdevice.sinh

### OP概述

计算输入参数 x 的双曲正弦值。

原型:

```python
triton.language.extra.cann.libdevice.sinh(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入 x 的双曲正弦值。

返回类型：`float32`

支持编译模式：SIMT

## 141. triton.language.extra.cann.libdevice.sinpi

### OP概述

计算 sin(π × x) 的值。

原型:

```python
triton.language.extra.cann.libdevice.sinpi(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 sin(π × x) 的值。

返回类型：`float32`

支持编译模式：SIMT

## 142. triton.language.extra.cann.libdevice.sqrt

### OP概述

计算 x 的平方根值。

原型:

```python
triton.language.extra.cann.libdevice.sqrt(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根值。

返回类型：`float32`

支持编译模式：SIMT

## 143. triton.language.extra.cann.libdevice.sqrt_rd

### OP概述

向下舍入模式计算 x 的平方根值。

原型:

```python
triton.language.extra.cann.libdevice.sqrt_rd(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根值。

返回类型：`float32`

支持编译模式：SIMT

## 144. triton.language.extra.cann.libdevice.sqrt_rn

### OP概述

最近偶数舍入模式计算 x 的平方根值。

原型:

```python
triton.language.extra.cann.libdevice.sqrt_rn(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根值。

返回类型：`float32`

支持编译模式：SIMT

## 145. triton.language.extra.cann.libdevice.sqrt_ru

### OP概述

向上舍入模式计算 x 的平方根值。

原型:

```python
triton.language.extra.cann.libdevice.sqrt_ru(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根值。

返回类型：`float32`

支持编译模式：SIMT

## 146. triton.language.extra.cann.libdevice.sqrt_rz

### OP概述

向零舍入模式计算 x 的平方根值。

原型:

```python
triton.language.extra.cann.libdevice.sqrt_rz(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回 x 的平方根值。

返回类型：`float32`

支持编译模式：SIMT

## 147. triton.language.extra.cann.libdevice.sub_rd

### OP概述

向下舍入浮点数减法。

原型:

```python
triton.language.extra.cann.libdevice.sub_rd(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回向下舍入的减法结果。

返回类型：`float32`

支持编译模式：SIMT

## 148. triton.language.extra.cann.libdevice.sub_rn

### OP概述

最近偶数舍入浮点数减法。

原型:

```python
triton.language.extra.cann.libdevice.sub_rn(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回最近偶数舍入的减法结果。

返回类型：`float32`

支持编译模式：SIMT

## 149. triton.language.extra.cann.libdevice.sub_ru

### OP概述

向上舍入浮点数减法。

原型:

```python
triton.language.extra.cann.libdevice.sub_ru(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回向上舍入的减法结果。

返回类型：`float32`

支持编译模式：SIMT

## 150. triton.language.extra.cann.libdevice.sub_rz

### OP概述

向零舍入浮点数减法。

原型:

```python
triton.language.extra.cann.libdevice.sub_rz(x, y, _builder=None)
```

输入类型：

- x: `float32`
- y: `float32`

返回值: `tl.tensor`, 返回向零舍入的减法结果。

返回类型：`float32`

支持编译模式：SIMT

## 151. triton.language.extra.cann.libdevice.tan

### OP概述

计算输入参数 x （弧度）的正切值。

原型:

```python
triton.language.extra.cann.libdevice.tan(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入 x 的正切值。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 152. triton.language.extra.cann.libdevice.tanh

### OP概述

计算输入参数 x 的双曲正切值。

原型:

```python
triton.language.extra.cann.libdevice.tanh(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入 x 的双曲正切值。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 153. triton.language.extra.cann.libdevice.tgamma

### OP概述

计算输入参数的伽马函数值。

原型:

```python
triton.language.extra.cann.libdevice.tgamma(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的伽马函数值。

返回类型：`float32`

支持编译模式：SIMT

## 154. triton.language.extra.cann.libdevice.trunc

### OP概述

截断取整，向零舍入到最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.trunc(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回取整结果。

返回类型：`float32`

支持编译模式：SIMT, SIMD

## 155. triton.language.extra.cann.libdevice.uint2float_rd

### OP概述

向下舍入模式将32位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.uint2float_rd(x, _builder=None)
```

输入类型：

- x: `uint32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 156. triton.language.extra.cann.libdevice.uint2float_rn

### OP概述

最近偶数舍入模式将32位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.uint2float_rn(x, _builder=None)
```

输入类型：

- x: `uint32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 157. triton.language.extra.cann.libdevice.uint2float_ru

### OP概述

向上舍入模式将32位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.uint2float_ru(x, _builder=None)
```

输入类型：

- x: `uint32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 158. triton.language.extra.cann.libdevice.uint2float_rz

### OP概述

向零舍入模式将32位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.uint2float_rz(x, _builder=None)
```

输入类型：

- x: `uint32`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 159. triton.language.extra.cann.libdevice.uint_as_float

### OP概述

将32位无符号整数的比特位重新解释为浮点数。不进行数值转换。

原型:

```python
triton.language.extra.cann.libdevice.uint_as_float(x, _builder=None)
```

输入类型：

- x: `uint32`

返回值: `tl.tensor`, 返回将32位无符号整数的比特位重新解释为浮点数的结果。

返回类型：`float32`

支持编译模式：SIMT

## 160. triton.language.extra.cann.libdevice.ull2float_rd

### OP概述

向下舍入模式将64位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ull2float_rd(x, _builder=None)
```

输入类型：

- x: `uint64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 161. triton.language.extra.cann.libdevice.ull2float_rn

### OP概述

最近偶数舍入模式将64位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ull2float_rn(x, _builder=None)
```

输入类型：

- x: `uint64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 162. triton.language.extra.cann.libdevice.ull2float_ru

### OP概述

向上舍入模式将64位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ull2float_ru(x, _builder=None)
```

输入类型：

- x: `uint64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 163. triton.language.extra.cann.libdevice.ull2float_rz

### OP概述

向零舍入模式将64位无符号整数转换为浮点数。

原型:

```python
triton.language.extra.cann.libdevice.ull2float_rz(x, _builder=None)
```

输入类型：

- x: `uint64`

返回值: `tl.tensor`, 返回转换后的浮点数。

返回类型：`float32`

支持编译模式：SIMT

## 164. triton.language.extra.cann.libdevice.umulhi

### OP概述

计算 x 和 y 的无符号乘法结果的高 32 位。

原型:

```python
triton.language.extra.cann.libdevice.umulhi(x, y, _builder=None)
```

输入类型：

- x: `int32`
- y: `int32`

返回值: `tl.tensor`, 返回 x 和 y 的无符号乘法结果的高 32 位。

返回类型：`int32`

支持编译模式：SIMT

## 165. triton.language.extra.cann.libdevice.y0

### OP概述

计算输入参数的零阶第二类贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.y0(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的零阶第二类贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 166. triton.language.extra.cann.libdevice.y1

### OP概述

计算输入参数的一阶第二类贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.y1(x, _builder=None)
```

输入类型：

- x: `float32`

返回值: `tl.tensor`, 返回输入参数的一阶第二类贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT

## 167. triton.language.extra.cann.libdevice.yn

### OP概述

计算输入参数的 n 阶第二类贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.yn(n, x, _builder=None)
```

输入类型：

- n: `int32`
- x: `float32`

返回值: `tl.tensor`, 返回输入参数的 n 阶第二类贝塞尔函数值。

返回类型：`float32`

支持编译模式：SIMT
