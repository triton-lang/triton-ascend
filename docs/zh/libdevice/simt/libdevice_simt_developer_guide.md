# Libdevice 开发者手册

## SIMT 编译示例

使用 SIMT 编译的 triton kernel 示例

```python
# Enable libdevice SIMT compilation
import os
os.environ['TRITON_ENABLE_LIBDEVICE_SIMT'] = '1'

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
input = torch.randn(shape, dtype=dtype).npu()
output = torch.randn(shape, dtype=dtype).npu()
triton_kernel[ncore, 1, 1](input, output, xblock, xblock_sub, force_simt_only=True)
```

## 1. triton.language.extra.cann.libdevice.abs

### OP概述

计算输入参数的绝对值。

原型:

```python
triton.language.extra.cann.libdevice.abs(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的绝对值。

支持类型：`int32`, `float32`

## 3. triton.language.extra.cann.libdevice.acos

### OP概述

计算输入参数的反余弦值。

原型:

```python
triton.language.extra.cann.libdevice.acos(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的反余弦值，取值范围 \[0, π] 弧度。

支持类型：`float32`

## 4. triton.language.extra.cann.libdevice.acosh

### OP概述

计算输入参数的反双曲余弦值。

原型:

```python
triton.language.extra.cann.libdevice.acosh(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的反双曲余弦值，取值范围 \[0, +∞]。

支持类型：`float32`

## 5. triton.language.extra.cann.libdevice.add_rd

### OP概述

向下舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_rd(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回向下舍入的加法结果。

支持类型：`float32`

## 6. triton.language.extra.cann.libdevice.add_rn

### OP概述

最近偶数舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_rn(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回最近偶数舍入的加法结果。

支持类型：`float32`

## 7. triton.language.extra.cann.libdevice.add_ru

### OP概述

向上舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_ru(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回向上舍入的加法结果。

支持类型：`float32`

## 8. triton.language.extra.cann.libdevice.add_rz

### OP概述

向零舍入浮点数加法。

原型:

```python
triton.language.extra.cann.libdevice.add_rz(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回向零舍入的加法结果。

支持类型：`float32`

## 9. triton.language.extra.cann.libdevice.asin

### OP概述

计算输入参数的反正弦值。

原型:

```python
triton.language.extra.cann.libdevice.asin(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的反正弦值，取值范围 \[-π/2, π/2] 弧度。

支持类型：`float32`

## 10. triton.language.extra.cann.libdevice.asinh

### OP概述

计算输入参数的反双曲正弦值。

原型:

```python
triton.language.extra.cann.libdevice.asinh(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的反双曲正弦值。

支持类型：`float32`

## 11. triton.language.extra.cann.libdevice.atan

### OP概述

计算输入参数的反正切值。

原型:

```python
triton.language.extra.cann.libdevice.atan(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的反正切值，取值范围 \[-π/2, π/2] 弧度。

支持类型：`float32`

## 12. triton.language.extra.cann.libdevice.atan2

### OP概述

反正切函数，计算 x / y 的反正切值。

原型:

```python
triton.language.extra.cann.libdevice.atan2(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x / y 的反正切值，取值范围 \[-π, π] 弧度。

支持类型：`float32`

## 13. triton.language.extra.cann.libdevice.atanh

### OP概述

反双曲正切函数，计算输入参数的反双曲正切值。

原型:

```python
triton.language.extra.cann.libdevice.atanh(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的反双曲正切值，取值范围 \[-1, 1]。

支持类型：`float32`

## 14. triton.language.extra.cann.libdevice.brev

### OP概述

位反转函数，反转32位整数的位顺序。

原型:

```python
triton.language.extra.cann.libdevice.brev(x, _builder=None)
```

返回值: `tl.tensor`, 返回位反转后的32位整数。

支持类型：`int32`

## 15. triton.language.extra.cann.libdevice.byte_perm

### OP概述

原型:

```python
triton.language.extra.cann.libdevice.byte_perm(x, y, s, _builder=None)
```

字节排列操作，从两个32位整数中选择字节组成新整数。输入整数 x 和 y 的字节顺序如下

```cpp
input[0] = x<7:0>     input[1] = x<15:8>
input[2] = x<23:16>   input[3] = x<31:24>
input[4] = y<7:0>     input[5] = y<15:8>
input[6] = y<23:16>   input[7] = y<31:24>
```

字节选择参数 s 为32位整数，各比特位与字节选择对应关系如下

```cpp
selector[0] = s<2:0>    selector[1] = s<6:4>
selector[2] = s<10:8>   selector[3] = s<14:12>
```

返回值: `tl.tensor`, 返回值 return\[n] := input\[selector\[n]]，n 表示输出整数的第 n 个字节。

支持类型：`int32`

## 16. triton.language.extra.cann.libdevice.ceil

### OP概述

向上取整，返回大于或等于 x 的最小整数。

原型:

```python
triton.language.extra.cann.libdevice.ceil(x, _builder=None)
```

返回值: `tl.tensor`, 返回向上取整的结果。

支持类型：`float32`

## 17. triton.language.extra.cann.libdevice.clz

### OP概述

计算32位整数的前导零数量。

原型:

```python
triton.language.extra.cann.libdevice.clz(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的前导零数量。范围 \[0, 32]。

支持类型：`int32`

## 18. triton.language.extra.cann.libdevice.copysign

### OP概述

生成一个浮点数，其绝对值等于 x 的绝对值，符号与 y 相同。

原型:

```python
triton.language.extra.cann.libdevice.copysign(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回一个浮点数，其绝对值等于 x 的绝对值，符号与 y 相同。

支持类型：`float32`

## 19. triton.language.extra.cann.libdevice.cos

### OP概述

计算输入参数（弧度）的余弦值。

原型:

```python
triton.language.extra.cann.libdevice.cos(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的余弦值。

支持类型：`float32`

## 20. triton.language.extra.cann.libdevice.cosh

### OP概述

计算输入参数的双曲余弦值。

原型:

```python
triton.language.extra.cann.libdevice.cosh(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的双曲余弦值。

支持类型：`float32`

## 21. triton.language.extra.cann.libdevice.cyl_bessel_i0

### OP概述

计算输入参数的修正零阶贝塞尔函数值。

原型:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i0(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的修正零阶贝塞尔函数值。

支持类型：`float32`

## 22. triton.language.extra.cann.libdevice.div_rd

### OP概述

向下舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_rd(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回除法结果。

支持类型：`float32`

## 23. triton.language.extra.cann.libdevice.div_rn

### OP概述

最近偶数舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_rn(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回除法结果。

支持类型：`float32`

## 24. triton.language.extra.cann.libdevice.div_ru

### OP概述

向上舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_ru(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回除法结果。

支持类型：`float32` |

## 25. triton.language.extra.cann.libdevice.div_rz

### OP概述

向零舍入浮点数除法。

原型:

```python
triton.language.extra.cann.libdevice.div_rz(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回除法结果。

支持类型：`float32`

## 26. triton.language.extra.cann.libdevice.erfinv

### OP概述

逆误差函数，找到满足 x = erf(y) 的值 y。

原型:

```python
triton.language.extra.cann.libdevice.erfinv(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入参数的逆误差函数值。

支持类型：`float32`

## 27. triton.language.extra.cann.libdevice.exp10

### OP概述

以 10 为底的指数函数，计算 10 的 x 次方。

原型:

```python
triton.language.extra.cann.libdevice.exp10(x, _builder=None)
```

返回值: `tl.tensor`, 返回 10 的 x 次方的计算结果。

支持类型：`float32`

## 29. triton.language.extra.cann.libdevice.exp2

### OP概述

以 2 为底的指数函数，计算 2 的 x 次方。

原型:

```python
triton.language.extra.cann.libdevice.exp2(x, _builder=None)
```

返回值: `tl.tensor`, 返回 2 的 x 次方的计算结果。

支持类型：`float32`

## 30. triton.language.extra.cann.libdevice.exp

### OP概述

指数函数，计算 e 的 x 次方。

原型:

```python
triton.language.extra.cann.libdevice.exp(x, _builder=None)
```

返回值: `tl.tensor`, 返回 e 的 x 次方的计算结果。

支持类型：`float32`

## 30. triton.language.extra.cann.libdevice.expm1

### OP概述

计算 e 的 x 次方减 1 的结果。

原型:

```python
triton.language.extra.cann.libdevice.expm1(x, _builder=None)
```

返回值: `tl.tensor`, 返回 e 的 x 次方减 1 的计算结果。

支持类型：`float32`

## 31. triton.language.extra.cann.libdevice.fast_dividef

### OP概述

快速近似除法。

原型:

```python
triton.language.extra.cann.libdevice.fast_dividef(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回快速近似除法的结果。

支持类型：`float32`

## 32. triton.language.extra.cann.libdevice.fast_expf

### OP概述

快速近似指数函数。

原型:

```python
triton.language.extra.cann.libdevice.fast_expf(x, _builder=None)
```

返回值: `tl.tensor`, 返回快速近似指数函数的结果。

支持类型：`float32`

## 33. triton.language.extra.cann.libdevice.fdim

### OP概述

计算 x 与 y 的正差。当 x > y 时，返回 x - y，否则返回 0。

原型:

```python
triton.language.extra.cann.libdevice.fdim(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 与 y 之间的正差。

支持类型：`float32`

## 34. triton.language.extra.cann.libdevice.ffs

### OP概述

查找第一个被置为1的位，返回最低被置为1的位的索引。

原型:

```python
triton.language.extra.cann.libdevice.ffs(x, _builder=None)
```

返回值: `tl.tensor`, 返回最低被置为1的位的索引，取值范围 \[0, 32]。

支持类型：`int32`

## 35. triton.language.extra.cann.libdevice.float_as_int

### OP概述

将浮点数的比特位重新解释为32位整数。不进行数值转换。

原型:

```python
triton.language.extra.cann.libdevice.float_as_int(x, _builder=None)
```

返回值: `tl.tensor`, 返回将浮点数的比特位重新解释为32位整数的结果。

支持类型：`float32`

## 36. triton.language.extra.cann.libdevice.floor

### OP概述

向下取整，返回小于或等于 x 的最大整数。

原型:

```python
triton.language.extra.cann.libdevice.floor(x, _builder=None)
```

返回值: `tl.tensor`, 返回向下取整的结果。

支持类型：`float32`

## 37. triton.language.extra.cann.libdevice.fma

### OP概述

融合乘加，计算 x × y + z。

原型:

```python
triton.language.extra.cann.libdevice.fma(x, y, z, _builder=None)
```

返回值: `tl.tensor`, 返回融合乘加的结果。

支持类型：`float32`

## 38. triton.language.extra.cann.libdevice.fma_rd

### OP概述

向下舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_rd(x, y, z, _builder=None)
```

返回值: `tl.tensor`, 返回融合乘加的结果。

支持类型：`float32`

## 39. triton.language.extra.cann.libdevice.fma_rn

### OP概述

最近偶数舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_rn(x, y, z, _builder=None)
```

返回值: `tl.tensor`, 返回融合乘加的结果。

支持类型：`float32`

## 40. triton.language.extra.cann.libdevice.fma_ru

### OP概述

向上舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_ru(x, y, z, _builder=None)
```

返回值: `tl.tensor`, 返回融合乘加的结果。

支持类型：`float32`

## 41. triton.language.extra.cann.libdevice.fma_rz

### OP概述

向零舍入模式下的融合乘加操作。

原型:

```python
triton.language.extra.cann.libdevice.fma_rz(x, y, z, _builder=None)
```

返回值: `tl.tensor`, 返回融合乘加的结果。

支持类型：`float32`

## 42. triton.language.extra.cann.libdevice.fmod

### OP概述

浮点数取模，计算 x / y 的余数，结果与 x 同号。

原型:

```python
triton.language.extra.cann.libdevice.fmod(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回浮点数取模的结果。

支持类型：`float32`

## 43. triton.language.extra.cann.libdevice.hadd

### OP概述

计算 x 和 y 的平均值。

原型:

```python
triton.language.extra.cann.libdevice.hadd(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 和 y 的平均值。

支持类型：`float32`

## 44. triton.language.extra.cann.libdevice.hypot

### OP概述

计算 x 和 y 之间的欧几里得距离。

原型:

```python
triton.language.extra.cann.libdevice.hypot(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 和 y 之间的欧几里得距离。

支持类型：`float32`

## 45. triton.language.extra.cann.libdevice.lgamma

### OP概述

计算输入为 x 的伽马函数绝对值的自然对数。

原型:

```python
triton.language.extra.cann.libdevice.lgamma(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入为 x 的伽马函数绝对值的自然对数。

支持类型：`float32`

## 46. triton.language.extra.cann.libdevice.log10

### OP概述

计算输入为 x 的以 10 为底的对数。

原型:

```python
triton.language.extra.cann.libdevice.log10(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入为 x 的以 10 为底的对数。

支持类型：`float32`

## 47. triton.language.extra.cann.libdevice.log2

### OP概述

计算输入为 x 的以 2 为底的对数。

原型:

```python
triton.language.extra.cann.libdevice.log2(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入为 x 的以 2 为底的对数。

支持类型：`float32`

## 48. triton.language.extra.cann.libdevice.log

### OP概述

计算输入为 x 的以 e 为底的对数。

原型:

```python
triton.language.extra.cann.libdevice.log(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入为 x 的以 e 为底的对数。

支持类型：`float32`

## 49. triton.language.extra.cann.libdevice.mul24

### OP概述

计算 x 和 y 的低24位乘法结果。

原型:

```python
triton.language.extra.cann.libdevice.mul24(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 和 y 的低24位乘法结果。

支持类型：`int32`

## 50. triton.language.extra.cann.libdevice.mul_rd

### OP概述

向下舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_rd(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回浮点数乘法的结果。

支持类型：`float32`

## 51. triton.language.extra.cann.libdevice.mul_rn

### OP概述

最近偶数舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_rn(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回浮点数乘法的结果。

支持类型：`float32`

## 52. triton.language.extra.cann.libdevice.mul_ru

### OP概述

向上舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_ru(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回浮点数乘法的结果。

支持类型：`float32`

## 53. triton.language.extra.cann.libdevice.mul_rz

### OP概述

向零舍入浮点数乘法。

原型:

```python
triton.language.extra.cann.libdevice.mul_rz(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回浮点数乘法的结果。

支持类型：`float32`

## 54. triton.language.extra.cann.libdevice.mulhi

### OP概述

计算 x 和 y 的乘法结果的高 32 位。

原型:

```python
triton.language.extra.cann.libdevice.mulhi(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 和 y 的乘法结果的高 32 位。

支持类型：`int32`

## 55. triton.language.extra.cann.libdevice.nearbyint

### OP概述

将 x 转换为最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.nearbyint(x, _builder=None)
```

返回值: `tl.tensor`, 返回最近邻整数。

支持类型：`float32`

## 56. triton.language.extra.cann.libdevice.nextafter

### OP概述

计算从 x 方向朝 y 的下一个可表示浮点数。

原型:

```python
triton.language.extra.cann.libdevice.nextafter(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回下一个可表示浮点数。

支持类型：`float32`

## 57. triton.language.extra.cann.libdevice.popc

### OP概述

计算 x 中置位为 1 的数量。

原型:

```python
triton.language.extra.cann.libdevice.popc(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 中置位为 1 的数量， 取值范围 \[0, 32]。

支持类型：`int32`

## 58. triton.language.extra.cann.libdevice.pow

### OP概述

幂函数，计算 x 的 y 次方。

原型:

```python
triton.language.extra.cann.libdevice.pow(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的 y 次方。

支持类型：`float32`

## 59. triton.language.extra.cann.libdevice.rcp_rd

### OP概述

向下舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_rd(x, _builder=None)
```

返回值: `tl.tensor`, 返回 1 / x。

支持类型：`float32`

## 60. triton.language.extra.cann.libdevice.rcp_rn

### OP概述

最近偶数舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_rn(x, _builder=None)
```

返回值: `tl.tensor`, 返回 1 / x。

支持类型：`float32`

## 61. triton.language.extra.cann.libdevice.rcp_ru

### OP概述

向上舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_ru(x, _builder=None)
```

返回值: `tl.tensor`, 返回 1 / x。

支持类型：`float32`

## 62. triton.language.extra.cann.libdevice.rcp_rz

### OP概述

向零舍入浮点数倒数运算。

原型:

```python
triton.language.extra.cann.libdevice.rcp_rz(x, _builder=None)
```

返回值: `tl.tensor`, 返回 1 / x。

支持类型：`float32`

## 63. triton.language.extra.cann.libdevice.remainder

### OP概述

计算 x 对 y 的余数，满足 r = x - ny，其中 n 是 x / y 的最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.remainder(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 对 y 的余数。

支持类型：`float32`

## 64. triton.language.extra.cann.libdevice.rhadd

### OP概述

计算 x 和 y 平均值的取整结果。

原型:

```python
triton.language.extra.cann.libdevice.rhadd(x, y, _builder=None)
```

返回值: `tl.tensor`, 返回 x 和 y 平均值的取整结果。

支持类型：`float32`

## 65. triton.language.extra.cann.libdevice.rint

### OP概述

按最近偶数舍入模式计算 x 的最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.rint(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的最近邻整数。

支持类型：`float32`

## 66. triton.language.extra.cann.libdevice.round

### OP概述

按最近偶数舍入模式计算 x 的最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.round(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的最近邻整数。

支持类型：`float32`

## 67. triton.language.extra.cann.libdevice.rsqrt

### OP概述

计算 x 的平方根倒数。

原型:

```python
triton.language.extra.cann.libdevice.rsqrt(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的平方根倒数。

支持类型：`float32`

## 68. triton.language.extra.cann.libdevice.rsqrt_rn

### OP概述

按最近偶数舍入模式计算 x 的平方根倒数。

原型:

```python
triton.language.extra.cann.libdevice.rsqrt_rn(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的平方根倒数。

支持类型：`float32`

## 69. triton.language.extra.cann.libdevice.sad

### OP概述

计算 |x-y|+z，其中 x 和 y 是有符号整数，z 是无符号整数。

原型:

```python
triton.language.extra.cann.libdevice.sad(x, y, z, _builder=None)
```

返回值: `tl.tensor`, 返回 |x-y|+z。

支持类型：`float32`

## 70. triton.language.extra.cann.libdevice.saturatef

### OP概述

将 x 限制在 \[+0.0, 1.0] 范围内。

原型:

```python
triton.language.extra.cann.libdevice.saturatef(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的饱和值，取值范围 \[+0.0, 1.0]。

支持类型：`float32`

## 71. triton.language.extra.cann.libdevice.saturatef

### OP概述

获取 x 的符号位。

原型:

```python
triton.language.extra.cann.libdevice.signbit(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的符号位。

支持类型：`float32`

## 72. triton.language.extra.cann.libdevice.sin

### OP概述

计算输入参数 x （弧度）的正弦值。

原型:

```python
triton.language.extra.cann.libdevice.sin(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入 x 的正弦值。

支持类型：`float32`

## 72. triton.language.extra.cann.libdevice.sinh

### OP概述

计算输入参数 x 的双曲正弦值。

原型:

```python
triton.language.extra.cann.libdevice.sinh(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入 x 的双曲正弦值。

支持类型：`float32`

## 74. triton.language.extra.cann.libdevice.sqrt

### OP概述

计算 x 的平方根值。

原型:

```python
triton.language.extra.cann.libdevice.sqrt(x, _builder=None)
```

返回值: `tl.tensor`, 返回 x 的平方根值。

支持类型：`float32`

## 75. triton.language.extra.cann.libdevice.tan

### OP概述

计算输入参数 x （弧度）的正切值。

原型:

```python
triton.language.extra.cann.libdevice.tan(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入 x 的正切值。

支持类型：`float32`

## 75. triton.language.extra.cann.libdevice.tanh

### OP概述

计算输入参数 x 的双曲正切值。

原型:

```python
triton.language.extra.cann.libdevice.tanh(x, _builder=None)
```

返回值: `tl.tensor`, 返回输入 x 的双曲正切值。

支持类型：`float32`

## 77. triton.language.extra.cann.libdevice.trunc

### OP概述

截断取整，向零舍入到最近邻整数。

原型:

```python
triton.language.extra.cann.libdevice.trunc(x, _builder=None)
```

返回值: `tl.tensor`, 返回取整结果。

支持类型：`float32`
