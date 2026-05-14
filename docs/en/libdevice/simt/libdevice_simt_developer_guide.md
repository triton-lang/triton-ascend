# Libdevice Developer Guide

## SIMT Compilation Mode Example

Triton kernel example with SIMT compilation mode

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

## 1. triton.language.extra.cann.abs

### OP Overview

Computes the absolute value of the input parameter.

Prototype:

```python
triton.language.extra.cann.abs(x, _builder=None)
```

Return Value: `tl.tensor`, containing the absolute value of the input parameter.

Supported Datatypes：`int32`, `float32`

## 3. triton.language.extra.cann.acos

### OP Overview

Computes the inverse cosine (arccos) of the input parameter.

Prototype:

```python
triton.language.extra.cann.acos(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse cosine of the input parameter, in the range [0, π] radians.

Supported Datatypes：`float32`

## 4. triton.language.extra.cann.acosh

### OP Overview

Computes the inverse hyperbolic cosine of the input parameter.

Prototype:

```python
triton.language.extra.cann.acosh(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse hyperbolic cosine of the input parameter, in the range [0, +∞].

Supported Datatypes：`float32`

## 5. triton.language.extra.cann.add_rd

### OP Overview

Floating-point addition with round-down (toward negative infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.add_rd(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the addition result rounded down.

Supported Datatypes：`float32`

## 6. triton.language.extra.cann.add_rn

### OP Overview

Floating-point addition with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.add_rn(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the addition result rounded to the nearest even number.

Supported Datatypes：`float32`

## 7. triton.language.extra.cann.add_ru

### OP Overview

Floating-point addition with round-up (toward positive infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.add_ru(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the addition result rounded up.

Supported Datatypes：`float32`

## 8. triton.language.extra.cann.add_rz

### OP Overview

Floating-point addition with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.add_rz(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the addition result rounded toward zero.

Supported Datatypes：`float32`

## 9. triton.language.extra.cann.asin

### OP Overview

Computes the inverse sine (arcsin) of the input parameter.

Prototype:

```python
triton.language.extra.cann.asin(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse sine of the input parameter, in the range [-π/2, π/2] radians.

Supported Datatypes：`float32`

## 10. triton.language.extra.cann.asinh

### OP Overview

Computes the inverse hyperbolic sine of the input parameter.

Prototype:

```python
triton.language.extra.cann.asinh(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse hyperbolic sine of the input parameter.

Supported Datatypes：`float32`

## 11. triton.language.extra.cann.atan

### OP Overview

Computes the inverse tangent (arctan) of the input parameter.

Prototype:

```python
triton.language.extra.cann.atan(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse tangent of the input parameter, in the range [-π/2, π/2] radians.

Supported Datatypes：`float32`

## 12. triton.language.extra.cann.atan2

### OP Overview

Two-argument inverse tangent function, computes the arctangent of x / y.

Prototype:

```python
triton.language.extra.cann.atan2(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the arctangent of x / y, in the range [-π, π] radians.

Supported Datatypes：`float32`

## 13. triton.language.extra.cann.atanh

### OP Overview

Inverse hyperbolic tangent function, computes the inverse hyperbolic tangent of the input parameter.

Prototype:

```python
triton.language.extra.cann.atanh(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse hyperbolic tangent of the input parameter, in the range [-1, 1].

Supported Datatypes：`float32`

## 14. triton.language.extra.cann.brev

### OP Overview

Bit reversal function, reverses the bit order of a 32-bit integer.

Prototype:

```python
triton.language.extra.cann.brev(x, _builder=None)
```

Return Value: `tl.tensor`, containing the 32-bit integer with reversed bit order.

Supported Datatypes：`int32`

## 15. triton.language.extra.cann.byte_perm

### OP Overview

Prototype:

```python
triton.language.extra.cann.byte_perm(x, y, s, _builder=None)
```

Byte permutation operation, selects bytes from two 32-bit integers to form a new integer. The byte order of input integers x and y is as follows:

```cpp
input[0] = x<7:0>     input[1] = x<15:8>
input[2] = x<23:16>   input[3] = x<31:24>
input[4] = y<7:0>     input[5] = y<15:8>
input[6] = y<23:16>   input[7] = y<31:24>
```

The byte selection parameter s is a 32-bit integer, with each bit group corresponding to byte selection as follows:

```cpp
selector[0] = s<2:0>    selector[1] = s<6:4>
selector[2] = s<10:8>   selector[3] = s<14:12>
```

Return Value: `tl.tensor`, where return[n] := input[selector[n]], where n represents the n-th byte of the output integer.

Supported Datatypes：`int32`

## 16. triton.language.extra.cann.ceil

### OP Overview

Ceiling operation, returns the smallest integer greater than or equal to x.

Prototype:

```python
triton.language.extra.cann.ceil(x, _builder=None)
```

Return Value: `tl.tensor`, containing the ceiling result.

Supported Datatypes：`float32`

## 17. triton.language.extra.cann.clz

### OP Overview

Counts the number of leading zeros in a 32-bit integer.

Prototype:

```python
triton.language.extra.cann.clz(x, _builder=None)
```

Return Value: `tl.tensor`, containing the number of leading zeros in the input parameter. Range: [0, 32].

Supported Datatypes：`int32`

## 18. triton.language.extra.cann.copysign

### OP Overview

Generates a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.

Prototype:

```python
triton.language.extra.cann.copysign(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.

Supported Datatypes：`float32`

## 19. triton.language.extra.cann.cos

### OP Overview

Computes the cosine of the input parameter (in radians).

Prototype:

```python
triton.language.extra.cann.cos(x, _builder=None)
```

Return Value: `tl.tensor`, containing the cosine of the input parameter.

Supported Datatypes：`float32`

## 20. triton.language.extra.cann.cosh

### OP Overview

Computes the hyperbolic cosine of the input parameter.

Prototype:

```python
triton.language.extra.cann.cosh(x, _builder=None)
```

Return Value: `tl.tensor`, containing the hyperbolic cosine of the input parameter.

## 21. triton.language.extra.cann.cyl_bessel_i0

### OP Overview

Computes the modified Bessel function of the first kind, order 0, of the input parameter.

Prototype:

```python
triton.language.extra.cann.cyl_bessel_i0(x, _builder=None)
```

Return Value: `tl.tensor`, containing the modified Bessel function of the first kind, order 0, of the input parameter.

Supported Datatypes：`float32`

## 22. triton.language.extra.cann.div_rd

### OP Overview

Floating-point division with round-down (toward negative infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.div_rd(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the division result.

Supported Datatypes：`float32`

## 23. triton.language.extra.cann.div_rn

### OP Overview

Floating-point division with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.div_rn(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the division result.

Supported Datatypes：`float32`

## 24. triton.language.extra.cann.div_ru

### OP Overview

Floating-point division with round-up (toward positive infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.div_ru(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the division result.

Supported Datatypes：`float32`

## 25. triton.language.extra.cann.div_rz

### OP Overview

Floating-point division with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.div_rz(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the division result.

Supported Datatypes：`float32`

## 26. triton.language.extra.cann.erfinv

### OP Overview

Inverse error function, finds the value y such that x = erf(y).

Prototype:

```python
triton.language.extra.cann.erfinv(x, _builder=None)
```

Return Value: `tl.tensor`, containing the inverse error function of the input parameter.

Supported Datatypes：`float32`

## 27. triton.language.extra.cann.exp10

### OP Overview

Base-10 exponential function, computes 10 raised to the power of x.

Prototype:

```python
triton.language.extra.cann.exp10(x, _builder=None)
```

Return Value: `tl.tensor`, containing the result of 10 raised to the power of x.

Supported Datatypes：`float32`

## 28. triton.language.extra.cann.exp2

### OP Overview

Base-2 exponential function, computes 2 raised to the power of x.

Prototype:

```python
triton.language.extra.cann.exp2(x, _builder=None)
```

Return Value: `tl.tensor`, containing the result of 2 raised to the power of x.

Supported Datatypes：`float32`

## 29. triton.language.extra.cann.exp

### OP Overview

Exponential function, computes e raised to the power of x.

Prototype:

```python
triton.language.extra.cann.exp(x, _builder=None)
```

Return Value: `tl.tensor`, containing the result of e raised to the power of x.

Supported Datatypes：`float32`

## 30. triton.language.extra.cann.expm1

### OP Overview

Computes e raised to the power of x, minus 1.

Prototype:

```python
triton.language.extra.cann.expm1(x, _builder=None)
```

Return Value: `tl.tensor`, containing the result of e raised to the power of x, minus 1.

## 31. triton.language.extra.cann.fast_dividef

### OP Overview

Fast approximate division.

Prototype:

```python
triton.language.extra.cann.fast_dividef(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fast approximate division.

Supported Datatypes：`float32`

## 32. triton.language.extra.cann.fast_expf

### OP Overview

Fast approximate exponential function.

Prototype:

```python
triton.language.extra.cann.fast_expf(x, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fast approximate exponential function.

Supported Datatypes：`float32`

## 33. triton.language.extra.cann.fdim

### OP Overview

Computes the positive difference between x and y. When x > y, returns x - y; otherwise returns 0.

Prototype:

```python
triton.language.extra.cann.fdim(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the positive difference between x and y.

Supported Datatypes：`float32`

## 34. triton.language.extra.cann.ffs

### OP Overview

Finds the first bit set to 1, returns the index of the lowest bit set to 1.

Prototype:

```python
triton.language.extra.cann.ffs(x, _builder=None)
```

Return Value: `tl.tensor`, containing the index of the lowest bit set to 1. Range: [0, 32].

Supported Datatypes：`int32`

## 35. triton.language.extra.cann.float_as_int

### OP Overview

Reinterprets the bit pattern of a floating-point number as a 32-bit integer. No numeric conversion is performed.

Prototype:

```python
triton.language.extra.cann.float_as_int(x, _builder=None)
```

Return Value: `tl.tensor`, containing the bit pattern of the floating-point number reinterpreted as a 32-bit integer.

## 36. triton.language.extra.cann.floor

### OP Overview

Floor operation, returns the largest integer less than or equal to x.

Prototype:

```python
triton.language.extra.cann.floor(x, _builder=None)
```

Return Value: `tl.tensor`, containing the floor result.

Supported Datatypes：`float32`

## 37. triton.language.extra.cann.fma

### OP Overview

Fused multiply-add, computes x × y + z.

Prototype:

```python
triton.language.extra.cann.fma(x, y, z, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Supported Datatypes：`float32`

## 39. triton.language.extra.cann.fma_rn

### OP Overview

Fused multiply-add operation with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.fma_rd(x, y, z, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Supported Datatypes：`float32`

## 39. triton.language.extra.cann.fma_rn

### OP Overview

Fused multiply-add operation with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.fma_rn(x, y, z, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Supported Datatypes：`float32`

## 40. triton.language.extra.cann.fma_ru

### OP Overview

Fused multiply-add operation with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.fma_ru(x, y, z, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Supported Datatypes：`float32`

## 41. triton.language.extra.cann.fma_rz

### OP Overview

Fused multiply-add operation with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.fma_rz(x, y, z, _builder=None)
```

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Supported Datatypes：`float32`

## 42. triton.language.extra.cann.fmod

### OP Overview

Floating-point modulo, computes the remainder of x / y, with the same sign as x.

Prototype:

```python
triton.language.extra.cann.fmod(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the floating-point modulo result.

Supported Datatypes：`float32`

## 43. triton.language.extra.cann.hadd

### OP Overview

Computes the average of x and y.

Prototype:

```python
triton.language.extra.cann.hadd(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the average of x and y.

Supported Datatypes：`int32`

## 44. triton.language.extra.cann.hypot

### OP Overview

Computes the Euclidean distance between x and y.

Prototype:

```python
triton.language.extra.cann.hypot(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the Euclidean distance between x and y.

Supported Datatypes：`float32`

## 45. triton.language.extra.cann.lgamma

### OP Overview

Computes the natural logarithm of the absolute value of the gamma function for input x.

Prototype:

```python
triton.language.extra.cann.lgamma(x, _builder=None)
```

Return Value: `tl.tensor`, containing the natural logarithm of the absolute value of the gamma function for input x.

Supported Datatypes：`float32`

## 46. triton.language.extra.cann.log10

### OP Overview

Computes the base-10 logarithm of input x.

Prototype:

```python
triton.language.extra.cann.log10(x, _builder=None)
```

Return Value: `tl.tensor`, containing the base-10 logarithm of input x.

Supported Datatypes：`float32`

## 47. triton.language.extra.cann.log2

### OP Overview

Computes the base-2 logarithm of input x.

Prototype:

```python
triton.language.extra.cann.log2(x, _builder=None)
```

Return Value: `tl.tensor`, containing the base-2 logarithm of input x.

Supported Datatypes：`float32`

## 48. triton.language.extra.cann.log

### OP Overview

Computes the natural (base-e) logarithm of input x.

Prototype:

```python
triton.language.extra.cann.log(x, _builder=None)
```

Return Value: `tl.tensor`, containing the natural logarithm of input x.

Supported Datatypes：`float32`

## 49. triton.language.extra.cann.mul24

### OP Overview

Computes the lower 24-bit multiplication result of x and y.

Prototype:

```python
triton.language.extra.cann.mul24(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the lower 24-bit multiplication result of x and y.

Supported Datatypes：`int32`

## 50. triton.language.extra.cann.mul_rd

### OP Overview

Floating-point multiplication with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.mul_rd(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Supported Datatypes：`float32`

## 51. triton.language.extra.cann.mul_rn

### OP Overview

Floating-point multiplication with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.mul_rn(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Supported Datatypes：`float32`

## 52. triton.language.extra.cann.mul_ru

### OP Overview

Floating-point multiplication with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.mul_ru(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Supported Datatypes：`float32`

## 53. triton.language.extra.cann.mul_rz

### OP Overview

Floating-point multiplication with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.mul_rz(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Supported Datatypes：`float32`

## 54. triton.language.extra.cann.mulhi

### OP Overview

Computes the high 32 bits of the multiplication result of x and y.

Prototype:

```python
triton.language.extra.cann.mulhi(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the high 32 bits of the multiplication result of x and y.

Supported Datatypes：`int32`

## 54. triton.language.extra.cann.nearbyint

### OP Overview

Converts x to the nearest integer.

Prototype:

```python
triton.language.extra.cann.nearbyint(x, _builder=None)
```

Return Value: `tl.tensor`, containing the nearest integer.

Supported Datatypes：`float32`

## 56. triton.language.extra.cann.nextafter

### OP Overview

Computes the next representable floating-point number from x toward y.

Prototype:

```python
triton.language.extra.cann.nextafter(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the next representable floating-point number.

Supported Datatypes：`float32`

## 57. triton.language.extra.cann.popc

### OP Overview

Counts the number of bits set to 1 in x.

Prototype:

```python
triton.language.extra.cann.popc(x, _builder=None)
```

Return Value: `tl.tensor`, containing the number of bits set to 1 in x. Range: [0, 32].

Supported Datatypes：`int32`

## 58. triton.language.extra.cann.pow

### OP Overview

Power function, computes x raised to the power of y.

Prototype:

```python
triton.language.extra.cann.pow(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing x raised to the power of y.

Supported Datatypes：`float32`

## 59. triton.language.extra.cann.rcp_rd

### OP Overview

Floating-point reciprocal with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.rcp_rd(x, _builder=None)
```

Return Value: `tl.tensor`, containing 1 / x.

Supported Datatypes：`float32`

## 60. triton.language.extra.cann.rcp_rn

### OP Overview

Floating-point reciprocal with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.rcp_rn(x, _builder=None)
```

Return Value: `tl.tensor`, containing 1 / x.

Supported Datatypes：`float32`

## 61. triton.language.extra.cann.rcp_ru

### OP Overview

Floating-point reciprocal with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.rcp_ru(x, _builder=None)
```

Return Value: `tl.tensor`, containing 1 / x.

Supported Datatypes：`float32`

## 62. triton.language.extra.cann.rcp_rz

### OP Overview

Floating-point reciprocal with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.rcp_rz(x, _builder=None)
```

Return Value: `tl.tensor`, containing 1 / x.

Supported Datatypes：`float32`

## 63. triton.language.extra.cann.remainder

### OP Overview

Computes the remainder of x divided by y, where r = x - ny, and n is the nearest integer to x / y.

Prototype:

```python
triton.language.extra.cann.remainder(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the remainder of x divided by y.

Supported Datatypes：`float32`

## 64. triton.language.extra.cann.rhadd

### OP Overview

Computes the rounded average of x and y.

Prototype:

```python
triton.language.extra.cann.rhadd(x, y, _builder=None)
```

Return Value: `tl.tensor`, containing the rounded average of x and y.

Supported Datatypes：`int32`

## 65. triton.language.extra.cann.rint

### OP Overview

Computes the nearest integer to x using round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.rint(x, _builder=None)
```

Return Value: `tl.tensor`, containing the nearest integer to x.

Supported Datatypes：`float32`

## 66. triton.language.extra.cann.round

### OP Overview

Computes the nearest integer to x using round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.round(x, _builder=None)
```

Return Value: `tl.tensor`, containing the nearest integer to x.

Supported Datatypes：`float32`

## 67. triton.language.extra.cann.rsqrt

### OP Overview

Computes the reciprocal square root of x.

Prototype:

```python
triton.language.extra.cann.rsqrt(x, _builder=None)
```

Return Value: `tl.tensor`, containing the reciprocal square root of x.

Supported Datatypes：`float32`

## 68. triton.language.extra.cann.rsqrt_rn

### OP Overview

Computes the reciprocal square root of x using round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.rsqrt_rn(x, _builder=None)
```

Return Value: `tl.tensor`, containing the reciprocal square root of x.

Supported Datatypes：`float32`

## 69. triton.language.extra.cann.sad

### OP Overview

Computes |x-y|+z, where x and y are signed integers and z is an unsigned integer.

Prototype:

```python
triton.language.extra.cann.sad(x, y, z, _builder=None)
```

Return Value: `tl.tensor`, containing |x-y|+z.

Supported Datatypes：`float32`

## 70. triton.language.extra.cann.saturatef

### OP Overview

Clamps x to the range [+0.0, 1.0].

Prototype:

```python
triton.language.extra.cann.saturatef(x, _builder=None)
```

Return Value: `tl.tensor`, containing the saturated value of x, in the range [+0.0, 1.0].

Supported Datatypes：`float32`

## 71. triton.language.extra.cann.signbit

### OP Overview

Extracts the sign bit of x.

Prototype:

```python
triton.language.extra.cann.signbit(x, _builder=None)
```

Return Value: `tl.tensor`, containing the sign bit of x.

Supported Datatypes：`float32`

## 72. triton.language.extra.cann.sin

### OP Overview

Computes the sine of the input parameter x (in radians).

Prototype:

```python
triton.language.extra.cann.sin(x, _builder=None)
```

Return Value: `tl.tensor`, containing the sine of input x.

Supported Datatypes：`float32`

## 73. triton.language.extra.cann.sinh

### OP Overview

Computes the hyperbolic sine of input parameter x.

Prototype:

```python
triton.language.extra.cann.sinh(x, _builder=None)
```

Return Value: `tl.tensor`, containing the hyperbolic sine of input x.

Supported Datatypes：`float32`

## 74. triton.language.extra.cann.sqrt

### OP Overview

Computes the square root of x.

Prototype:

```python
triton.language.extra.cann.sqrt(x, _builder=None)
```

Return Value: `tl.tensor`, containing the square root of x.

Supported Datatypes：`float32`

## 75. triton.language.extra.cann.tan

### OP Overview

Computes the tangent of input parameter x (in radians).

Prototype:

```python
triton.language.extra.cann.tan(x, _builder=None)
```

Return Value: `tl.tensor`, containing the tangent of input x.

Supported Datatypes：`float32`

## 76. triton.language.extra.cann.tanh

### OP Overview

Computes the hyperbolic tangent of input parameter x.

Prototype:

```python
triton.language.extra.cann.tanh(x, _builder=None)
```

Return Value: `tl.tensor`, containing the hyperbolic tangent of input x.

Supported Datatypes：`float32`

## 77. triton.language.extra.cann.trunc

### OP Overview

Truncation operation, rounds toward zero to the nearest integer.

Prototype:

```python
triton.language.extra.cann.trunc(x, _builder=None)
```

Return Value: `tl.tensor`, containing the truncation result.

Supported Datatypes：`float32`
