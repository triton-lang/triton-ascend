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

## 1. triton.language.extra.cann.libdevice.abs

### OP Overview

Computes the absolute value of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.abs(x, _builder=None)
```

Input Types:
- x: `int32`, `float32`

Return Value: `tl.tensor`, containing the absolute value of the input parameter.

Return Type: `int32`, `float32`

Supported Compilation Mode: SIMT

## 2. triton.language.extra.cann.libdevice.acos

### OP Overview

Computes the inverse cosine (arccos) of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.acos(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse cosine of the input parameter, in the range \[0, π] radians.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 3. triton.language.extra.cann.libdevice.acosh

### OP Overview

Computes the inverse hyperbolic cosine of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.acosh(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse hyperbolic cosine of the input parameter, in the range \[0, +∞].

Return Type: `float32`

Supported Compilation Mode: SIMT

## 4. triton.language.extra.cann.libdevice.add_rd

### OP Overview

Floating-point addition with round-down (toward negative infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.add_rd(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the addition result rounded down.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 5. triton.language.extra.cann.libdevice.add_rn

### OP Overview

Floating-point addition with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.add_rn(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the addition result rounded to the nearest even number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 6. triton.language.extra.cann.libdevice.add_ru

### OP Overview

Floating-point addition with round-up (toward positive infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.add_ru(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the addition result rounded up.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 7. triton.language.extra.cann.libdevice.add_rz

### OP Overview

Floating-point addition with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.add_rz(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the addition result rounded toward zero.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 8. triton.language.extra.cann.libdevice.asin

### OP Overview

Computes the inverse sine (arcsin) of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.asin(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse sine of the input parameter, in the range \[-π/2, π/2] radians.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 9. triton.language.extra.cann.libdevice.asinh

### OP Overview

Computes the inverse hyperbolic sine of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.asinh(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse hyperbolic sine of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 10. triton.language.extra.cann.libdevice.atan

### OP Overview

Computes the inverse tangent (arctan) of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.atan(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse tangent of the input parameter, in the range \[-π/2, π/2] radians.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 11. triton.language.extra.cann.libdevice.atan2

### OP Overview

Two-argument inverse tangent function, computes the arctangent of x / y.

Prototype:

```python
triton.language.extra.cann.libdevice.atan2(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the arctangent of x / y, in the range \[-π, π] radians.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 12. triton.language.extra.cann.libdevice.atanh

### OP Overview

Inverse hyperbolic tangent function, computes the inverse hyperbolic tangent of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.atanh(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse hyperbolic tangent of the input parameter, in the range \[-1, 1].

Return Type: `float32`

Supported Compilation Mode: SIMT

## 13. triton.language.extra.cann.libdevice.brev

### OP Overview

Bit reversal function, reverses the bit order of a 32-bit integer.

Prototype:

```python
triton.language.extra.cann.libdevice.brev(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the 32-bit integer with reversed bit order.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 14. triton.language.extra.cann.libdevice.byte_perm

### OP Overview

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

Prototype:

```python
triton.language.extra.cann.libdevice.byte_perm(x, y, s, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`
- s: `int32`

Return Value: `tl.tensor`, where return[n] := input[selector[n]], where n represents the n-th byte of the output integer.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 15. triton.language.extra.cann.libdevice.cbrt

### OP Overview

Computes the cube root of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.cbrt(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the cube root of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 16. triton.language.extra.cann.libdevice.ceil

### OP Overview

Ceiling operation, returns the smallest integer greater than or equal to x.

Prototype:

```python
triton.language.extra.cann.libdevice.ceil(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the ceiling result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 17. triton.language.extra.cann.libdevice.clz

### OP Overview

Counts the number of leading zeros in a 32-bit integer.

Prototype:

```python
triton.language.extra.cann.libdevice.clz(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the number of leading zeros in the input parameter. Range: \[0, 32].

Return Type: `int32`

Supported Compilation Mode: SIMT

## 18. triton.language.extra.cann.libdevice.copysign

### OP Overview

Generates a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.

Prototype:

```python
triton.language.extra.cann.libdevice.copysign(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 19. triton.language.extra.cann.libdevice.cos

### OP Overview

Computes the cosine of the input parameter (in radians).

Prototype:

```python
triton.language.extra.cann.libdevice.cos(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the cosine of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 20. triton.language.extra.cann.libdevice.cosh

### OP Overview

Computes the hyperbolic cosine of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.cosh(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the hyperbolic cosine of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 21. triton.language.extra.cann.libdevice.cospi

### OP Overview

Computes the value of cos(π × x).

Prototype:

```python
triton.language.extra.cann.libdevice.cospi(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the value of cos(π × x).

Return Type: `float32`

Supported Compilation Mode: SIMT

## 22. triton.language.extra.cann.libdevice.cyl_bessel_i0

### OP Overview

Computes the modified Bessel function of the first kind, order 0, of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i0(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the modified Bessel function of the first kind, order 0, of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 23. triton.language.extra.cann.libdevice.cyl_bessel_i1

### OP Overview

Computes the modified Bessel function of the first kind, order 1, of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i1(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the modified Bessel function of the first kind, order 1, of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 24. triton.language.extra.cann.libdevice.div_rd

### OP Overview

Floating-point division with round-down (toward negative infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.div_rd(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the division result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 25. triton.language.extra.cann.libdevice.div_rn

### OP Overview

Floating-point division with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.div_rn(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the division result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 26. triton.language.extra.cann.libdevice.div_ru

### OP Overview

Floating-point division with round-up (toward positive infinity) rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.div_ru(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the division result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 27. triton.language.extra.cann.libdevice.div_rz

### OP Overview

Floating-point division with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.div_rz(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the division result.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 28. triton.language.extra.cann.libdevice.erf

### OP Overview

Computes the error function of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.erf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the error function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 29. triton.language.extra.cann.libdevice.erfc

### OP Overview

Computes the complementary error function of the input parameter, i.e., 1 - erf(x).

Prototype:

```python
triton.language.extra.cann.libdevice.erfc(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the complementary error function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 30. triton.language.extra.cann.libdevice.erfcinv

### OP Overview

Inverse complementary error function, finds the value y such that x = erfc(y).

Prototype:

```python
triton.language.extra.cann.libdevice.erfcinv(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse complementary error function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 31. triton.language.extra.cann.libdevice.erfcx

### OP Overview

Computes the scaled complementary error function of the input parameter, i.e., exp(x²) × erfc(x).

Prototype:

```python
triton.language.extra.cann.libdevice.erfcx(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the scaled complementary error function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 32. triton.language.extra.cann.libdevice.erfinv

### OP Overview

Inverse error function, finds the value y such that x = erf(y).

Prototype:

```python
triton.language.extra.cann.libdevice.erfinv(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse error function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 33. triton.language.extra.cann.libdevice.exp

### OP Overview

Exponential function, computes e raised to the power of x.

Prototype:

```python
triton.language.extra.cann.libdevice.exp(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of e raised to the power of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 34. triton.language.extra.cann.libdevice.exp10

### OP Overview

Base-10 exponential function, computes 10 raised to the power of x.

Prototype:

```python
triton.language.extra.cann.libdevice.exp10(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of 10 raised to the power of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 35. triton.language.extra.cann.libdevice.exp2

### OP Overview

Base-2 exponential function, computes 2 raised to the power of x.

Prototype:

```python
triton.language.extra.cann.libdevice.exp2(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of 2 raised to the power of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 36. triton.language.extra.cann.libdevice.expm1

### OP Overview

Computes e raised to the power of x, minus 1.

Prototype:

```python
triton.language.extra.cann.libdevice.expm1(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of e raised to the power of x, minus 1.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 37. triton.language.extra.cann.libdevice.fast_cosf

### OP Overview

Fast approximate cosine function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_cosf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate cosine function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 38. triton.language.extra.cann.libdevice.fast_dividef

### OP Overview

Fast approximate division.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_dividef(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the result of fast approximate division.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 39. triton.language.extra.cann.libdevice.fast_exp10f

### OP Overview

Fast approximate base-10 exponential function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_exp10f(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate base-10 exponential function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 40. triton.language.extra.cann.libdevice.fast_expf

### OP Overview

Fast approximate exponential function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_expf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate exponential function.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 41. triton.language.extra.cann.libdevice.fast_log10f

### OP Overview

Fast approximate base-10 logarithm function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_log10f(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate base-10 logarithm function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 42. triton.language.extra.cann.libdevice.fast_log2f

### OP Overview

Fast approximate base-2 logarithm function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_log2f(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate base-2 logarithm function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 43. triton.language.extra.cann.libdevice.fast_logf

### OP Overview

Fast approximate natural logarithm function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_logf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate natural logarithm function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 44. triton.language.extra.cann.libdevice.fast_powf

### OP Overview

Fast approximate power function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_powf(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the result of fast approximate power function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 45. triton.language.extra.cann.libdevice.fast_sinf

### OP Overview

Fast approximate sine function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_sinf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate sine function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 46. triton.language.extra.cann.libdevice.fast_tanf

### OP Overview

Fast approximate tangent function.

Prototype:

```python
triton.language.extra.cann.libdevice.fast_tanf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the fast approximate tangent function.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 47. triton.language.extra.cann.libdevice.fdim

### OP Overview

Computes the positive difference between x and y. When x > y, returns x - y; otherwise returns 0.

Prototype:

```python
triton.language.extra.cann.libdevice.fdim(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the positive difference between x and y.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 48. triton.language.extra.cann.libdevice.fdiv

### OP Overview

Floating-point division.

Prototype:

```python
triton.language.extra.cann.libdevice.fdiv(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the floating-point division result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 49. triton.language.extra.cann.libdevice.ffs

### OP Overview

Finds the first bit set to 1 and returns the index of the lowest bit set to 1.

Prototype:

```python
triton.language.extra.cann.libdevice.ffs(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the index of the lowest bit set to 1. Range: \[0, 32].

Return Type: `int32`

Supported Compilation Mode: SIMT

## 50. triton.language.extra.cann.libdevice.finitef

### OP Overview

Determines whether the input is a finite floating-point number.

Prototype:

```python
triton.language.extra.cann.libdevice.finitef(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, returns True if the input is finite, otherwise returns False.

Return Type: `bool`

Supported Compilation Mode: SIMT

## 51. triton.language.extra.cann.libdevice.flip

### OP Overview

Reverses the order of tensor elements along the specified dimension.

Input Types:
- ptr: `tensor`
- dim: `int32`

Return Value: `tl.tensor`, containing the tensor with elements reversed along the specified dimension.

Return Type: `tensor`

Supported Compilation Mode: SIMT

## 52. triton.language.extra.cann.libdevice.float2int_rd

### OP Overview

Converts a floating-point number to a 32-bit integer with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2int_rd(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit integer.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 53. triton.language.extra.cann.libdevice.float2int_rn

### OP Overview

Converts a floating-point number to a 32-bit integer with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2int_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit integer.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 54. triton.language.extra.cann.libdevice.float2int_ru

### OP Overview

Converts a floating-point number to a 32-bit integer with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2int_ru(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit integer.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 55. triton.language.extra.cann.libdevice.float2int_rz

### OP Overview

Converts a floating-point number to a 32-bit integer with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2int_rz(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit integer.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 56. triton.language.extra.cann.libdevice.float2ll_rd

### OP Overview

Converts a floating-point number to a 64-bit integer with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ll_rd(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit integer.

Return Type: `int64`

Supported Compilation Mode: SIMT

## 57. triton.language.extra.cann.libdevice.float2ll_rn

### OP Overview

Converts a floating-point number to a 64-bit integer with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ll_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit integer.

Return Type: `int64`

Supported Compilation Mode: SIMT

## 58. triton.language.extra.cann.libdevice.float2ll_ru

### OP Overview

Converts a floating-point number to a 64-bit integer with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ll_ru(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit integer.

Return Type: `int64`

Supported Compilation Mode: SIMT

## 59. triton.language.extra.cann.libdevice.float2ll_rz

### OP Overview

Converts a floating-point number to a 64-bit integer with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ll_rz(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit integer.

Return Type: `int64`

Supported Compilation Mode: SIMT

## 60. triton.language.extra.cann.libdevice.float2uint_rd

### OP Overview

Converts a floating-point number to a 32-bit unsigned integer with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2uint_rd(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit unsigned integer.

Return Type: `uint32`

Supported Compilation Mode: SIMT

## 61. triton.language.extra.cann.libdevice.float2uint_rn

### OP Overview

Converts a floating-point number to a 32-bit unsigned integer with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2uint_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit unsigned integer.

Return Type: `uint32`

Supported Compilation Mode: SIMT

## 62. triton.language.extra.cann.libdevice.float2uint_ru

### OP Overview

Converts a floating-point number to a 32-bit unsigned integer with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2uint_ru(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit unsigned integer.

Return Type: `uint32`

Supported Compilation Mode: SIMT

## 63. triton.language.extra.cann.libdevice.float2uint_rz

### OP Overview

Converts a floating-point number to a 32-bit unsigned integer with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2uint_rz(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 32-bit unsigned integer.

Return Type: `uint32`

Supported Compilation Mode: SIMT

## 64. triton.language.extra.cann.libdevice.float2ull_rd

### OP Overview

Converts a floating-point number to a 64-bit unsigned integer with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ull_rd(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit unsigned integer.

Return Type: `uint64`

Supported Compilation Mode: SIMT

## 65. triton.language.extra.cann.libdevice.float2ull_rn

### OP Overview

Converts a floating-point number to a 64-bit unsigned integer with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ull_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit unsigned integer.

Return Type: `uint64`

Supported Compilation Mode: SIMT

## 66. triton.language.extra.cann.libdevice.float2ull_ru

### OP Overview

Converts a floating-point number to a 64-bit unsigned integer with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ull_ru(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit unsigned integer.

Return Type: `uint64`

Supported Compilation Mode: SIMT

## 67. triton.language.extra.cann.libdevice.float2ull_rz

### OP Overview

Converts a floating-point number to a 64-bit unsigned integer with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.float2ull_rz(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the converted 64-bit unsigned integer.

Return Type: `uint64`

Supported Compilation Mode: SIMT

## 68. triton.language.extra.cann.libdevice.float_as_int

### OP Overview

Reinterprets the bit pattern of a floating-point number as a 32-bit integer. No numeric conversion is performed.

Prototype:

```python
triton.language.extra.cann.libdevice.float_as_int(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the bit pattern of the floating-point number reinterpreted as a 32-bit integer.

Return Type: `int32`

Supported Compilation Mode: SIMT, SIMD

## 69. triton.language.extra.cann.libdevice.float_as_uint

### OP Overview

Reinterprets the bit pattern of a floating-point number as a 32-bit unsigned integer. No numeric conversion is performed.

Prototype:

```python
triton.language.extra.cann.libdevice.float_as_uint(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the bit pattern of the floating-point number reinterpreted as a 32-bit unsigned integer.

Return Type: `uint32`

Supported Compilation Mode: SIMT

## 70. triton.language.extra.cann.libdevice.floor

### OP Overview

Floor operation, returns the largest integer less than or equal to x.

Prototype:

```python
triton.language.extra.cann.libdevice.floor(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the floor result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 71. triton.language.extra.cann.libdevice.fma

### OP Overview

Fused multiply-add, computes x × y + z.

Prototype:

```python
triton.language.extra.cann.libdevice.fma(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 72. triton.language.extra.cann.libdevice.fma_rd

### OP Overview

Fused multiply-add operation with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.fma_rd(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 73. triton.language.extra.cann.libdevice.fma_rn

### OP Overview

Fused multiply-add operation with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.fma_rn(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 74. triton.language.extra.cann.libdevice.fma_ru

### OP Overview

Fused multiply-add operation with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.fma_ru(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 75. triton.language.extra.cann.libdevice.fma_rz

### OP Overview

Fused multiply-add operation with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.fma_rz(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the result of fused multiply-add.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 76. triton.language.extra.cann.libdevice.fmod

### OP Overview

Floating-point modulo, computes the remainder of x / y, with the same sign as x.

Prototype:

```python
triton.language.extra.cann.libdevice.fmod(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the floating-point modulo result.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 77. triton.language.extra.cann.libdevice.gamma

### OP Overview

Computes the gamma function of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.gamma(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the gamma function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 78. triton.language.extra.cann.libdevice.hadd

### OP Overview

Computes the average of x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.hadd(x, y, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`

Return Value: `tl.tensor`, containing the average of x and y.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 79. triton.language.extra.cann.libdevice.hypot

### OP Overview

Computes the Euclidean distance between x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.hypot(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the Euclidean distance between x and y.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 80. triton.language.extra.cann.libdevice.ilogb

### OP Overview

Extracts the unbiased exponent (base-2 integer logarithm) of a floating-point number.

Prototype:

```python
triton.language.extra.cann.libdevice.ilogb(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the unbiased exponent of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 81. triton.language.extra.cann.libdevice.int2float_rd

### OP Overview

Converts a 32-bit integer to a floating-point number with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.int2float_rd(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 82. triton.language.extra.cann.libdevice.int2float_rn

### OP Overview

Converts a 32-bit integer to a floating-point number with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.int2float_rn(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 83. triton.language.extra.cann.libdevice.int2float_ru

### OP Overview

Converts a 32-bit integer to a floating-point number with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.int2float_ru(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 84. triton.language.extra.cann.libdevice.int2float_rz

### OP Overview

Converts a 32-bit integer to a floating-point number with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.int2float_rz(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 85. triton.language.extra.cann.libdevice.int_as_float

### OP Overview

Reinterprets the bit pattern of a 32-bit integer as a floating-point number. No numeric conversion is performed.

Prototype:

```python
triton.language.extra.cann.libdevice.int_as_float(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the bit pattern of the 32-bit integer reinterpreted as a floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 86. triton.language.extra.cann.libdevice.isfinited

### OP Overview

Determines whether the input is a finite value.

Prototype:

```python
triton.language.extra.cann.libdevice.isfinited(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, returns True if the input is finite, otherwise returns False.

Return Type: `bool`

Supported Compilation Mode: SIMT

## 87. triton.language.extra.cann.libdevice.isinf

### OP Overview

Determines whether the input is infinity.

Prototype:

```python
triton.language.extra.cann.libdevice.isinf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, returns True if the input is infinity, otherwise returns False.

Return Type: `bool`

Supported Compilation Mode: SIMT, SIMD

## 88. triton.language.extra.cann.libdevice.isnan

### OP Overview

Determines whether the input is NaN (not a number).

Prototype:

```python
triton.language.extra.cann.libdevice.isnan(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, returns True if the input is NaN, otherwise returns False.

Return Type: `bool`

Supported Compilation Mode: SIMT, SIMD

## 89. triton.language.extra.cann.libdevice.j0

### OP Overview

Computes the Bessel function of the first kind of order 0 of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.j0(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the Bessel function of the first kind of order 0 of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 90. triton.language.extra.cann.libdevice.j1

### OP Overview

Computes the Bessel function of the first kind of order 1 of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.j1(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the Bessel function of the first kind of order 1 of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 91. triton.language.extra.cann.libdevice.jn

### OP Overview

Computes the Bessel function of the first kind of order n of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.jn(n, x, _builder=None)
```

Input Types:
- n: `int32`
- x: `float32`

Return Value: `tl.tensor`, containing the Bessel function of the first kind of order n of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 92. triton.language.extra.cann.libdevice.ldexp

### OP Overview

Computes the value of x × 2^exp.

Prototype:

```python
triton.language.extra.cann.libdevice.ldexp(x, exp, _builder=None)
```

Input Types:
- x: `float32`
- exp: `int32`

Return Value: `tl.tensor`, containing the result of x × 2^exp.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 93. triton.language.extra.cann.libdevice.lgamma

### OP Overview

Computes the natural logarithm of the absolute value of the gamma function for input x.

Prototype:

```python
triton.language.extra.cann.libdevice.lgamma(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the natural logarithm of the absolute value of the gamma function for input x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 94. triton.language.extra.cann.libdevice.ll2float_rd

### OP Overview

Converts a 64-bit integer to a floating-point number with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ll2float_rd(x, _builder=None)
```

Input Types:
- x: `int64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 95. triton.language.extra.cann.libdevice.ll2float_rn

### OP Overview

Converts a 64-bit integer to a floating-point number with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ll2float_rn(x, _builder=None)
```

Input Types:
- x: `int64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 96. triton.language.extra.cann.libdevice.ll2float_ru

### OP Overview

Converts a 64-bit integer to a floating-point number with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ll2float_ru(x, _builder=None)
```

Input Types:
- x: `int64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 97. triton.language.extra.cann.libdevice.ll2float_rz

### OP Overview

Converts a 64-bit integer to a floating-point number with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ll2float_rz(x, _builder=None)
```

Input Types:
- x: `int64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 98. triton.language.extra.cann.libdevice.llrint

### OP Overview

Rounds a floating-point number to the nearest 64-bit integer value.

Prototype:

```python
triton.language.extra.cann.libdevice.llrint(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the rounded 64-bit integer.

Return Type: `int64`

Supported Compilation Mode: SIMT

## 99. triton.language.extra.cann.libdevice.llround

### OP Overview

Rounds a floating-point number to the nearest 64-bit integer value.

Prototype:

```python
triton.language.extra.cann.libdevice.llround(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the rounded 64-bit integer.

Return Type: `int64`

Supported Compilation Mode: SIMT

## 100. triton.language.extra.cann.libdevice.log

### OP Overview

Computes the natural (base-e) logarithm of input x.

Prototype:

```python
triton.language.extra.cann.libdevice.log(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the natural logarithm of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 101. triton.language.extra.cann.libdevice.log10

### OP Overview

Computes the base-10 logarithm of input x.

Prototype:

```python
triton.language.extra.cann.libdevice.log10(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the base-10 logarithm of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 102. triton.language.extra.cann.libdevice.log1p

### OP Overview

Computes the value of log(1 + x).

Prototype:

```python
triton.language.extra.cann.libdevice.log1p(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of log(1 + x).

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 103. triton.language.extra.cann.libdevice.log2

### OP Overview

Computes the base-2 logarithm of input x.

Prototype:

```python
triton.language.extra.cann.libdevice.log2(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the base-2 logarithm of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 104. triton.language.extra.cann.libdevice.logb

### OP Overview

Extracts the exponent value of a floating-point number.

Prototype:

```python
triton.language.extra.cann.libdevice.logb(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the exponent value of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 105. triton.language.extra.cann.libdevice.mul24

### OP Overview

Computes the lower 24-bit multiplication result of x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.mul24(x, y, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`

Return Value: `tl.tensor`, containing the lower 24-bit multiplication result of x and y.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 106. triton.language.extra.cann.libdevice.mul_rd

### OP Overview

Floating-point multiplication with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.mul_rd(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 107. triton.language.extra.cann.libdevice.mul_rn

### OP Overview

Floating-point multiplication with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.mul_rn(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 108. triton.language.extra.cann.libdevice.mul_ru

### OP Overview

Floating-point multiplication with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.mul_ru(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 109. triton.language.extra.cann.libdevice.mul_rz

### OP Overview

Floating-point multiplication with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.mul_rz(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the floating-point multiplication result.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 110. triton.language.extra.cann.libdevice.mulhi

### OP Overview

Computes the high 32 bits of the multiplication result of x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.mulhi(x, y, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`

Return Value: `tl.tensor`, containing the high 32 bits of the multiplication result of x and y.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 111. triton.language.extra.cann.libdevice.nearbyint

### OP Overview

Converts x to the nearest integer.

Prototype:

```python
triton.language.extra.cann.libdevice.nearbyint(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the nearest integer.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 112. triton.language.extra.cann.libdevice.nextafter

### OP Overview

Computes the next representable floating-point number from x toward y.

Prototype:

```python
triton.language.extra.cann.libdevice.nextafter(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the next representable floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 113. triton.language.extra.cann.libdevice.norm3d

### OP Overview

Computes the Euclidean norm of a 3D vector, i.e., sqrt(x² + y² + z²).

Prototype:

```python
triton.language.extra.cann.libdevice.norm3d(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the Euclidean norm of the 3D vector.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 114. triton.language.extra.cann.libdevice.norm4d

### OP Overview

Computes the Euclidean norm of a 4D vector, i.e., sqrt(x² + y² + z² + w²).

Prototype:

```python
triton.language.extra.cann.libdevice.norm4d(x, y, z, w, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`
- w: `float32`

Return Value: `tl.tensor`, containing the Euclidean norm of the 4D vector.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 115. triton.language.extra.cann.libdevice.normcdf

### OP Overview

Computes the cumulative distribution function of the standard normal distribution.

Prototype:

```python
triton.language.extra.cann.libdevice.normcdf(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the cumulative distribution function of the standard normal distribution.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 116. triton.language.extra.cann.libdevice.normcdfinv

### OP Overview

Computes the inverse of the cumulative distribution function of the standard normal distribution.

Prototype:

```python
triton.language.extra.cann.libdevice.normcdfinv(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the inverse of the cumulative distribution function of the standard normal distribution.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 117. triton.language.extra.cann.libdevice.popc

### OP Overview

Counts the number of bits set to 1 in x.

Prototype:

```python
triton.language.extra.cann.libdevice.popc(x, _builder=None)
```

Input Types:
- x: `int32`

Return Value: `tl.tensor`, containing the number of bits set to 1 in x. Range: \[0, 32].

Return Type: `int32`

Supported Compilation Mode: SIMT

## 118. triton.language.extra.cann.libdevice.pow

### OP Overview

Power function, computes x raised to the power of y.

Prototype:

```python
triton.language.extra.cann.libdevice.pow(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing x raised to the power of y.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 119. triton.language.extra.cann.libdevice.rcbrt

### OP Overview

Computes the reciprocal cube root of x.

Prototype:

```python
triton.language.extra.cann.libdevice.rcbrt(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the reciprocal cube root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 120. triton.language.extra.cann.libdevice.rcp_rd

### OP Overview

Floating-point reciprocal with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.rcp_rd(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing 1 / x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 121. triton.language.extra.cann.libdevice.rcp_rn

### OP Overview

Floating-point reciprocal with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.rcp_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing 1 / x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 122. triton.language.extra.cann.libdevice.rcp_ru

### OP Overview

Floating-point reciprocal with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.rcp_ru(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing 1 / x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 123. triton.language.extra.cann.libdevice.rcp_rz

### OP Overview

Floating-point reciprocal with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.rcp_rz(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing 1 / x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 124. triton.language.extra.cann.libdevice.reciprocal

### OP Overview

Computes the reciprocal 1 / x.

Prototype:

```python
triton.language.extra.cann.libdevice.reciprocal(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing 1 / x.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 125. triton.language.extra.cann.libdevice.relu

### OP Overview

Rectified linear unit function, returns x when x > 0, otherwise returns 0.

Prototype:

```python
triton.language.extra.cann.libdevice.relu(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the result of the rectified linear unit.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 126. triton.language.extra.cann.libdevice.remainder

### OP Overview

Computes the remainder of x divided by y, where r = x - ny, and n is the nearest integer to x / y.

Prototype:

```python
triton.language.extra.cann.libdevice.remainder(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the remainder of x divided by y.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 127. triton.language.extra.cann.libdevice.rhadd

### OP Overview

Computes the rounded average of x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.rhadd(x, y, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`

Return Value: `tl.tensor`, containing the rounded average of x and y.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 128. triton.language.extra.cann.libdevice.rhypot

### OP Overview

Computes the reciprocal of the Euclidean distance between x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.rhypot(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the reciprocal of the Euclidean distance between x and y.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 129. triton.language.extra.cann.libdevice.rint

### OP Overview

Computes the nearest integer to x using round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.rint(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the nearest integer to x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 130. triton.language.extra.cann.libdevice.rnorm3d

### OP Overview

Computes the reciprocal of the Euclidean norm of a 3D vector.

Prototype:

```python
triton.language.extra.cann.libdevice.rnorm3d(x, y, z, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`

Return Value: `tl.tensor`, containing the reciprocal of the Euclidean norm of the 3D vector.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 131. triton.language.extra.cann.libdevice.rnorm4d

### OP Overview

Computes the reciprocal of the Euclidean norm of a 4D vector.

Prototype:

```python
triton.language.extra.cann.libdevice.rnorm4d(x, y, z, w, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`
- z: `float32`
- w: `float32`

Return Value: `tl.tensor`, containing the reciprocal of the Euclidean norm of the 4D vector.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 132. triton.language.extra.cann.libdevice.round

### OP Overview

Computes the nearest integer to x using round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.round(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the nearest integer to x.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 133. triton.language.extra.cann.libdevice.rsqrt

### OP Overview

Computes the reciprocal square root of x.

Prototype:

```python
triton.language.extra.cann.libdevice.rsqrt(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the reciprocal square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 134. triton.language.extra.cann.libdevice.rsqrt_rn

### OP Overview

Computes the reciprocal square root of x using round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.rsqrt_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the reciprocal square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 135. triton.language.extra.cann.libdevice.sad

### OP Overview

Computes |x-y|+z, where x and y are signed integers and z is an unsigned integer.

Prototype:

```python
triton.language.extra.cann.libdevice.sad(x, y, z, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`
- z: `int32`

Return Value: `tl.tensor`, containing |x-y|+z.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 136. triton.language.extra.cann.libdevice.saturatef

### OP Overview

Clamps x to the range \[+0.0, 1.0].

Prototype:

```python
triton.language.extra.cann.libdevice.saturatef(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the saturated value of x, in the range \[+0.0, 1.0].

Return Type: `float32`

Supported Compilation Mode: SIMT

## 137. triton.language.extra.cann.libdevice.scalbn

### OP Overview

Computes the value of x × 2^n.

Prototype:

```python
triton.language.extra.cann.libdevice.scalbn(x, n, _builder=None)
```

Input Types:
- x: `float32`
- n: `int32`

Return Value: `tl.tensor`, containing the result of x × 2^n.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 138. triton.language.extra.cann.libdevice.signbit

### OP Overview

Extracts the sign bit of x.

Prototype:

```python
triton.language.extra.cann.libdevice.signbit(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the sign bit of x.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 139. triton.language.extra.cann.libdevice.sin

### OP Overview

Computes the sine of the input parameter x (in radians).

Prototype:

```python
triton.language.extra.cann.libdevice.sin(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the sine of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 140. triton.language.extra.cann.libdevice.sinh

### OP Overview

Computes the hyperbolic sine of input parameter x.

Prototype:

```python
triton.language.extra.cann.libdevice.sinh(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the hyperbolic sine of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 141. triton.language.extra.cann.libdevice.sinpi

### OP Overview

Computes the value of sin(π × x).

Prototype:

```python
triton.language.extra.cann.libdevice.sinpi(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the value of sin(π × x).

Return Type: `float32`

Supported Compilation Mode: SIMT

## 142. triton.language.extra.cann.libdevice.sqrt

### OP Overview

Computes the square root of x.

Prototype:

```python
triton.language.extra.cann.libdevice.sqrt(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 143. triton.language.extra.cann.libdevice.sqrt_rd

### OP Overview

Computes the square root of x with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sqrt_rd(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 144. triton.language.extra.cann.libdevice.sqrt_rn

### OP Overview

Computes the square root of x with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sqrt_rn(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 145. triton.language.extra.cann.libdevice.sqrt_ru

### OP Overview

Computes the square root of x with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sqrt_ru(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 146. triton.language.extra.cann.libdevice.sqrt_rz

### OP Overview

Computes the square root of x with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sqrt_rz(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the square root of x.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 147. triton.language.extra.cann.libdevice.sub_rd

### OP Overview

Floating-point subtraction with round-down rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sub_rd(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the subtraction result rounded down.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 148. triton.language.extra.cann.libdevice.sub_rn

### OP Overview

Floating-point subtraction with round-to-nearest-even rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sub_rn(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the subtraction result rounded to the nearest even number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 149. triton.language.extra.cann.libdevice.sub_ru

### OP Overview

Floating-point subtraction with round-up rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sub_ru(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the subtraction result rounded up.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 150. triton.language.extra.cann.libdevice.sub_rz

### OP Overview

Floating-point subtraction with round-toward-zero rounding mode.

Prototype:

```python
triton.language.extra.cann.libdevice.sub_rz(x, y, _builder=None)
```

Input Types:
- x: `float32`
- y: `float32`

Return Value: `tl.tensor`, containing the subtraction result rounded toward zero.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 151. triton.language.extra.cann.libdevice.tan

### OP Overview

Computes the tangent of input parameter x (in radians).

Prototype:

```python
triton.language.extra.cann.libdevice.tan(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the tangent of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 152. triton.language.extra.cann.libdevice.tanh

### OP Overview

Computes the hyperbolic tangent of input parameter x.

Prototype:

```python
triton.language.extra.cann.libdevice.tanh(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the hyperbolic tangent of input x.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 153. triton.language.extra.cann.libdevice.tgamma

### OP Overview

Computes the gamma function of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.tgamma(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the gamma function of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 154. triton.language.extra.cann.libdevice.trunc

### OP Overview

Truncation operation, rounds toward zero to the nearest integer.

Prototype:

```python
triton.language.extra.cann.libdevice.trunc(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the truncation result.

Return Type: `float32`

Supported Compilation Mode: SIMT, SIMD

## 155. triton.language.extra.cann.libdevice.uint2float_rd

### OP Overview

Converts a 32-bit unsigned integer to a floating-point number with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.uint2float_rd(x, _builder=None)
```

Input Types:
- x: `uint32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 156. triton.language.extra.cann.libdevice.uint2float_rn

### OP Overview

Converts a 32-bit unsigned integer to a floating-point number with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.uint2float_rn(x, _builder=None)
```

Input Types:
- x: `uint32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 157. triton.language.extra.cann.libdevice.uint2float_ru

### OP Overview

Converts a 32-bit unsigned integer to a floating-point number with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.uint2float_ru(x, _builder=None)
```

Input Types:
- x: `uint32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 158. triton.language.extra.cann.libdevice.uint2float_rz

### OP Overview

Converts a 32-bit unsigned integer to a floating-point number with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.uint2float_rz(x, _builder=None)
```

Input Types:
- x: `uint32`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 159. triton.language.extra.cann.libdevice.uint_as_float

### OP Overview

Reinterprets the bit pattern of a 32-bit unsigned integer as a floating-point number. No numeric conversion is performed.

Prototype:

```python
triton.language.extra.cann.libdevice.uint_as_float(x, _builder=None)
```

Input Types:
- x: `uint32`

Return Value: `tl.tensor`, containing the bit pattern of the 32-bit unsigned integer reinterpreted as a floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 160. triton.language.extra.cann.libdevice.ull2float_rd

### OP Overview

Converts a 64-bit unsigned integer to a floating-point number with round-down mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ull2float_rd(x, _builder=None)
```

Input Types:
- x: `uint64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 161. triton.language.extra.cann.libdevice.ull2float_rn

### OP Overview

Converts a 64-bit unsigned integer to a floating-point number with round-to-nearest-even mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ull2float_rn(x, _builder=None)
```

Input Types:
- x: `uint64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 162. triton.language.extra.cann.libdevice.ull2float_ru

### OP Overview

Converts a 64-bit unsigned integer to a floating-point number with round-up mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ull2float_ru(x, _builder=None)
```

Input Types:
- x: `uint64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 163. triton.language.extra.cann.libdevice.ull2float_rz

### OP Overview

Converts a 64-bit unsigned integer to a floating-point number with round-toward-zero mode.

Prototype:

```python
triton.language.extra.cann.libdevice.ull2float_rz(x, _builder=None)
```

Input Types:
- x: `uint64`

Return Value: `tl.tensor`, containing the converted floating-point number.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 164. triton.language.extra.cann.libdevice.umulhi

### OP Overview

Computes the high 32 bits of the unsigned multiplication result of x and y.

Prototype:

```python
triton.language.extra.cann.libdevice.umulhi(x, y, _builder=None)
```

Input Types:
- x: `int32`
- y: `int32`

Return Value: `tl.tensor`, containing the high 32 bits of the unsigned multiplication result of x and y.

Return Type: `int32`

Supported Compilation Mode: SIMT

## 165. triton.language.extra.cann.libdevice.y0

### OP Overview

Computes the Bessel function of the second kind of order 0 of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.y0(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the Bessel function of the second kind of order 0 of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 166. triton.language.extra.cann.libdevice.y1

### OP Overview

Computes the Bessel function of the second kind of order 1 of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.y1(x, _builder=None)
```

Input Types:
- x: `float32`

Return Value: `tl.tensor`, containing the Bessel function of the second kind of order 1 of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT

## 167. triton.language.extra.cann.libdevice.yn

### OP Overview

Computes the Bessel function of the second kind of order n of the input parameter.

Prototype:

```python
triton.language.extra.cann.libdevice.yn(n, x, _builder=None)
```

Input Types:
- n: `int32`
- x: `float32`

Return Value: `tl.tensor`, containing the Bessel function of the second kind of order n of the input parameter.

Return Type: `float32`

Supported Compilation Mode: SIMT
