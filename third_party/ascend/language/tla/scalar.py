# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Scalar expressions for explicit TLA programs."""

import math

import triton
from triton.language import core as tl
from triton.language.math import abs as _triton_abs

__all__ = [
    "select", "abs", "maximum", "minimum", "cast", "max_value", "cdiv", "infinity", "exp", "log1p", "sigmoid", "clamp"
]


def _check_scalar(value, name):
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, tl.tensor):
        if value.type.is_block() or not (value.dtype.is_int() or value.dtype.is_floating()):
            raise TypeError(f"{name} must be a boolean, integer, or floating-point scalar; got {value.type}")
        if value.dtype == tl.bfloat16:
            raise TypeError(f"{name} does not support bfloat16")
        return value
    if type(value) not in (bool, int, float):
        raise TypeError(f"{name} must be a boolean, integer, or floating-point scalar; got {type(value).__name__}")
    return value


@tl.builtin
def select(condition, true_value, false_value, _semantic=None):
    """Scalar-only ``tl.where``, with identical conversion and promotion rules.

    Bfloat16 inputs are unsupported. Condition conversion,
    branch type promotion, and diagnostics follow ``tl.where`` directly.
    Both value expressions are evaluated before the selection. This operation
    cannot guard side effects or out-of-bounds accesses.
    """
    condition = _check_scalar(condition, "tla.select condition")
    true_value = _check_scalar(true_value, "tla.select true_value")
    false_value = _check_scalar(false_value, "tla.select false_value")
    return tl.where(condition, true_value, false_value, _semantic=_semantic)


@tl.builtin
def abs(x, _semantic=None):
    """Return the absolute value of a scalar, following ``tl.abs``.

    Accepts Python and runtime boolean, integer, float16 and float32
    scalars. Runtime inputs retain their dtype; unsigned integers and booleans
    are unchanged. Signed integer minima retain their fixed-width value,
    without saturation or widening. Floating-point absolute value clears the
    sign of zero, infinity and NaN; NaN payload preservation is not guaranteed.

    Bfloat16, float64 and float8 inputs are unsupported. Buffers, memory access
    positions, SIMD registers and Blocks require their explicit interfaces.
    This operation allocates no buffer and performs no SIMD broadcast.
    """
    x = _check_scalar(x, "tla.abs x")
    if isinstance(x, tl.tensor) and x.dtype.is_floating() and x.dtype not in (tl.float16, tl.float32):
        raise TypeError(f"tla.abs does not support dtype {x.dtype}")
    return _triton_abs(x, _semantic=_semantic)


def _fp32_scalar(x, name, semantic):
    x = tl._unwrap_if_constexpr(x)
    if isinstance(x, tl.tensor):
        if not x.type.is_block() and x.dtype == tl.float32:
            return x
    elif isinstance(x, float):
        return semantic.scalar_constant(x, tl.float32)
    raise TypeError(f"{name} expects a float32 scalar tensor or a Python float")


# On the tested Ascend950PR TLA pipeline, scalar exp and log lower to
# expf and a vln template call respectively, and both fail to link.
# Arithmetic expansions keep all intermediates in scalar values.
_LN2_HI = tl.constexpr(0.693145751953125)
_LN2_LO = tl.constexpr(1.428606765330187e-6)


@triton.jit
def _normal_power_of_two(exponent):
    return tl.cast((exponent + 127) << 23, tl.float32, bitcast=True)


@triton.jit
def _exp_fp32(x):
    overflow = x >= 88.72283935546875
    underflow = x <= -103.97208404541016
    in_range = (x == x) & ~overflow & ~underflow
    value = tl.where(in_range, x, 0.0)

    # Choose the nearest multiple of ln(2). Its split keeps the remainder
    # accurate, with absolute value at most ln(2)/2.
    n = tl.cast(value * 1.4426950408889634 + tl.where(value >= 0.0, 0.5, -0.5), tl.int32)
    n_float = tl.cast(n, tl.float32)
    r = (value - n_float * _LN2_HI) - n_float * _LN2_LO
    poly = (1.0 / 5040.0) * r + (1.0 / 720.0)
    poly = poly * r + (1.0 / 120.0)
    poly = poly * r + (1.0 / 24.0)
    poly = poly * r + (1.0 / 6.0)
    poly = poly * r + 0.5
    poly = (poly * r + 1.0) * r + 1.0

    # Split the scale so both factors use normal exponents even near the
    # output's overflow and subnormal boundaries.
    high = tl.minimum(tl.maximum(n, -126), 127)
    y = (poly * _normal_power_of_two(n - high)) * _normal_power_of_two(high)
    return tl.where(overflow, float("inf"), tl.where(underflow, 0.0, tl.where(in_range, y, x)))


@triton.jit
def _log1p_fp32(x):
    # The smallest positive 1+x from a float32 x > -1 is normal, so extracting
    # its exponent and mantissa does not require subnormal normalization.
    value = tl.where((x > -1.0) & (x < float("inf")), x, 0.0)
    u = 1.0 + value
    bits = tl.cast(u, tl.uint32, bitcast=True)
    exponent = tl.cast(bits >> 23, tl.int32) - 127
    mantissa = tl.cast((bits & 0x007FFFFF) | 0x3F800000, tl.float32, bitcast=True)
    upper = mantissa > 1.4142135623730951
    mantissa = tl.where(upper, mantissa * 0.5, mantissa)
    exponent = exponent + tl.cast(upper, tl.int32)

    # For exponent zero, the original input already gives m-1 exactly.
    # Elsewhere |log1p(x)| is bounded away from zero, so rounding 1+x
    # contributes only a small absolute error.
    delta = tl.where(exponent == 0, value, mantissa - 1.0)
    # ln(m) = 2 * atanh(z), z = (m-1)/(m+1) and abs(z) <= 0.172.
    z = delta / (2.0 + delta)
    z2 = z * z
    poly = (1.0 / 11.0) * z2 + (1.0 / 9.0)
    poly = poly * z2 + (1.0 / 7.0)
    poly = poly * z2 + (1.0 / 5.0)
    poly = poly * z2 + (1.0 / 3.0)
    exponent_float = tl.cast(exponent, tl.float32)
    log_u = exponent_float * _LN2_HI + (exponent_float * _LN2_LO + 2.0 * z * (poly * z2 + 1.0))

    # This selection returns very small x and preserves signed zero, +inf and NaN.
    y = tl.where(u == 1.0, x, log_u)
    domain_error = tl.where(x == -1.0, -float("inf"), float("nan"))
    return tl.where(x <= -1.0, domain_error, y)


@triton.jit
def _sigmoid_fp32(x):
    # The exponent is always nonpositive, so negative inputs cannot
    # overflow before division.
    nonnegative = x >= 0.0
    z = _exp_fp32(tl.where(nonnegative, -x, x))
    return tl.where(nonnegative, 1.0, z) / (1.0 + z)


@tl.builtin
def exp(x, _semantic=None, _generator=None):
    """Return the natural exponential of a float32 scalar or Python float.

    The result is a float32 scalar. Use the SIMD-specific interface for
    vector registers.
    """
    x = _fp32_scalar(x, "tla.exp", _semantic)
    return _generator.call_JitFunction(_exp_fp32, [x], {})


@tl.builtin
def log1p(x, _semantic=None, _generator=None):
    """Return log(1 + x) for a float32 scalar or Python float.

    Preserve small inputs, including signed zero. Negative one maps to
    negative infinity and inputs below negative one map to NaN.
    """
    x = _fp32_scalar(x, "tla.log1p", _semantic)
    return _generator.call_JitFunction(_log1p_fp32, [x], {})


@tl.builtin
def sigmoid(x, _semantic=None, _generator=None):
    """Return the logistic sigmoid of a float32 scalar or Python float."""
    x = _fp32_scalar(x, "tla.sigmoid", _semantic)
    return _generator.call_JitFunction(_sigmoid_fp32, [x], {})


@tl.builtin
def maximum(a, b, propagate_nan: tl.constexpr = tl.PropagateNan.NONE, _semantic=None):
    """Scalar-only ``tl.maximum`` with standard type promotion.

    Float16 and bfloat16 inputs are unsupported. ``propagate_nan`` defaults to
    ``tl.PropagateNan.NONE``; use ``ALL`` to propagate either input NaN.
    With ``NONE``, exactly one NaN input is temporarily unsupported on Ascend.
    """
    a = _check_scalar(a, "tla.maximum a")
    b = _check_scalar(b, "tla.maximum b")
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, tl.tensor) and value.dtype == tl.float16:
            raise TypeError(f"tla.maximum {name} does not support float16")
    return tl.maximum(a, b, propagate_nan=propagate_nan, _semantic=_semantic)


@tl.builtin
def minimum(a, b, propagate_nan: tl.constexpr = tl.PropagateNan.NONE, _semantic=None):
    """Scalar-only ``tl.minimum`` with standard type promotion.

    Float16 and bfloat16 inputs are unsupported. ``propagate_nan`` defaults to
    ``tl.PropagateNan.NONE``.
    With ``NONE``, exactly one NaN input is temporarily unsupported on Ascend.
    """
    a = _check_scalar(a, "tla.minimum a")
    b = _check_scalar(b, "tla.minimum b")
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, tl.tensor) and value.dtype == tl.float16:
            raise TypeError(f"tla.minimum {name} does not support float16")
    return tl.minimum(a, b, propagate_nan=propagate_nan, _semantic=_semantic)


@tl.builtin
def cdiv(x, div, _semantic=None, _generator=None):
    """Scalar-only ``tl.cdiv``, with identical promotion and division rules.

    :param x: Python or runtime scalar dividend.
    :param div: Python or runtime scalar divisor. Must be nonzero.
    :return: The scalar result of standard Triton ``tl.cdiv``.

    Delegates to Triton's ``(x + (div - 1)) // div`` implementation.
    Integer overflow and signed division follow standard Triton semantics;
    this does not provide overflow-safe ceiling division. Bfloat16 inputs
    are unsupported.
    """
    from triton.language import cdiv as triton_cdiv

    x = _check_scalar(x, "tla.cdiv x")
    div = _check_scalar(div, "tla.cdiv div")
    # tl.cdiv is a JIT helper; the existing frontend pipeline inlines the call.
    return _generator.call_JitFunction(triton_cdiv, [x, div], {})


@tl.builtin
def cast(input, dtype, fp_downcast_rounding=None, bitcast=False, _semantic=None):
    """Scalar-only ``tl.cast``, with identical conversion and rounding rules.

    Accepts Python and runtime scalars. ``bitcast=True``
    reinterprets equal-width scalar bit patterns. Buffers and SIMD registers
    require their explicit memory and SIMD interfaces. Bfloat16 source or
    destination types are unsupported.
    """
    input = _check_scalar(input, "tla.cast input")
    dtype = tl._unwrap_if_constexpr(dtype)
    if dtype == tl.bfloat16:
        raise TypeError("tla.cast does not support bfloat16")
    return tl.cast(input, dtype, fp_downcast_rounding=fp_downcast_rounding, bitcast=bitcast, _semantic=_semantic)


@tl.builtin
def max_value(dtype, _semantic=None):
    """Return the largest finite scalar value representable by ``dtype``.

    :param dtype: Compile-time scalar dtype, for example ``tla.float32``.
        Supports bool, signed and unsigned 8/16/32/64-bit integers,
        float16, bfloat16, float32 and float64.
    :type dtype: triton.language.dtype
    :return: A scalar constant with the requested dtype. Floating-point
        results are finite; integer results are the type's upper bound.

    Examples inside ``@tla.jit`` kernels::

        limit = tla.max_value(tla.float32)
        upper = tla.max_value(tla.int32)

    This operation creates no buffer and performs no runtime conversion.
    Pointer, block, buffer, SIMD vector and float8 types are unsupported.
    """
    dtype = tl._unwrap_if_constexpr(dtype)
    if not isinstance(dtype, tl.dtype) or dtype.is_ptr() or dtype.is_block():
        raise TypeError("tla.max_value dtype must be a compile-time scalar dtype")
    if dtype.is_int():
        value = dtype.get_int_max_value()
    elif dtype.is_standard_floating():
        value = math.ldexp(2.0 - 2.0**(-dtype.fp_mantissa_width), dtype.exponent_bias)
    else:
        raise ValueError(f"tla.max_value does not support dtype {dtype}")
    return _semantic.scalar_constant(value, dtype)


@tl.builtin
def infinity(dtype, _semantic=None):
    """Return a positive infinity scalar with exactly the requested dtype.

    ``dtype`` must be a compile-time ``float16``, ``bfloat16`` or ``float32``
    dtype. The result is a scalar, with no buffer allocation or SIMD
    broadcast. Use ``-tla.infinity(dtype)`` for negative infinity and
    ``tla.simd.vdup`` to broadcast it explicitly.
    """
    dtype = tl._unwrap_if_constexpr(dtype)
    if not isinstance(dtype, tl.dtype):
        raise TypeError("tla.infinity expects a compile-time dtype")
    if dtype not in (tl.float16, tl.bfloat16, tl.float32):
        raise ValueError(f"tla.infinity supports float16, bfloat16 and float32; got {dtype}")
    return tl.full((), float("inf"), dtype, _semantic=_semantic)


@tl.builtin
def clamp(x, min, max, propagate_nan: tl.constexpr = tl.PropagateNan.NONE, _semantic=None):
    """Scalar-only ``tl.clamp`` with standard type promotion.

    Clamp ``x`` to [min, max]. Accepts Python and runtime numeric scalars;
    float16 and bfloat16 inputs are unsupported. Triton requires a floating-point
    computation. ``propagate_nan`` defaults to ``tl.PropagateNan.NONE`` and applies
    only to ``x``. With ``NONE``, NaN ``x`` is temporarily unsupported on Ascend.
    Behavior is undefined if either bound is NaN or ``min > max``.
    No buffer allocation or SIMD broadcast is performed. Compute the scalar
    outside SIMD VF and broadcast it explicitly inside VF.
    """
    x = _check_scalar(x, "tla.clamp x")
    min = _check_scalar(min, "tla.clamp min")
    max = _check_scalar(max, "tla.clamp max")
    for name, value in (("x", x), ("min", min), ("max", max)):
        if isinstance(value, tl.tensor) and value.dtype == tl.float16:
            raise TypeError(f"tla.clamp {name} does not support float16")
    return tl.clamp(x, min, max, propagate_nan=propagate_nan, _semantic=_semantic)
