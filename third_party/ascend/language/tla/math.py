# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Math helpers for TLA vector kernels.

``compensated_rstd`` is the TLA-side expansion of TileKernels
``tile_kernels/quant/rstd.py`` (``make_rstd_impl``): it re-constructs the
compensated variance ``sum_sq / hidden + eps`` with an error-free TwoSum-style
multiplication-error recovery, then refines a naive 1/sqrt estimate with a
Newton step using the high/low parts of both constants.  The maths is *not*
the plain RMSNorm formula -- it must stay faithful to ``rstd.py`` (see
``ai/operators/quant/norm_forward.md`` §2).

All helpers take an explicit ``_semantic`` because they are inlined into a
Triton kernel and compose the ``@builtin`` simd ops directly.
"""

import struct

from triton.language.core import builtin, _unwrap_if_constexpr

from . import simd as S

__all__ = ["compensated_rstd"]

_FP32_MIN_NORMAL = 2.0**-126


def split_fp32(value: float) -> tuple:
    """Split a Python float into two FP32 values whose sum approximates it."""
    high = struct.unpack('=f', struct.pack('=f', value))[0]
    low = struct.unpack('=f', struct.pack('=f', value - high))[0]
    return high, low


def _naive_rsqrt(value, _semantic):
    """1/sqrt(v) from ``rstd.py`` naive_rsqrt (vdiv + vsqrt)."""
    return S.vdiv(S.vsplat(1.0, value.dtype, _semantic=_semantic), S.vsqrt(value, _semantic=_semantic),
                  _semantic=_semantic)


def _compensated_div_add(a, reciprocal_c0_hi, reciprocal_c0_lo, c1_hi, c1_lo, _semantic):
    """
    High/low parts of ``a / c0 + c1`` with TwoSum error recovery
    (``rstd.py`` compensated_div_add; vmuls/vadds/vsub are expressed through
    splat + mul/add/neg).
    """
    r_c0_hi_v = S.vsplat(reciprocal_c0_hi, a.dtype, _semantic=_semantic)
    product = S.vmul(a, r_c0_hi_v, _semantic=_semantic)
    value = S.vadd(product, S.vsplat(c1_hi, a.dtype, _semantic=_semantic), _semantic=_semantic)

    # Error-free TwoSum for product + c1_hi.
    virtual_c1 = S.vadd(value, S.vneg(product, _semantic=_semantic), _semantic=_semantic)
    product_sum_error = S.vadd(
        product, S.vneg(S.vadd(value, S.vneg(virtual_c1, _semantic=_semantic), _semantic=_semantic),
                        _semantic=_semantic), _semantic=_semantic)
    c1_sum_error = S.vadd(S.vneg(virtual_c1, _semantic=_semantic), S.vsplat(c1_hi, a.dtype, _semantic=_semantic),
                          _semantic=_semantic)
    sum_error = S.vadd(product_sum_error, c1_sum_error, _semantic=_semantic)

    # Recover the multiplication error and add the low parts of both constants.
    product_error = S.vfma(S.vneg(product, _semantic=_semantic), a, r_c0_hi_v, _semantic=_semantic)
    value_error = S.vadd(
        S.vadd(product_error, sum_error, _semantic=_semantic),
        S.vadd(S.vmul(a, S.vsplat(reciprocal_c0_lo, a.dtype, _semantic=_semantic), _semantic=_semantic),
               S.vsplat(c1_lo, a.dtype, _semantic=_semantic), _semantic=_semantic), _semantic=_semantic)
    return value, value_error


def _compensated_rsqrt(value, value_error, estimate, _semantic):
    """
    Refine an rsqrt estimate against the high and low parts of its input
    (``rstd.py`` compensated_rsqrt; the alloc_local ``vmula`` accumulators are
    SSA ``fma`` chains).
    """
    estimate_squared = S.vmul(estimate, estimate, _semantic=_semantic)
    # square_error starts at -estimate_squared, then vmula adds estimate*estimate.
    square_error = S.vfma(S.vneg(estimate_squared, _semantic=_semantic), estimate, estimate, _semantic=_semantic)

    residual = S.vsplat(1.0, value.dtype, _semantic=_semantic)
    negative_value = S.vneg(value, _semantic=_semantic)
    negative_value_error = S.vneg(value_error, _semantic=_semantic)
    residual = S.vfma(residual, negative_value, estimate_squared, _semantic=_semantic)
    residual = S.vfma(residual, negative_value, square_error, _semantic=_semantic)
    residual = S.vfma(residual, negative_value_error, estimate_squared, _semantic=_semantic)
    residual = S.vfma(residual, negative_value_error, square_error, _semantic=_semantic)

    # refined = estimate + (estimate * 0.5) * residual
    refined = S.vfma(estimate, S.vmul(estimate, S.vsplat(0.5, value.dtype, _semantic=_semantic), _semantic=_semantic),
                     residual, _semantic=_semantic)
    return refined


@builtin
def compensated_rstd(square_sum, hidden, eps, _semantic=None):
    """
    Compensated reciprocal standard deviation of ``square_sum / hidden + eps``
    (TLA expansion of ``rstd.py`` make_rstd_impl).

    :param square_sum: f32 vector register holding ``sum_h x[h]**2`` (one value
        broadcast across all lanes, e.g. from an ``ONEPT_B32`` + ``BRC_B32``
        read-back)
    :param hidden: hidden size (positive Python/constexpr scalar); used to
        build the exact ``1/hidden`` high/low split
    :param eps: epsilon (>= 2^-126, ``rstd.py`` bound)
    :return: f32 vector register with the refined rstd replicated per lane
    """
    hidden = float(_unwrap_if_constexpr(hidden))
    eps = float(_unwrap_if_constexpr(eps))
    if eps < _FP32_MIN_NORMAL:
        raise ValueError(f"eps must be at least the smallest normal FP32 value "
                         f"({_FP32_MIN_NORMAL}), got {eps}")
    reciprocal_c0_hi, reciprocal_c0_lo = split_fp32(1.0 / hidden)
    c1_hi, c1_lo = split_fp32(eps)
    variance, variance_error = _compensated_div_add(square_sum, reciprocal_c0_hi, reciprocal_c0_lo, c1_hi, c1_lo,
                                                    _semantic)
    estimate = _naive_rsqrt(variance, _semantic)
    return _compensated_rsqrt(variance, variance_error, estimate, _semantic)
