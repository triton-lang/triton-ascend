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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""``tla.where`` -- eager scalar / SIMD-register selection.

``where(condition, x, y)`` returns ``x`` where ``condition`` is true and ``y``
elsewhere.  Unlike TileLang's lazy ``T.if_then_else`` control-flow helper, both
arms are ordinary values that are always evaluated *before* the call, so
``where`` cannot guard an invalid load or skip an expensive operand.  Use a
masked load/store (``tla.simd.vlds`` mask / ``tl.load(..., mask=...)``) for
that.

Two disjoint paths are supported:

- scalar: ``condition`` is a Python/constexpr ``bool`` or a scalar ``int1``
  ``tl.tensor``; ``x``/``y`` are scalar numbers, constexprs, or scalar
  ``tl.tensor`` values.  Delegates to :func:`triton.language.where`, which does
  the usual type promotion and emits ``arith.select``.  Integer conditions and
  block tensors are rejected instead of being truth-cast.
- SIMD: when either arm is a :class:`tla.vector`, both arms must be vectors of
  the identical type (no scalar broadcast).  A vector ``int1`` condition is
  lowered with :func:`tla.simd.vsel` (``ave.hir.vsel``); a Python/constexpr
  ``bool`` folds to the chosen arm without emitting IR.  A runtime scalar bool
  cannot select between vectors -- build a vector mask with matching lanes
  (e.g. :func:`tla.simd.vcmp`) instead.  A vector condition with scalar arms is
  likewise rejected.
"""
import numbers

import triton.language as tl
from triton.language.core import builtin, tensor, _unwrap_if_constexpr

from . import simd
from .core import buffer, buffer_access, vector

__all__ = ["where"]


def _condition(condition):
    """Classify and validate a ``where`` condition (no builder side effects)."""
    condition = _unwrap_if_constexpr(condition)
    if isinstance(condition, bool):
        return "const", condition
    if isinstance(condition, vector):
        return "vector", condition
    if isinstance(condition, tensor):
        if condition.type.is_block():
            raise TypeError("where condition must not be a block tensor; use a "
                            "tla.vector<lanes x int1> SIMD mask instead")
        if condition.dtype != tl.int1:
            raise TypeError(f"where condition scalar tensor must have int1 dtype, got {condition.dtype}")
        return "runtime", condition
    raise TypeError("where condition must be a Python bool, constexpr bool, scalar int1 "
                    f"tl.tensor, or tla.vector<int1> mask, got {type(condition).__name__}")


def _scalar_operand(value, name):
    """Validate one arm of the scalar path; return the unwrapped value."""
    value = _unwrap_if_constexpr(value)
    if isinstance(value, (vector, buffer, buffer_access)):
        raise TypeError(f"where {name} must be a scalar value, got {type(value).__name__}")
    if isinstance(value, tensor):
        if value.type.is_block():
            raise TypeError(f"where {name} must be a scalar value, got a block tensor")
        return value
    if isinstance(value, numbers.Number):
        return value
    raise TypeError(f"where {name} must be a scalar number, constexpr, or scalar tl.tensor, "
                    f"got {type(value).__name__}")


@builtin
def where(condition, x, y, _semantic=None):
    """Select ``x`` or ``y`` elementwise.

    Both arms are eager (always evaluated); see the module docstring for the
    scalar-vs-SIMD restrictions.

    :param condition: Python/constexpr ``bool``, scalar ``int1`` ``tl.tensor``,
        or a ``tla.vector<lanes x int1>`` mask for vector arms.
    :param x: scalar value or ``tla.vector`` selected where true.
    :param y: scalar value or ``tla.vector`` selected where false.
    """
    kind, cond = _condition(condition)
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    x_is_vector, y_is_vector = isinstance(x, vector), isinstance(y, vector)

    if x_is_vector or y_is_vector:
        if not (x_is_vector and y_is_vector):
            raise TypeError("where does not mix scalar and tla.vector arms; "
                            "both x and y must be tla.vector")
        if x.type != y.type:
            raise ValueError(f"where requires same-typed vector arms, got {x.type} and {y.type}")
        if kind == "vector":
            return simd.vsel(cond, x, y, _semantic=_semantic)
        if kind == "const":
            return x if cond else y
        raise TypeError("where cannot select between tla.vector arms with a runtime scalar "
                        "bool condition; use a tla.vector<lanes x int1> mask (e.g. "
                        "tla.simd.vcmp(...)) with the same lanes as the values")

    if kind == "vector":
        raise TypeError("where vector condition requires tla.vector arms; scalar arms cannot "
                        "be selected with a SIMD mask")

    x = _scalar_operand(x, "x")
    y = _scalar_operand(y, "y")
    # Reuse Triton's own promotion + arith.select instead of hand-rolling casts.
    return tl.where(cond, x, y, _semantic=_semantic)
