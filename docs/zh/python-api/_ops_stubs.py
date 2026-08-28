# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Doc-build stubs for ``tensor`` operator syntax (``x / y``, ``x & y``, ``x >= y``, ...).

These operators are implemented as magic methods on ``triton.language.tensor``
(``__truediv__`` / ``__floordiv__`` / ``__mod__`` / ``__neg__`` / ``__invert__`` /
``__and__`` / ``__or__`` / ``__xor__`` / ``__not__`` / ``__lshift__`` /
``__rshift__`` / comparison methods) and have **no** ``tl.``-prefixed top-level
functions.  To render them in the ``triton.language`` autosummary index (like
``add`` / ``sub`` / ``mul``), conf.py attaches these lightweight stubs to
``triton.language`` at doc-build time.

The stubs are doc-only artifacts — they are never imported by real code.
"""


def _div(x, y):
    """
    Element-wise division of :code:`x` by :code:`y`, the ``/`` operator on tensors.

    The implementation is equivalent to :func:`fdiv`, but without the
    floating-point-only restriction: integer operands are automatically
    converted to floating point before the division.

    * ``int / int`` — both operands are cast to :code:`float32`
    * ``int / float`` or ``float / int`` — the integer operand is cast to the float type
    * ``float / float`` — both operands are unified to the higher-precision float type

    :param x: the dividend
    :type x: Block or scalar number
    :param y: the divisor
    :type y: Block or scalar number
    """


def _floordiv(x, y):
    """
    Element-wise integer division of :code:`x` by :code:`y`, the ``//`` operator on tensors.

    The result truncates toward zero.  Only integer operands are supported.  Signed integers use
    signed division (:code:`sdiv`); unsigned integers use unsigned division
    (:code:`udiv`).  Floating-point operands raise :code:`TypeError`.

    :param x: the dividend
    :type x: Block or scalar number
    :param y: the divisor
    :type y: Block or scalar number
    """


def _mod(x, y):
    """
    Element-wise remainder of :code:`x` divided by :code:`y`, the ``%`` operator on tensors.

    * float % float — floating-point remainder (:code:`frem`)
    * int % int — integer remainder; signed integers use :code:`srem`,
      unsigned integers use :code:`urem`.  Operands with mismatched
      signedness raise :code:`TypeError`.

    :param x: the dividend
    :type x: Block or scalar number
    :param y: the divisor
    :type y: Block or scalar number
    """


def _neg(x):
    """
    Element-wise negation, the unary ``-`` operator on tensors.

    Equivalent to :code:`0 - x`.  Pointer-typed inputs raise :code:`ValueError`.

    :param x: the input
    :type x: Block
    """


def _invert(x):
    """
    Element-wise bitwise NOT, the unary ``~`` operator on tensors.

    Only integer inputs are supported.  Floating-point or pointer inputs
    raise :code:`ValueError`.

    :param x: the input
    :type x: Block
    """


def _bitwise_and(x, y):
    """
    Element-wise bitwise AND of :code:`x` and :code:`y`, the ``&`` operator on tensors.

    Only integer operands are supported.

    :param x: the first input
    :type x: Block
    :param y: the second input
    :type y: Block
    """


def _bitwise_or(x, y):
    """
    Element-wise bitwise OR of :code:`x` and :code:`y`, the ``|`` operator on tensors.

    Only integer operands are supported.

    :param x: the first input
    :type x: Block
    :param y: the second input
    :type y: Block
    """


def _bitwise_xor(x, y):
    """
    Element-wise bitwise XOR of :code:`x` and :code:`y`, the ``^`` operator on tensors.

    Only integer operands are supported.

    :param x: the first input
    :type x: Block
    :param y: the second input
    :type y: Block
    """


def _logical_not(x):
    """
    Element-wise logical NOT, the ``not`` operator on tensors.

    The input is bit-cast to :code:`int1` and then bitwise inverted,
    so the result is always an :code:`int1` (boolean) tensor.

    :param x: the input
    :type x: Block
    """


def _logical_and(x, y):
    """
    Element-wise logical AND, the ``logical_and`` method on tensors.

    Both operands are bit-cast to :code:`int1` and then combined with a
    bitwise AND.  The result is always an :code:`int1` (boolean) tensor.

    :param x: the first input
    :type x: Block
    :param y: the second input
    :type y: Block
    """


def _logical_or(x, y):
    """
    Element-wise logical OR, the ``logical_or`` method on tensors.

    Both operands are bit-cast to :code:`int1` and then combined with a
    bitwise OR.  The result is always an :code:`int1` (boolean) tensor.

    :param x: the first input
    :type x: Block
    :param y: the second input
    :type y: Block
    """


def _lshift(x, y):
    """
    Element-wise left shift of :code:`x` by :code:`y` bits, the ``<<`` operator on tensors.

    Only integer operands are supported.  Bit widths of both operands are
    checked for consistency before the shift (:code:`shl`).

    :param x: the value to shift
    :type x: Block
    :param y: the shift amount
    :type y: Block
    """


def _rshift(x, y):
    """
    Element-wise right shift of :code:`x` by :code:`y` bits, the ``>>`` operator on tensors.

    Only integer operands are supported.  Signed integers use arithmetic
    shift (:code:`ashr`, preserving the sign bit); unsigned integers use
    logical shift (:code:`lshr`).  Bit widths of both operands are checked
    for consistency before the shift.

    :param x: the value to shift
    :type x: Block
    :param y: the shift amount
    :type y: Block
    """


def _gt(x, y):
    """
    Element-wise greater-than comparison, the ``>`` operator on tensors.

    * float — ordered greater-than (:code:`fcmp OGT`)
    * signed int — signed greater-than (:code:`icmp SGT`)
    * unsigned int — unsigned greater-than (:code:`icmp UGT`)

    The result is an :code:`int1` (boolean) tensor with the same shape.

    :param x: the left operand
    :type x: Block
    :param y: the right operand
    :type y: Block
    """


def _ge(x, y):
    """
    Element-wise greater-or-equal comparison, the ``>=`` operator on tensors.

    * float — ordered greater-or-equal (:code:`fcmp OGE`)
    * signed int — signed greater-or-equal (:code:`icmp SGE`)
    * unsigned int — unsigned greater-or-equal (:code:`icmp UGE`)

    The result is an :code:`int1` (boolean) tensor with the same shape.

    :param x: the left operand
    :type x: Block
    :param y: the right operand
    :type y: Block
    """


def _lt(x, y):
    """
    Element-wise less-than comparison, the ``<`` operator on tensors.

    * float — ordered less-than (:code:`fcmp OLT`)
    * signed int — signed less-than (:code:`icmp SLT`)
    * unsigned int — unsigned less-than (:code:`icmp ULT`)

    The result is an :code:`int1` (boolean) tensor with the same shape.

    :param x: the left operand
    :type x: Block
    :param y: the right operand
    :type y: Block
    """


def _le(x, y):
    """
    Element-wise less-or-equal comparison, the ``<=`` operator on tensors.

    * float — ordered less-or-equal (:code:`fcmp OLE`)
    * signed int — signed less-or-equal (:code:`icmp SLE`)
    * unsigned int — unsigned less-or-equal (:code:`icmp ULE`)

    The result is an :code:`int1` (boolean) tensor with the same shape.

    :param x: the left operand
    :type x: Block
    :param y: the right operand
    :type y: Block
    """


def _eq(x, y):
    """
    Element-wise equality comparison, the ``==`` operator on tensors.

    * float — ordered equal (:code:`fcmp OEQ`)
    * int — integer equal (:code:`icmp EQ`)

    The result is an :code:`int1` (boolean) tensor with the same shape.

    :param x: the left operand
    :type x: Block
    :param y: the right operand
    :type y: Block
    """


def _ne(x, y):
    """
    Element-wise not-equal comparison, the ``!=`` operator on tensors.

    * float — unordered not-equal (:code:`fcmp UNE`)
    * int — integer not-equal (:code:`icmp NE`)

    The result is an :code:`int1` (boolean) tensor with the same shape.

    :param x: the left operand
    :type x: Block
    :param y: the right operand
    :type y: Block
    """


# Doc-only stubs: ``and`` / ``or`` / ``not`` are Python keywords, so the
# functions are defined under alias names and attached by conf.py.
def install(triton_language):
    """Attach operator doc stubs to the ``triton.language`` module."""
    for name, fn in [
        ("div", _div),
        ("floordiv", _floordiv),
        ("mod", _mod),
        ("neg", _neg),
        ("invert", _invert),
        ("and", _bitwise_and),
        ("or", _bitwise_or),
        ("xor", _bitwise_xor),
        ("not", _logical_not),
        ("logical_and", _logical_and),
        ("logical_or", _logical_or),
        ("lshift", _lshift),
        ("rshift", _rshift),
        ("gt", _gt),
        ("ge", _ge),
        ("lt", _lt),
        ("le", _le),
        ("eq", _eq),
        ("ne", _ne),
    ]:
        if not hasattr(triton_language, name):
            fn.__name__ = name
            fn.__qualname__ = name
            fn.__module__ = "triton.language"
            setattr(triton_language, name, fn)
