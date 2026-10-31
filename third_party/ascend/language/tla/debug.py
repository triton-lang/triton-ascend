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
"""tla.device_print / tla.device_assert: device diagnostics.

``device_print`` emits ``hivm.hir.debug`` directly instead of ``tt.print``
(``tt.print`` only takes TT_Type operands and the pure-buffer interface set has
no scalar-extraction op): a UB buffer/memref or a ``buf[i0, i1, ...]`` element
range (lowered to a 0-d subview) can be printed at the buffer stage.  Only
allowed outside ``tla.scope`` and on UB-backed values.

``device_assert`` mirrors ``tl.device_assert``: emitted only under
``options.debug`` (else no-op).  The condition must be a scalar i1 — the
pure-buffer interface set cannot derive one from vector registers, so vector
conditions are rejected to keep the ``tensor<``-free IR contract.
"""

from triton.language.core import _unwrap_if_constexpr
from triton.language.core import builtin

from .core import buffer
from .core import buffer_access
from .core import is_ub
from .core import space_of

__all__ = ["device_assert", "device_print"]


def _in_scope(generator):
    """True when codegen is inside a ``scope.scope`` body.

    The bindings expose ``block -> region`` and ``op -> region`` but not
    ``region -> parent op``, so the owning scope is found by a module scan.
    """
    region_id = generator.builder.get_insertion_block().get_parent().id()
    found = [False]

    def visit(op):
        # scope.scope is single-region (SingleBlock in ScopeOps.td).
        if op.get_name() == "scope.scope" and op.get_region(0).id() == region_id:
            found[0] = True

    generator.module.walk(visit)
    return found[0]


def _print_arg(semantic, builder, arg):
    """Map a tla value to an ``hivm.hir.debug`` operand.

    ``buffer_access`` -> 0-d memref subview (only the addressed element),
    UB ``buffer`` -> the memref itself, scalar -> tensor handle.  Non-UB
    spaces (L1/GM) are rejected: the debug chain prints UB-backed values only.
    """
    if isinstance(arg, buffer_access):
        src = arg.buffer
        if not is_ub(src):
            raise RuntimeError(f"tla.device_print: only UB values are supported, got {space_of(src)}")
        offsets = [semantic.to_tensor(i).handle for i in arg.indices]
        rank = len(src.shape)
        return builder.create_tla_subview(src.handle, offsets, [1] * rank, [1] * rank)
    if isinstance(arg, buffer):
        if not is_ub(arg):
            raise RuntimeError(f"tla.device_print: only UB values are supported, got {space_of(arg)}")
        return arg.handle
    return semantic.to_tensor(arg).handle


@builtin
def device_print(prefix, *args, hex=False, _semantic=None, _generator=None):
    """Print values on device; only outside ``tla.scope`` and on UB values.

    Each value emits one ``hivm.hir.debug`` op; ``buf[i0, i1, ...]`` element
    ranges lower to a 0-d subview so only that element is dumped.
    """
    if _generator is not None and _in_scope(_generator):
        raise RuntimeError("tla.device_print: not allowed inside tla.scope")
    prefix = _unwrap_if_constexpr(prefix)
    if not isinstance(prefix, str):
        raise TypeError(f"device_print prefix must be a string, got {prefix!r}")
    builder = _semantic.builder
    for arg in args:
        builder.create_hivm_debug("print", prefix, hex, _print_arg(_semantic, builder, arg))


@builtin
def device_assert(cond, msg="", _semantic=None):
    """Assert ``cond`` (scalar i1) at runtime; terminate with ``msg`` on failure.

    Vector conditions are rejected — the pure-buffer interface set cannot
    produce a data-dependent scalar from AVE vector registers.
    """
    msg = _unwrap_if_constexpr(msg)
    if not isinstance(msg, str):
        raise TypeError(f"device_assert message must be a string, got {msg!r}")
    cond_t = _semantic.to_tensor(cond)
    if len(cond_t.shape) != 0:
        raise ValueError("tla.device_assert requires a scalar i1 condition; "
                         f"got shape {tuple(cond_t.shape)} "
                         "(the pure-buffer interface set cannot produce a data-dependent "
                         "scalar from AVE vector registers)")
    return _semantic.device_assert(cond_t, msg, None)
