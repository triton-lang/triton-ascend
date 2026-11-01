"""Bounded MMAD dimensions for structurally masked causal tails.

Uses the existing HIVM real_m/real_k/real_n operands. Physical buffers keep
their full declared shapes; the operation computes only the valid submatrix.
"""
import triton.language as tl
from triton.language.core import builtin, _unwrap_if_constexpr
from triton.language.extra.tla.cube import _logical_shape
from tla_native_ext import _call, vexpdif, vmuls
from triton.language.extra.tla.core import buffer, buffer_type, _shape, _row_major_strides
from math import prod


@builtin
def prefix_view(src, shape, _semantic=None):
    """Alias a bounded contiguous prefix of a zero-offset allocation.

    Unlike a rank-preserving subview, this changes the row pitch. Only
    root/reshape identity buffers are accepted, so offset zero refers to
    the same allocation origin and every output element stays in bounds.
    """
    sizes = _shape(shape)
    if not isinstance(src, buffer) or src.offset != 0:
        raise ValueError('prefix_view requires a zero-offset buffer')
    if src.strides != _row_major_strides(src.shape) or prod(sizes) > prod(src.shape):
        raise ValueError('prefix_view must fit inside a contiguous allocation')
    strides = _row_major_strides(sizes)
    ty = buffer_type(sizes, src.dtype, src.space, strides, offset=None)
    handle = _semantic.builder.create_tla_reinterpret_cast(src.handle,
                                                           _semantic.to_tensor(0).handle, list(sizes), list(strides),
                                                           ty.to_ir(_semantic.builder))
    return buffer(handle, ty)


@builtin
def matmul(a, b, acc, transpose_b=False, clear_accum=False, transpose_a=False, real_m=None, real_k=None, real_n=None,
           _semantic=None):
    ta, tb, clear = (bool(_unwrap_if_constexpr(x)) for x in (transpose_a, transpose_b, clear_accum))
    m, k = _logical_shape(a.shape)
    k2, n = _logical_shape(b.shape)
    if ta:
        m, k = k, m
    if tb:
        k2, n = n, k2
    if (a.space, b.space, acc.space) != ('l1', 'l1', 'l0c'):
        raise ValueError('MMAD requires L1/L1/L0C')
    if a.dtype != b.dtype or k != k2:
        raise ValueError('MMAD logical shapes or dtypes do not agree')
    actual = []
    for bound, value in zip((m, k, n), (real_m, real_k, real_n)):
        value = _unwrap_if_constexpr(value)
        value = bound if value is None else value
        if not isinstance(value, int) or not 0 < value <= bound or value % 16:
            raise ValueError('Actual MMAD extents must be positive aligned constants within the physical shape')
        actual.append(value)
    rm, rk, rn = actual
    if _logical_shape(acc.shape) != (rm, rn):
        raise ValueError('L0C must have the actual M/N shape used by MMAD and Fixpipe')
    attrs = ', '.join(x for x, enabled in [('a_transpose', ta), ('b_transpose', tb)] if enabled)
    attrs = ' {' + attrs + '}' if attrs else ''
    at, bt, ct = (str(x.handle.get_type()) for x in (a, b, acc))
    body = f'''%init = arith.constant {str(clear).lower()}
%m = arith.constant {rm} : index
%k = arith.constant {rk} : index
%n = arith.constant {rn} : index
hivm.hir.mmadL1{attrs} ins(%a0, %a1, %init, %m, %k, %n : {at}, {bt}, i1, index, index, index) outs(%a2 : {ct})
tt.return'''
    _call(_semantic, [a.handle, b.handle, acc.handle], body)
