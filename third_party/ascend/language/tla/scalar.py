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
"""Scalar reads of local-memory values used by control flow and DMA addresses."""
from triton.language.core import builtin, int32, int64, tensor, _unwrap_if_constexpr
from .core import buffer, buffer_access, _scalar_index


@builtin
def scalar_load(src, indices=None, _semantic=None):
    """Read one int32 UB element into a scalar through ``memref.load``.

    Use this in scalar control flow inside ``tla.scope.vector()``. It is not
    a SIMD register load. ``indices`` contains one scalar integer per source
    dimension (Python integers or signed int32/int64 tensors); a single integer is accepted for a rank-one source. The
    ``scalar_load(src[i, ...])`` spelling is equivalent.

    Source offsets and strides are preserved. Static out-of-bounds indices
    are rejected; runtime indices must be in bounds. This is an ordinary
    memory read, not a barrier: the producer must have completed before it.
    The compiler can infer its UB memory dependency with automatic sync;
    explicitly managed pipelines remain responsible for that dependency.
    """
    if isinstance(src, buffer_access):
        if indices is not None:
            raise TypeError("scalar_load indices must not be specified twice")
        src, indices = src.buffer, src.indices
    if not isinstance(src, buffer) or src.space != "ub" or src.dtype != int32:
        raise TypeError("scalar_load expects an int32 UB buffer")
    indices = _unwrap_if_constexpr(indices)
    if isinstance(indices, (int, tensor)):
        indices = (indices, )
    if indices is not None:
        try:
            indices = tuple(indices)
        except TypeError as error:
            raise TypeError("indices must be scalar integers or a sequence of indices") from error
    if indices is None or len(indices) != len(src.shape):
        raise ValueError("scalar_load needs one index per source dimension")
    handles = []
    for index, extent in zip(indices, src.shape):
        index = _scalar_index(index)
        if isinstance(index, int) and not 0 <= index < extent:
            raise ValueError("scalar_load index is outside the source buffer")
        value = _semantic.to_tensor(index)
        # Python integers in [2**31, 2**32) otherwise become unsigned i32.
        if isinstance(index, int):
            value = _semantic.cast(value, int64)
        handles.append(value.handle)
    handle = _semantic.builder.create_tla_scalar_load(src.handle, handles)
    return tensor(handle, src.dtype)
