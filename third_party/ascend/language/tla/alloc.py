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
"""Explicit on-chip TLA memref allocation."""
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer, buffer_type
from .aux_ops import annotate_buffer_versions


@builtin
def alloc(shape, dtype, space="ub", versions=1, _semantic=None):
    space = _unwrap_if_constexpr(space)
    versions = _unwrap_if_constexpr(versions)
    if space == "gm":
        raise ValueError("GM buffers must be supplied as kernel arguments")
    if type(versions) is not int or versions < 1:
        raise ValueError("buffer versions must be a positive integer")
    ty = buffer_type(shape, dtype, space)
    buf = buffer(_semantic.builder.create_tla_alloc(ty.to_ir(_semantic.builder)), ty)
    annotate_buffer_versions(buf, versions, _semantic=_semantic)
    return buf


@builtin
def alloc_buffer(shape, dtype, space="ub", versions=1, _semantic=None):
    return alloc(shape, dtype, space, versions, _semantic=_semantic)
