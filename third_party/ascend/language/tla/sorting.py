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
"""Explicit UB sorting of scores and caller-provided indices."""

from triton.language.core import builtin, float32, uint32

from .core import buffer


def _ub_vector(value, dtype):
    if (not isinstance(value, buffer) or value.space != "ub" or len(value.shape) != 1 or value.strides != (1, )
            or value.dtype != dtype):
        raise TypeError("expected a contiguous one-dimensional UB buffer")
    return value.shape[0]


def _check_scores(scores, indices, dst):
    width = _ub_vector(scores, float32)
    if not 32 <= width <= 4096 or width % 32:
        raise ValueError("score width must be a multiple of 32 in [32, 4096]")
    if _ub_vector(indices, uint32) != width or _ub_vector(dst, uint32) != 2 * width:
        raise ValueError("expected one index and two destination words per score")
    return width


@builtin
def sort_pairs(scores, indices, dst, scratch, _semantic=None):
    """Sort descending into interleaved FP32 score bits and uint32 indices.

    Use disjoint 32-byte-aligned UB buffers. The score count must be a
    power of two in [32, 4096], including any padding supplied by the caller.
    Both dst and scratch have twice the score count in uint32 words.
    Equal scores are permitted and have unspecified relative order.
    """
    width = _check_scores(scores, indices, dst)
    if width & (width - 1) or _ub_vector(scratch, uint32) != 2 * width:
        raise ValueError("expected power-of-two width and a two-word-per-score workspace")
    _semantic.builder.create_tla_sort_pairs(scores.handle, indices.handle, dst.handle, scratch.handle)
