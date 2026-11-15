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
from triton.language.core import builtin, int32, tensor

from . import simd
from .core import buffer
from .scope import simd as simd_scope

# Up to four 256-bin segments always use the aligned int32 dhistv2 path.
_MAX_BINS = 1024


def _zero_bins(dst, semantic):
    builder = semantic.builder
    scope = builder.create_scope_op(simd_scope.mlir_attrs(builder), [])
    insertion_point = builder.get_insertion_point()
    body = builder.get_scope_entry_block(scope)
    builder.set_insertion_point_to_start(body)

    # The SIMT template accumulates into dst. Zero every bin, including the tail.
    zero = simd.vdup(0, int32, _semantic=semantic)
    bins = builder.get_int32(dst.shape[0])
    loop = builder.create_for_op(builder.get_int32(0), bins, builder.get_int32(zero.shape[0]), [])
    builder.set_insertion_point_to_start(loop.get_body(0))
    index = loop.get_induction_var()
    remaining = tensor(builder.create_sub(bins, index), int32)
    mask = simd.tail_mask(remaining, _semantic=semantic)
    simd.vsts(dst, [tensor(index, int32)], zero, mask=mask, _semantic=semantic)

    builder.set_insertion_point_to_end(body)
    builder.scope_return([])
    builder.restore_insertion_point(insertion_point)


@builtin
def histogram(src, dst, _semantic=None):
    """Count integer values in ``src`` and overwrite the bins in ``dst``.

    The number of bins is ``dst.shape[0]`` (1 to 1024). Values outside
    ``[0, dst.shape[0])`` are ignored. Both buffers must be one-dimensional,
    contiguous ``int32`` UB buffers at offset zero with disjoint storage.

    Call this operation in a ``tla.scope.vector()`` region on Ascend 950.
    """
    for name, operand in (("src", src), ("dst", dst)):
        if not isinstance(operand, buffer):
            raise TypeError(f"tla.histogram {name} must be a tla.buffer")
        if operand.space != "ub" or operand.dtype != int32 or len(operand.shape) != 1:
            raise ValueError(f"tla.histogram {name} must be a 1D int32 UB buffer")
        if operand.strides != (1, ) or operand.offset != 0:
            raise ValueError(f"tla.histogram {name} must be contiguous at offset 0")
    if src is dst:
        raise ValueError("tla.histogram requires disjoint src and dst buffers")

    if not 1 <= dst.shape[0] <= _MAX_BINS:
        raise ValueError(f"tla.histogram supports 1 to {_MAX_BINS} bins")

    builder = _semantic.builder
    name = "__builtin_histogram"
    attrs = {
        "symbol": builder.get_string_attr(name),
        "gm_addr_args_indices": builder.parse_attr("array<i32>"),
        "hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<VECTOR>"),
        "hivm.vf_mode": builder.parse_attr("#hivm.vf_mode<SIMT>"),
        "hivm.pipe": builder.parse_attr("#hivm.pipe<PIPE_V>"),
    }
    _zero_bins(dst, _semantic)
    builder.create_custom_op(name, attrs, [src.handle, builder.get_int64(dst.shape[0])], [dst.handle], [])
