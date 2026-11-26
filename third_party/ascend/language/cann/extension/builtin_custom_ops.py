# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

import math

import triton.language.core as tl
from triton.language.core import _unwrap_if_constexpr
from .custom_op import register_custom_op
from .core import CORE, PIPE, MODE, AddressSpace, DataLayout, ascend_address_space
from ..utils import _deprecated
from ._utils import _is_int_like_elem, _assert_int_like_tuple


@register_custom_op
class _sparse_gather_load_to_L1buffer:
    """Gather index-selected regions of a paged cache from GM straight into L1.

    Sparse attention picks a handful of small KV regions per query. Loading them
    one region at a time gives the hardware many short, discontinuous GM
    accesses, and staging them in UB adds a vector pass plus a vec->cube handoff.
    This op writes the regions into L1 in NZ layout, ready for L0, and coalesces
    neighbouring region ids into a single transfer -- the selector emits each
    row's regions sorted, and a region's tokens are contiguous in the physical
    token space, so a run of consecutive ids is one contiguous GM range.

    Arguments:
    - base: pointer type, base of the gathered tensor (in GM). Fold any
      head/group offset into the pointer; token rows are ``stride_token``
      elements apart and ``dim_size`` elements long.
    - index: pointer type, base of this row's packed region metadata (in GM).
      Entry ``s`` is ``region_offset | (region_valid << region_valid_shift)``,
      or negative for a padding slot. Entries must be sorted by region_offset.
    - block_r: int, slots in the destination tile.
    - valid_region_count: scalar int, only the first
      ``min(block_r, valid_region_count)`` entries of ``index`` are read.
    - region_size: int, tokens per region.
    - dim_size: int, elements per token.
    - stride_token: int, element stride between consecutive tokens of ``base``.
    - region_valid_shift: int, bit position of the valid-token count.
    - region_id_mask: int, mask selecting the region offset.
    - l2_cache_mode: int, MTE2 cache-control field, passed through untouched.
    - out: the destination buffer, which must live in L1
      (``bl.alloc(..., al.ascend_address_space.L1)``).

    Padding slots and the unused tail of a partial region are zero filled, so
    every row of ``out`` is written on every call.

    A5 (dav-c310) only: GM->L1 is a cube-core MTE2 path there, and no other
    supported target has a GM->L1 instruction.
    """
    name = '__builtin_sparse_gather_load_to_L1buffer'
    core = CORE.CUBE
    pipe = PIPE.PIPE_MTE2
    # The template lands the tile in zN, whatever shape the destination was
    # declared with. Saying so lets a consumer's ND -> Fractal convert_layout
    # fold away; without it the compiler believes the ND declaration and
    # synthesises a vector-core transpose, which drags the tile back out of L1.
    data_layout = {'out': DataLayout.Fractal}
    # Likewise for where it lands: hivm-infer-hivm-mem-scope reads this off
    # arg_attrs rather than guessing from the pipe, which would say GM.
    address_space = {'out': AddressSpace.L1}

    def __init__(self, base, index, block_r, valid_region_count, region_size, dim_size, stride_token,
                 region_valid_shift, region_id_mask, l2_cache_mode=0, out=None):
        assert base.type.is_ptr() or base.dtype.is_ptr(), f"base should be a pointer, but got {base.type}"
        assert index.type.is_ptr() or index.dtype.is_ptr(), f"index should be a pointer, but got {index.type}"
        for name_, value in (("block_r", block_r), ("region_size", region_size), ("dim_size", dim_size),
                             ("stride_token", stride_token), ("region_valid_shift", region_valid_shift),
                             ("region_id_mask", region_id_mask), ("l2_cache_mode", l2_cache_mode)):
            assert _is_int_like_elem(value), f"{name_} should be an integer"
        assert _is_int_like_elem(valid_region_count), "valid_region_count should be an integer"

        assert out is not None, "out is required"
        assert out.dtype == base.dtype.element_ty, "out should have same dtype as base"
        space = getattr(out, "space", None)
        l1 = ascend_address_space.L1
        assert space is not None and getattr(space, "real_address_space", None) == l1.real_address_space, \
            "out must be a buffer allocated in L1, e.g. bl.alloc(dtype, shape, al.ascend_address_space.L1)"

        # The destination is an NZ tile and each token row occupies exactly one
        # 32B block of every C0 column, so a row that is not a whole number of
        # 32B units would not tile.
        elem_bytes = base.dtype.element_ty.primitive_bitwidth // 8
        if isinstance(dim_size, int):
            assert (dim_size * elem_bytes) % 32 == 0, \
                f"dim_size * sizeof(dtype) must be 32B aligned, got {dim_size} * {elem_bytes}"

        # The template fills block_r * region_size token rows without bounds
        # checking, so a destination that holds fewer elements than that
        # corrupts whatever follows it in L1. Only the capacity is fixed: the
        # tile is written in zN, so the destination is equally well declared as
        # [block_r * region_size, dim_size] or as its 4D fractal.
        shape = [_unwrap_if_constexpr(d) for d in out.shape]
        if all(isinstance(v, int) for v in (block_r, region_size, dim_size)) and \
                all(isinstance(d, int) for d in shape):
            needed = block_r * region_size * dim_size
            got = math.prod(shape)
            assert got >= needed, \
                f"out holds {got} elements, needs block_r * region_size * dim_size = {needed}"

        # stride_token is an element count that reaches the hardware as a 16-bit
        # field; the template falls back to a per-row transfer beyond that.
        self.arg_type['stride_token'] = tl.int64

        # The compiler derives a builtin's library call name from its operand
        # types, so a symbol is normally redundant. It is not here: the HIVM
        # dialect linked into the frontend need not be the one bishengir-compile
        # uses, and one that predates this builtin verifies the op as an
        # ordinary custom op, which requires a symbol. Must stay in step with
        # getSparseGatherLoadToL1BufferLibraryCallName() and with
        # REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER in the template.
        self.symbol = (f"sparse_gather_load_to_L1buffer_{base.dtype.element_ty.cname}"
                       f"_{index.dtype.element_ty.cname}")


@_deprecated(fn_name="triton.language.extra.cann.extension.custom('__builtin_index_select', ...)")
@register_custom_op
class _index_select:
    """
    This operation gathers values from the src GM tensor into the out UB tensor
    at positions with offsets specified by the index UB tensor along the specified
    dimension using a SIMT template. This operation supports 2D–5D.

    Arguments:
    - src: pointer type, the source tensor pointer (in GM)
    - index: tensor, a tensor to gather (in UB)
    - dim: int, the dimension to gather along
    - bound: int, the upper boundary for index
    - end_offset: tuple of int, the end offsets of each dimension for index tensor
    - start_offset: tuple of int, the start offsets of each dimension for src tensor
    - src_stride: tuple of int, the stride of each dimension of src tensor
    - other(Optional): scalar value, the default value when index is out of boundary (in UB)
    - out: the output tensor (in UB)

    Note:
    - Supported source ranks: 2D ~ 5D.
    - Supported index ranks: 1D or 2D.
    - `dim` must be valid (0 <= dim < source ranks).

    Reference formula:
    Index select operation for different tensor ranks:
    1. 2D index gather (0 <= dim <= 1)
        1.1 dim = 0, index_rank = 1, src_rank = 2, out_rank = 2
            index_shape = (Ai,)
            end_offset = (Ai_end, B_end)
            start_offset = (0, B_begin)
            out[i][0:B_end-B_begin] = src[index[i]][B_begin:B_end]
        1.2 dim = 0, index_rank = 2, src_rank = 2, out_rank = 3
            index_shape = (Ai, Aj)
            end_offset = (Ai_end, Aj_end, B_end)
            start_offset = (0, B_begin)
            out[i][j][0:B_end-B_begin] = src[index[i][j]][B_begin:B_end]
    2. 3D index gather (0 <= dim <= 2)
        2.1 dim = 0, index_rank = 2, src_rank = 3, out_rank = 4
            index_shape = (Ai, Aj)
            end_offset = (Ai_end, Aj_end, B_end, C_end)
            start_offset = (0, B_begin, C_begin)
            out[i][j][0:B_end-B_begin][0:C_end-C_begin] = src[index[i][j]][B_begin:B_end][C_begin:C_end]
        and so on.
    """
    name = '__builtin_index_select'
    core = CORE.VECTOR
    pipe = PIPE.PIPE_V
    mode = MODE.SIMT

    def __init__(self, src, index, dim, bound: tl.int64, end_offset, start_offset, src_stride, other=None, out=None):
        assert src.type.is_ptr() or src.dtype.is_ptr(), f"src should be a pointer, but got {src.type}"
        assert index.dtype.is_int(), "index should be integer tensor"
        src_rank = len(src_stride)
        idx_rank = len(index.shape)
        assert 2 <= src_rank <= 5, f"src rank should in [2, 5], but got {src_rank}"
        assert 1 <= idx_rank <= 2, f"index rank should in [1, 2], but got {idx_rank}"
        assert _is_int_like_elem(dim), "dim should be an integer"
        assert _is_int_like_elem(bound), "bound should be an integer"
        assert 0 <= dim < src_rank, f"dim should in [0, {src_rank - 1}], but got {dim}"
        assert len(start_offset) == len(src_stride), "start_offset and src_stride should have same size"
        assert len(end_offset) == idx_rank + len(
            start_offset) - 1, "len(end_offset) should be equal to index rank + len(start_offset) - 1"

        _assert_int_like_tuple("end_offset", end_offset)
        _assert_int_like_tuple("start_offset", start_offset)
        _assert_int_like_tuple("src_stride", src_stride)

        assert out, "out is required"
        assert out.dtype == src.dtype.element_ty, "out should have same dtype as src"

        # use index type for end_offset, start_offset and src_stride.
        self.arg_type['end_offset'] = index.dtype
        self.arg_type['start_offset'] = index.dtype
        self.arg_type['src_stride'] = index.dtype
        self.extra_attr = f"src_stride_len={len(src_stride)}"
