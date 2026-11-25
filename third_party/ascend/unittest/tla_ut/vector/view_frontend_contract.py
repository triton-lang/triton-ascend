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
"""Host-only bounds, layout and offset contracts for TLA buffer views.

Recording builders check Python metadata and rejection behavior. Device and
real-IR tests live separately in vector/subview.py; these mocks do not stand
in for compilation or numerical validation.
"""
import importlib
from pathlib import Path
import sys
import types

import pytest
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[3] / "language" / "tla"
if "view_tla_src" not in sys.modules:
    package = types.ModuleType("view_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["view_tla_src"] = package
view = importlib.import_module("view_tla_src.view")
core = importlib.import_module("view_tla_src.core")


class Handle:

    def __init__(self, value=None):
        self.value = value


class Recorder:

    def __init__(self):
        self.calls = []

    def get_bf16_ty(self):
        return "bf16"

    def create_tla_buffer_type(self, shape, dtype, space, strides, offset):
        return (tuple(shape), dtype, space, tuple(strides), offset)

    def create_tla_subview(self, source, offsets, sizes, strides):
        self.calls.append(("subview", source, offsets, sizes, strides))
        return Handle()

    def create_tla_pointer_view(self, source, linear, result_type, sizes, strides):
        self.calls.append(("pointer_view", source, linear, result_type, sizes, strides))
        return Handle()

    def create_tla_collapse_shape(self, source, result_type, reassociation):
        self.calls.append(("collapse", source, result_type, reassociation))
        return Handle()

    def create_tla_expand_shape(self, source, result_type, reassociation):
        self.calls.append(("expand", source, result_type, reassociation))
        return Handle()


class Semantic:

    def __init__(self):
        self.builder = Recorder()

    def to_tensor(self, value):
        return value if isinstance(value, tl.tensor) else types.SimpleNamespace(handle=Handle(value))


class PtrSemantic(Semantic):
    """Semantic with scalar integer arithmetic for pointer-entry views."""

    def mul(self, lhs, rhs, _):
        return tl.tensor(Handle(("mul", lhs, rhs)), tl.int64)

    def add(self, lhs, rhs, _):
        return tl.tensor(Handle(("add", lhs, rhs)), tl.int64)

    def cast(self, value, dtype):
        if isinstance(value, tl.tensor) and value.dtype == dtype:
            return value
        return tl.tensor(Handle(("cast", value)), dtype)


def buf(shape=(8, 2064), strides=None, offset=0):
    return view.buffer(Handle(), view.buffer_type(shape, tl.bfloat16, "ub", strides, offset))


def test_subview_preserves_pitch():
    source = buf()
    sem = Semantic()
    sliced = view.subview(source, (0, 0), (8, 1040), _semantic=sem)
    assert sliced.shape == (8, 1040) and sliced.strides == (2064, 1)
    assert sliced.dtype == source.dtype and sliced.space == "ub"
    assert sliced.offset is None
    assert sem.builder.calls[0][0] == "subview"


def test_view_rejects_buffer_source():
    with pytest.raises(TypeError, match="tla.subview"):
        view.view(buf(), (8, 1040), _semantic=Semantic())


def test_relative_subview_strides():
    source = buf((16, 16), (32, 1), offset=64)
    sem = Semantic()
    sliced = view.subview(source, (1, 2), (4, 8), (2, 1), _semantic=sem)
    assert sliced.strides == (64, 1)
    call = sem.builder.calls[0]
    assert call[1] is source.handle
    assert [x.value for x in call[2]] == [1, 2]
    assert call[3:] == ([4, 8], [2, 1])


def test_compact_prefix_uses_subview_and_reshape():
    sem = Semantic()
    flat = view.reshape(buf(), (16512, ), _semantic=sem)
    prefix = view.subview(flat, (0, ), (8320, ), _semantic=sem)
    compact = view.reshape(prefix, (8, 1040), _semantic=sem)
    assert compact.shape == (8, 1040) and compact.strides == (1040, 1)
    assert compact.offset is None
    assert [c[0] for c in sem.builder.calls] == ["collapse", "subview", "expand"]
    assert sem.builder.calls[-1][2] == ((8, 1040), "bf16", "ub", (1040, 1), None)


@pytest.mark.parametrize("offset", [0, 128, None])
def test_reshape_preserves_offset_in_every_result_type(offset):
    sem = Semantic()
    result = view.reshape(buf((8, 16), offset=offset), (4, 32), _semantic=sem)
    assert result.offset == offset
    assert [c[0] for c in sem.builder.calls] == ["collapse", "expand"]
    assert all(c[2][-1] == offset for c in sem.builder.calls)


def test_runtime_offset_is_forwarded_to_subview():
    sem = Semantic()
    offset = tl.tensor(Handle(), tl.int32)
    result = view.subview(buf((16512, )), (offset, ), (8320, ), _semantic=sem)
    assert sem.builder.calls[0][2][0] is offset.handle
    assert result.offset is None


def test_runtime_size_and_stride_are_forwarded_to_subview():
    sem = Semantic()
    size = tl.tensor(Handle(), tl.int32)
    stride = tl.tensor(Handle(), tl.int32)
    result = view.subview(buf((16512, )), (0, ), (size, ), (stride, ), _semantic=sem)
    call = sem.builder.calls[0]
    assert call[3][0] is size.handle
    assert call[4][0] is stride.handle
    assert result.shape == (None, ) and result.strides == (None, )


def test_runtime_gm_entry_forwards_shape_and_strides():
    sem = PtrSemantic()
    size = tl.tensor(Handle(), tl.int32)
    stride = tl.tensor(Handle(), tl.int32)
    src = tl.tensor(Handle(), tl.pointer_type(tl.bfloat16))
    result = view.view(src, (128, size), (stride, 1), (0, 0), _semantic=sem)
    call = sem.builder.calls[0]
    assert call[0] == "pointer_view"
    assert call[1] is src.handle
    # Result type: dynamic dims carry the MLIR sentinel, never -1.
    assert call[3][0] == (128, core._DYNAMIC) and call[3][3] == (core._DYNAMIC, 1)
    assert call[4][0] == 128  # static size stays a constant
    assert call[4][1].value[0] == "cast" and call[4][1].value[1] is size
    assert call[5][0].value[0] == "cast" and call[5][0].value[1] is stride
    assert call[5][1] == 1
    assert result.shape == (128, None) and result.strides == (None, 1)


def test_subview_on_gm_view_preserves_space():
    sem = Semantic()
    source = view.buffer(Handle(), view.buffer_type((8, 256), tl.bfloat16, "gm", (256, 1)))
    sliced = view.subview(source, (0, 128), (4, 128), _semantic=sem)
    assert sliced.space == "gm", "subview must keep the GM address space"
    assert sliced.shape == (4, 128) and sliced.strides == (256, 1)
    assert sliced.offset is None
    assert sem.builder.calls[0][1] is source.handle


def test_default_row_major_strides_with_dynamic_extent():
    sem = PtrSemantic()
    size = tl.tensor(Handle(), tl.int32)
    src = tl.tensor(Handle(), tl.pointer_type(tl.bfloat16))
    result = view.view(src, (128, size), _semantic=sem)
    call = sem.builder.calls[0]
    assert call[0] == "pointer_view"
    # Default row-major: dim0 stride is the runtime product 1 * size.
    assert call[5][0].value == ("mul", 1, size)
    assert call[5][1] == 1
    assert result.strides == (None, 1)


def test_dynamic_entries_skip_static_bounds_check():
    sem = Semantic()
    rows = tl.tensor(Handle(), tl.int32)
    # Statically 16 columns would exceed the 16-wide source only together with
    # the offset; the dynamic size defers the bounds decision to runtime.
    result = view.subview(buf((8, 16)), (0, 8), (rows, 16), _semantic=sem)
    assert result.shape == (None, 16)


@pytest.mark.parametrize("offsets,sizes,strides,reason", [
    ((0, ), (8, 16), None, "source rank"),
    ((0, 0), (8, ), None, "source rank"),
    ((0, 0), (8, 16), (1, ), "source rank"),
    ((-1, 0), (8, 16), None, "outside"),
    ((1, 0), (8, 16), None, "outside"),
    ((0, 0), (8, 16), (2, 1), "outside"),
    ((0, 0), (0, 16), None, "positive compile-time"),
    ((0, 0), (8, 16), (1, 0), "positive compile-time"),
])
def test_invalid_slices_are_rejected(offsets, sizes, strides, reason):
    with pytest.raises(ValueError, match=reason):
        view.subview(buf((8, 16)), offsets, sizes, strides, _semantic=Semantic())


@pytest.mark.parametrize(
    "offset", [True, 1.5, tl.tensor(Handle(), tl.float32),
               tl.tensor(Handle(), tl.block_type(tl.int32, [2]))])
def test_offsets_must_be_scalar_integers(offset):
    with pytest.raises(TypeError, match="scalar integers"):
        view.subview(buf((16512, )), (offset, ), (8320, ), _semantic=Semantic())


def test_subview_rejects_non_buffer_source():
    with pytest.raises(TypeError, match="tla.buffer"):
        view.subview(Handle(), (0, ), (16, ), _semantic=Semantic())


@pytest.mark.parametrize("shape,strides", [((8, 1040), (2064, 1)), ((128, ), (2, ))])
def test_reshape_rejects_non_contiguous_slices(shape, strides):
    with pytest.raises(ValueError, match="contiguous row-major"):
        view.reshape(buf(shape, strides, offset=None), (8, 1040) if len(shape) == 2 else (8, 16), _semantic=Semantic())


def test_reshape_rejects_element_count_change():
    with pytest.raises(ValueError, match="element count mismatch"):
        view.reshape(buf((8320, ), offset=None), (8, 2064), _semantic=Semantic())
