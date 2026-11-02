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
"""Frontend rejection and forwarding contracts; numerical tests are separate."""
import importlib
from pathlib import Path
import sys
import types

import pytest
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[2] / "language" / "tla"
if "memory_tla_src" not in sys.modules:
    package = types.ModuleType("memory_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["memory_tla_src"] = package
core = importlib.import_module("memory_tla_src.core")
view = importlib.import_module("memory_tla_src.view")
scalar = importlib.import_module("memory_tla_src.scalar")
dma = importlib.import_module("memory_tla_src.dma")


class Recorder:

    def __init__(self):
        self.calls = []

    def get_float_ty(self):
        return "fp32"

    def get_bf16_ty(self):
        return "bf16"

    def create_tla_buffer_type(self, shape, dtype, space, strides, offset):
        return tuple(shape), dtype, space, tuple(strides), offset

    def create_tla_byte_view(self, *args):
        self.calls.append(("byte_view", args))
        return object()

    def create_tla_scalar_load(self, *args):
        self.calls.append(("scalar_load", args))
        return object()

    def create_tla_load_strided(self, *args):
        self.calls.append(("strided_load", args))


class Semantic:

    def __init__(self):
        self.builder = Recorder()
        self.casts = []

    def cast(self, value, dtype):
        assert dtype == tl.int64
        self.casts.append((value.handle, dtype))
        return value

    def to_tensor(self, value):
        return value if isinstance(value, tl.tensor) else types.SimpleNamespace(handle=("constant", value))


def buf(shape=(1024, ), dtype=tl.int8, space="ub", strides=None, offset=0):
    return core.buffer(object(), core.buffer_type(shape, dtype, space, strides, offset))


@pytest.mark.parametrize("offset", [0, 128])
@pytest.mark.parametrize("dtype", [tl.float32, tl.bfloat16])
def test_byte_view_forwards_shared_storage(offset, dtype):
    sem = Semantic()
    source = buf()
    result = view.byte_view(source, (8, 16), dtype, byte_offset=offset, _semantic=sem)
    op, args = sem.builder.calls[0]
    assert op == "byte_view" and args[:2] == (source.handle, offset)
    assert result.shape == (8, 16) and result.strides == (16, 1)
    assert result.dtype == dtype and result.space == source.space and result.offset == 0


@pytest.mark.parametrize("source", [
    buf(dtype=tl.float32),
    buf(space="gm"),
    buf((8, 128)),
    buf(strides=(2, )),
    buf(offset=16),
    buf(offset=None),
])
def test_byte_view_rejects_non_base_storage(source):
    with pytest.raises(TypeError, match="zero-offset contiguous"):
        view.byte_view(source, (128, ), tl.float32, _semantic=Semantic())


@pytest.mark.parametrize("offset", [-4, 2, True, 16.0, tl.tensor(object(), tl.int32)])
def test_byte_view_offset_contract(offset):
    with pytest.raises(ValueError, match="byte_offset"):
        view.byte_view(buf(), (128, ), tl.float32, byte_offset=offset, _semantic=Semantic())


@pytest.mark.parametrize("shape,offset", [((257, ), 0), ((128, ), 516)])
def test_byte_view_rejects_capacity_overflow(shape, offset):
    with pytest.raises(ValueError, match="exceeds"):
        view.byte_view(buf(), shape, tl.float32, byte_offset=offset, _semantic=Semantic())


def test_byte_view_rejects_sub_byte_dtype():
    with pytest.raises(TypeError, match="byte-addressable"):
        view.byte_view(buf(), (128, ), tl.int1, _semantic=Semantic())


def test_scalar_load_preserves_source_view_and_runtime_index():
    sem = Semantic()
    source = buf((16, 16), tl.int32, strides=(32, 2), offset=128)
    index = tl.tensor(object(), tl.int32)
    result = scalar.scalar_load(source, (index, 3), _semantic=sem)
    assert result.dtype == tl.int32
    assert sem.builder.calls == [("scalar_load", (source.handle, [index.handle, ("constant", 3)]))]


@pytest.mark.parametrize("index", [-1, 64, (0, 1)])
def test_scalar_load_rejects_bad_rank_or_bounds(index):
    with pytest.raises(ValueError):
        scalar.scalar_load(buf((64, ), tl.int32), index, _semantic=Semantic())


@pytest.mark.parametrize(
    "index",
    [True, (1.5, ), (tl.tensor(object(), tl.float32), ), (tl.tensor(object(), tl.block_type(tl.int32, [2])), )])
def test_scalar_load_rejects_non_scalar_integer_indices(index):
    with pytest.raises(TypeError, match="scalar integers"):
        scalar.scalar_load(buf((64, ), tl.int32), index, _semantic=Semantic())


@pytest.mark.parametrize("source", [buf((64, ), tl.float32), buf((64, ), tl.int32, "gm")])
def test_scalar_load_rejects_unsupported_source(source):
    with pytest.raises(TypeError, match="int32 UB"):
        scalar.scalar_load(source, 0, _semantic=Semantic())


def test_scalar_load_accepts_element_access():
    sem = Semantic()
    source = buf((64, ), tl.int32)
    scalar.scalar_load(source[7], _semantic=sem)
    assert sem.builder.calls == [("scalar_load", (source.handle, [("constant", 7)]))]
    with pytest.raises(TypeError, match="twice"):
        scalar.scalar_load(source[7], 0, _semantic=sem)


def test_strided_load_forwards_runtime_values_and_aliased_handles():
    sem = Semantic()
    source = buf((4096, ), tl.bfloat16, "gm", offset=None)
    destination = buf((2, 64), tl.bfloat16, strides=(96, 1), offset=None)
    values = [tl.tensor(object(), tl.int32) for _ in range(3)]
    dma.load_gm_to_ub_strided(source, destination, *values, _semantic=sem)
    assert sem.builder.calls == [("strided_load", (source.handle, destination.handle, *(x.handle for x in values)))]


def test_strided_load_default_copies_all_rows():
    sem = Semantic()
    source = buf((4096, ), tl.bfloat16, "gm")
    destination = buf((4, 64), tl.bfloat16)
    dma.load_gm_to_ub_strided(source, destination, 128, 256, _semantic=sem)
    assert sem.builder.calls[0][1] == (source.handle, destination.handle, 128, 256, 4)


@pytest.mark.parametrize("offset,stride,rows", [(-1, 64, 2), (0, 63, 2), (0, 64, 0), (0, 64, 3), (4000, 128, 2)])
def test_strided_load_checks_static_bounds(offset, stride, rows):
    sem = Semantic()
    with pytest.raises(ValueError):
        dma.load_gm_to_ub_strided(buf((4096, ), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16), offset, stride, rows,
                                  _semantic=sem)
    assert not sem.builder.calls


@pytest.mark.parametrize("source,destination", [
    (buf((4096, ), tl.bfloat16), buf((2, 64), tl.bfloat16)),
    (buf((4096, ), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16, "l1")),
    (buf((4096, ), tl.float16, "gm"), buf((2, 64), tl.bfloat16)),
    (buf((64, 64), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16)),
    (buf((4096, ), tl.bfloat16, "gm", strides=(2, )), buf((2, 64), tl.bfloat16)),
    (buf((4096, ), tl.bfloat16, "gm"), buf((128, ), tl.bfloat16)),
    (buf((4096, ), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16, strides=(128, 2))),
    (buf((4096, ), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16, strides=(32, 1))),
])
def test_strided_load_rejects_unsupported_routes_and_layouts(source, destination):
    with pytest.raises((TypeError, ValueError)):
        dma.load_gm_to_ub_strided(source, destination, 0, 64, _semantic=Semantic())


@pytest.mark.parametrize("dtype", [tl.int1, tl.int8, tl.int16, tl.uint8, tl.uint16, tl.uint32, tl.uint64])
def test_scalar_load_rejects_unsigned_and_narrow_indices(dtype):
    with pytest.raises(TypeError, match="signed int32 or int64"):
        scalar.scalar_load(buf((256, ), tl.int32), tl.tensor(object(), dtype), _semantic=Semantic())


@pytest.mark.parametrize("dtype", [tl.int1, tl.int8, tl.int16, tl.uint8, tl.uint16, tl.uint32, tl.uint64])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_strided_load_rejects_unsigned_and_narrow_indices(dtype, position):
    values = [0, 64, 2]
    values[position] = tl.tensor(object(), dtype)
    with pytest.raises(TypeError, match="signed int32 or int64"):
        dma.load_gm_to_ub_strided(buf((4096, ), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16), *values,
                                  _semantic=Semantic())


@pytest.mark.parametrize("source_size,offset", [(32, tl.tensor(object(), tl.int32)), (4096, 4090)])
def test_strided_load_checks_static_first_row_with_dynamic_dimensions(source_size, offset):
    dynamic = tl.tensor(object(), tl.int64)
    with pytest.raises(ValueError, match="exceeds the source"):
        dma.load_gm_to_ub_strided(buf((source_size, ), tl.bfloat16, "gm"), buf((2, 64), tl.bfloat16), offset, dynamic,
                                  dynamic, _semantic=Semantic())


@pytest.mark.parametrize("dtype", [tl.int32, tl.int64])
def test_scalar_load_accepts_signed_indices(dtype):
    sem = Semantic()
    index = tl.tensor(object(), dtype)
    source = buf((256, ), tl.int32)
    scalar.scalar_load(source, index, _semantic=sem)
    assert sem.builder.calls[0][1] == (source.handle, [index.handle])


def test_scalar_load_converts_large_constant_to_signed_int64():
    sem = Semantic()
    scalar.scalar_load(buf((1 << 33, ), tl.int32), 1 << 31, _semantic=sem)
    assert sem.builder.calls[0][1][1] == [("constant", 1 << 31)]
    assert sem.casts == [(("constant", 1 << 31), tl.int64)]


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 63])
def test_index_constants_must_fit_int64(value):
    with pytest.raises(ValueError, match="fit signed int64"):
        core._scalar_index(value)
