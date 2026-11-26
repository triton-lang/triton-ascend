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
"""Host-only contract for ``tla.fill``.

The recorder proves that every rejection happens before the builder call and
that exactly one ``create_tla_fill`` is emitted for legal inputs.  It does not
replace IR, compile or device checks; those live in ``fill.py``.
"""
import importlib
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("triton")
from triton.language import core as tl_core
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[3] / "language" / "tla"
if "fill_tla_src" not in sys.modules:
    package = types.ModuleType("fill_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["fill_tla_src"] = package
fill_mod = importlib.import_module("fill_tla_src.fill")
core = importlib.import_module("fill_tla_src.core")


class Recorder:

    def __init__(self):
        self.calls = []

    def create_tla_fill(self, dst, value):
        self.calls.append((dst, value))


class Semantic:

    def __init__(self):
        self.builder = Recorder()
        self.casts = []

    def to_tensor(self, value):
        if isinstance(value, tl_core.tensor):
            return value
        if isinstance(value, bool):
            return tl_core.tensor(None, tl.int1)
        if isinstance(value, int):
            return tl_core.tensor(None, tl.int32)
        if isinstance(value, float):
            return tl_core.tensor(None, tl.float32)
        raise TypeError(f"cannot convert {value!r} to a scalar tensor")

    def cast(self, value, dtype):
        self.casts.append((value.dtype, dtype))
        return tl_core.tensor(None, dtype)


def buf(shape=(16, ), dtype=tl.float32, space="ub", strides=None, offset=0):
    return core.buffer(object(), core.buffer_type(shape, dtype, space, strides, offset))


@pytest.mark.parametrize("dtype", [tl.float32, tl.bfloat16, tl.int32])
def test_supported_dtypes_emit_one_fill(dtype):
    sem = Semantic()
    dst = buf(dtype=dtype)
    assert fill_mod.fill(dst, 1, _semantic=sem) is None
    assert len(sem.builder.calls) == 1 and sem.builder.calls[0][0] is dst.handle
    assert sem.casts == [(tl.int32, dtype)]


def test_value_is_cast_to_the_destination_element_type():
    sem = Semantic()
    fill_mod.fill(buf(dtype=tl.bfloat16), 1.5, _semantic=sem)
    assert sem.casts == [(tl.float32, tl.bfloat16)]

    sem = Semantic()
    fill_mod.fill(buf(dtype=tl.float32), -3, _semantic=sem)
    assert sem.casts == [(tl.int32, tl.float32)]


def test_reshape_result_is_accepted():
    sem = Semantic()
    dst = buf(shape=(4, 4), dtype=tl.float32, strides=(4, 1))
    fill_mod.fill(dst, 0.0, _semantic=sem)
    assert len(sem.builder.calls) == 1


def test_non_buffer_destination_is_rejected():
    sem = Semantic()
    with pytest.raises(TypeError, match="tla.buffer"):
        fill_mod.fill(object(), 1, _semantic=sem)
    assert sem.builder.calls == []


@pytest.mark.parametrize("space", ["gm", "l1", "l0a", "l0b", "l0c", "bt", "fb"])
def test_non_ub_spaces_are_rejected(space):
    sem = Semantic()
    with pytest.raises(ValueError, match="only UB buffers"):
        fill_mod.fill(buf(space=space), 1, _semantic=sem)
    assert sem.builder.calls == []


def test_offset_and_pitch_are_rejected():
    cases = [
        buf(shape=(4, 4), strides=(1, 4)),  # column-major
        buf(shape=(4, 4), strides=(8, 1)),  # padded pitch
        buf(shape=(16, ), offset=3),  # static nonzero offset
        buf(shape=(16, ), offset=None),  # subview/dynamic offset
    ]
    for dst in cases:
        sem = Semantic()
        with pytest.raises(ValueError, match="contiguous row-major buffer at offset 0"):
            fill_mod.fill(dst, 1, _semantic=sem)
        assert sem.builder.calls == []


def test_i1_has_a_dedicated_message():
    sem = Semantic()
    with pytest.raises(ValueError, match="does not support i1"):
        fill_mod.fill(buf(dtype=tl.int1), 0, _semantic=sem)
    assert sem.builder.calls == []


@pytest.mark.parametrize("dtype", [tl.float16, tl.int8, tl.uint32, tl.float64])
def test_unverified_dtypes_are_rejected(dtype):
    sem = Semantic()
    with pytest.raises(ValueError, match="does not support element type"):
        fill_mod.fill(buf(dtype=dtype), 1, _semantic=sem)
    assert sem.builder.calls == []


@pytest.mark.parametrize("value,exc,match", [
    (True, TypeError, "got bool"),
    (2**31, ValueError, "does not fit"),
    (-(2**31) - 1, ValueError, "does not fit"),
    ("1", TypeError, "scalar number"),
    (None, TypeError, "scalar number"),
])
def test_invalid_scalars_are_rejected(value, exc, match):
    sem = Semantic()
    with pytest.raises(exc, match=match):
        fill_mod.fill(buf(dtype=tl.int32), value, _semantic=sem)
    assert sem.builder.calls == []


def test_float_value_for_integer_destination_is_rejected():
    sem = Semantic()
    with pytest.raises(TypeError, match="pass an int instead"):
        fill_mod.fill(buf(dtype=tl.int32), 1.5, _semantic=sem)
    assert sem.builder.calls == []


def test_block_tensor_value_is_rejected():
    sem = Semantic()
    block = tl_core.tensor(None, tl_core.block_type(tl.float32, [4]))
    with pytest.raises(TypeError, match="must be a scalar"):
        fill_mod.fill(buf(dtype=tl.float32), block, _semantic=sem)
    assert sem.builder.calls == []


def test_float_block_value_for_integer_destination_is_rejected():
    sem = Semantic()
    block = tl_core.tensor(None, tl_core.block_type(tl.float32, [4]))
    with pytest.raises(TypeError, match="must be a scalar|floating type"):
        fill_mod.fill(buf(dtype=tl.int32), block, _semantic=sem)
    assert sem.builder.calls == []
