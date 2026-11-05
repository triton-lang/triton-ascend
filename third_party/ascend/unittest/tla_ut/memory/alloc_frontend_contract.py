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
"""Host-only address-space contract for ``tla.alloc``.

The recorder proves that every rejection happens before any builder call.  It
does not stand in for IR, compilation or device validation; those live in
``alloc.py`` and the existing Cube tests.
"""
import importlib
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("triton")
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[3] / "language" / "tla"
if "alloc_tla_src" not in sys.modules:
    package = types.ModuleType("alloc_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["alloc_tla_src"] = package
alloc = importlib.import_module("alloc_tla_src.alloc")
core = importlib.import_module("alloc_tla_src.core")
AddressSpace = core.AddressSpace


class Handle:

    def __init__(self, value=None):
        self.value = value


class Recorder:

    def __init__(self):
        self.calls = []
        self.types = []

    def get_float_ty(self):
        return "f32"

    def get_bf16_ty(self):
        return "bf16"

    def get_int32_ty(self):
        return "i32"

    def get_int1_ty(self):
        return "i1"

    def create_tla_buffer_type(self, shape, dtype, space, strides, offset):
        self.types.append(("ty", tuple(shape), dtype, space, tuple(strides), offset))
        return self.types[-1]

    def create_tla_alloc(self, ty):
        self.calls.append(("alloc", ty))
        return Handle()

    def get_int32_attr(self, value):
        return ("i32", value)

    def create_annotation_mark(self, handle, name, attr):
        self.calls.append(("mark", handle, name, attr))


class Semantic:

    def __init__(self):
        self.builder = Recorder()


@pytest.mark.parametrize("space", ["ub", "l1", "l0c"])
def test_alloc_accepts_backend_supported_spaces(space):
    sem = Semantic()
    buf = alloc.alloc((16, ), tl.float32, space=space, _semantic=sem)
    assert buf.space == space
    assert buf.shape == (16, )
    assert buf.strides == (1, )
    assert buf.offset == 0
    assert [call[0] for call in sem.builder.calls] == ["alloc"]
    assert sem.builder.calls[0][1][3] == space
    assert len(sem.builder.types) == 1 and sem.builder.types[0][3] == space


def test_address_space_enum_values_match_frontend_names():
    assert [space.value for space in AddressSpace] == list(core.SPACES)
    # str mixin keeps legacy string comparisons and calls working.
    assert AddressSpace.UB == "ub" and str(AddressSpace.UB) == "ub"
    assert AddressSpace("l0c") is AddressSpace.L0C


def test_alloc_defaults_to_ub():
    sem = Semantic()
    buf = alloc.alloc((16, ), tl.float32, _semantic=sem)
    assert buf.space == "ub"
    assert sem.builder.types[0][3] == "ub"
    assert sem.builder.calls[0][1][3] == "ub"


@pytest.mark.parametrize("space", [AddressSpace.UB, AddressSpace.L1, AddressSpace.L0C])
def test_alloc_accepts_address_space_enum(space):
    sem = Semantic()
    buf = alloc.alloc((16, ), tl.float32, space=space, _semantic=sem)
    # The enum normalizes to the canonical string inside the buffer type.
    assert buf.space == space.value
    assert type(buf.space) is str
    assert sem.builder.types[0][3] == space.value
    assert sem.builder.calls[0][1][3] == space.value


def test_buffer_type_normalizes_address_space_enum():
    ty = core.buffer_type((4, ), tl.float32, AddressSpace.L1)
    assert ty.space == "l1"
    assert type(ty.space) is str


@pytest.mark.parametrize("space", ["l0a", "l0b", "bt", "fb"])
def test_alloc_rejects_recognized_but_non_allocatable_spaces(space):
    sem = Semantic()
    with pytest.raises(ValueError, match="does not support space"):
        alloc.alloc((16, ), tl.float32, space=space, _semantic=sem)
    assert sem.builder.calls == [] and sem.builder.types == []


@pytest.mark.parametrize("space", [AddressSpace.L0A, AddressSpace.L0B, AddressSpace.BT, AddressSpace.FB])
def test_alloc_rejects_non_allocatable_address_space_enum(space):
    sem = Semantic()
    with pytest.raises(ValueError, match="does not support space"):
        alloc.alloc((16, ), tl.float32, space=space, _semantic=sem)
    assert sem.builder.calls == [] and sem.builder.types == []


def test_alloc_rejects_gm_and_unknown_spaces_before_emitting_ir():
    for space, message in ((AddressSpace.GM, "GM buffers must be supplied"), ("gm", "GM buffers must be supplied"),
                           ("ssbuf", "unsupported buffer space"), ("not-a-space", "unsupported buffer space")):
        sem = Semantic()
        with pytest.raises(ValueError, match=message):
            alloc.alloc((16, ), tl.float32, space=space, _semantic=sem)
        assert sem.builder.calls == [] and sem.builder.types == [], space


def test_alloc_buffer_forwards_the_same_contract():
    sem = Semantic()
    buf = alloc.alloc_buffer((16, ), tl.float32, space="l0c", _semantic=sem)
    assert buf.space == "l0c"
    assert [call[0] for call in sem.builder.calls] == ["alloc"]
    assert len(sem.builder.types) == 1

    bad = Semantic()
    with pytest.raises(ValueError, match="does not support space"):
        alloc.alloc_buffer((16, ), tl.float32, space="l0a", _semantic=bad)
    assert bad.builder.calls == [] and bad.builder.types == []


def test_versions_one_has_no_annotation_and_two_emits_one():
    one = Semantic()
    alloc.alloc((16, ), tl.float32, space="ub", versions=1, _semantic=one)
    assert [call[0] for call in one.builder.calls] == ["alloc"]

    two = Semantic()
    alloc.alloc((16, ), tl.float32, space="ub", versions=2, _semantic=two)
    assert [call[0] for call in two.builder.calls] == ["alloc", "mark"]
    assert two.builder.calls[1][2] == "hivm.multi_buffer"
    assert two.builder.calls[1][3] == ("i32", 2)


@pytest.mark.parametrize("versions", [0, -1, True, False, 1.5, "2", None])
def test_invalid_versions_are_rejected_before_builder(versions):
    sem = Semantic()
    with pytest.raises(ValueError, match="buffer versions"):
        alloc.alloc((16, ), tl.float32, versions=versions, _semantic=sem)
    assert sem.builder.calls == [] and sem.builder.types == []
