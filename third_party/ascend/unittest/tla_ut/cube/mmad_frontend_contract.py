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
"""Host contracts for MMAD physical shapes, extents, and legacy arguments.

Device precision is tested separately in cube/mmad.py. This module loads the
in-tree frontend and records builder calls; it does not substitute device ops.
"""
import importlib
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("triton")
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[3] / "language" / "tla"
if "mmad_tla_src" not in sys.modules:
    package = types.ModuleType("mmad_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["mmad_tla_src"] = package
cube = importlib.import_module("mmad_tla_src.cube")
core = importlib.import_module("mmad_tla_src.core")


class Recorder:

    def __init__(self):
        self.calls = []

    def create_hivm_mmadl1(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class Semantic:

    def __init__(self):
        self.builder = Recorder()


def buf(shape, space="l1", dtype=None):
    dtype = dtype or (tl.float32 if space == "l0c" else tl.bfloat16)
    return core.buffer(object(), core.buffer_type(shape, dtype, space))


def nz(rows, cols):
    return (cols // 16, rows // 16, 16, 16)


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("a_nz,b_nz,c_nz", [(False, False, False), (True, True, True), (True, False, True),
                                            (False, True, False)])
def test_transpose_uses_physical_shapes(ta, tb, a_nz, b_nz, c_nz):
    a_shape = (64, 32) if ta else (32, 64)
    b_shape = (128, 64) if tb else (64, 128)
    a = buf(nz(*a_shape) if a_nz else a_shape)
    b = buf(nz(*b_shape) if b_nz else b_shape)
    acc = buf(nz(32, 128) if c_nz else (32, 128), "l0c")
    sem = Semantic()
    cube.mmad(a, b, acc, transpose_a=ta, transpose_b=tb, clear_accum=True, _semantic=sem)
    assert sem.builder.calls == [((a.handle, b.handle, acc.handle),
                                  dict(init_condition=True, real_m=32, real_k=64, real_n=128, transpose_a=ta,
                                       transpose_b=tb, unit_flag_ctrl=None, unit_flag_group_id=None))]


def test_extents_are_checked_after_transpose_and_use_compact_l0c():
    sem = Semantic()
    cube.mmad(buf((64, 128)), buf((256, 64)), buf((32, 48), "l0c"), transpose_a=True, transpose_b=True, real_m=32,
              real_k=16, real_n=48, _semantic=sem)
    args = sem.builder.calls[0][1]
    assert (args["real_m"], args["real_k"], args["real_n"]) == (32, 16, 48)
    assert args["transpose_a"] and args["transpose_b"]


def test_positional_arguments_and_unit_flag_order():
    sem = Semantic()
    cube.mmad(buf((64, 32)), buf((64, 128)), buf((32, 128), "l0c"), True, False, True, 2, unit_flag_group_id=3,
              _semantic=sem)
    args = sem.builder.calls[0][1]
    assert args["init_condition"] and args["transpose_a"] and not args["transpose_b"]
    assert (args["unit_flag_ctrl"], args["unit_flag_group_id"]) == (2, 3)


@pytest.mark.parametrize("axis,bound", [("real_m", 32), ("real_k", 64), ("real_n", 128)])
@pytest.mark.parametrize("invalid", [0, -16, 17, True, 16.0, "16", "too_large"])
def test_invalid_extents_are_rejected_before_emission(axis, bound, invalid):
    sem = Semantic()
    extent = bound + 16 if invalid == "too_large" else invalid
    with pytest.raises(ValueError, match=axis):
        cube.mmad(buf((32, 64)), buf((64, 128)), buf((32, 128), "l0c"), **{axis: extent}, _semantic=sem)
    assert not sem.builder.calls


def test_runtime_extent_is_rejected():
    sem = Semantic()
    with pytest.raises(ValueError, match="real_k"):
        cube.mmad(buf((32, 64)), buf((64, 128)), buf((32, 128), "l0c"), real_k=tl.tensor(object(), tl.int32),
                  _semantic=sem)
    assert not sem.builder.calls


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_real_k_does_not_hide_incompatible_input_reductions(ta, tb):
    a = buf((64, 32) if ta else (32, 64))
    b = buf((128, 96) if tb else (96, 128))
    with pytest.raises(ValueError, match="reduction"):
        cube.mmad(a, b, buf((32, 128), "l0c"), transpose_a=ta, transpose_b=tb, real_k=16, _semantic=Semantic())


@pytest.mark.parametrize("real", [{"real_m": 16}, {"real_n": 64}])
def test_oversized_accumulator_is_rejected(real):
    with pytest.raises(ValueError, match="acc must"):
        cube.mmad(buf((32, 64)), buf((64, 128)), buf((32, 128), "l0c"), **real, _semantic=Semantic())


@pytest.mark.parametrize("a,b,acc,message", [
    (object(), buf((64, 128)), buf((32, 128), "l0c"), "expects tla.buffer"),
    (buf((32, 64), "ub"), buf((64, 128)), buf((32, 128), "l0c"), "L1 buffers"),
    (buf((32, 64)), buf((64, 128), "ub"), buf((32, 128), "l0c"), "L1 buffers"),
    (buf((32, 64)), buf((64, 128)), buf((32, 128), "ub"), "L0C buffer"),
    (buf((32, 64), dtype=tl.float16), buf((64, 128)), buf((32, 128), "l0c"), "same element type"),
    (buf((2, 16, 64)), buf((64, 128)), buf((32, 128), "l0c"), "2-D ND or 4-D zN"),
])
def test_invalid_operand_contracts(a, b, acc, message):
    sem = Semantic()
    with pytest.raises((TypeError, ValueError), match=message):
        cube.mmad(a, b, acc, _semantic=sem)
    assert not sem.builder.calls


@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("constexpr", [False, True])
def test_transpose_b_split_reduction_accumulates_into_same_l0c(m, constexpr):
    """DSA QK: initialize with K=512, then add the K=64 RoPE product."""
    sem = Semantic()
    acc = buf(nz(m, 128), "l0c")
    wrap = tl.constexpr if constexpr else lambda value: value
    for k, clear in ((512, True), (64, False)):
        a, b = buf(nz(m, k)), buf(nz(128, k))
        cube.mmad(a, b, acc, transpose_b=wrap(True), clear_accum=wrap(clear), _semantic=sem)
        assert sem.builder.calls[-1] == ((a.handle, b.handle, acc.handle),
                                         dict(init_condition=clear, real_m=m, real_k=k, real_n=128, transpose_a=False,
                                              transpose_b=True, unit_flag_ctrl=None, unit_flag_group_id=None))
    assert len(sem.builder.calls) == 2
