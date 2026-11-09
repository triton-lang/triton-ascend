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
"""Host contracts for the ``tla.cube.mmad_mx`` microscaling interface.

These tests load the in-tree frontend and record builder calls; they do not
substitute device ops or exercise the backend.
"""
import importlib
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("triton")
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[3] / "language" / "tla"
if "mmad_mx_tla_src" not in sys.modules:
    package = types.ModuleType("mmad_mx_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["mmad_mx_tla_src"] = package
cube = importlib.import_module("mmad_mx_tla_src.cube")
core = importlib.import_module("mmad_mx_tla_src.core")
dma = importlib.import_module("mmad_mx_tla_src.dma")


class Recorder:

    def __init__(self):
        self.calls = []

    def create_hivm_mmadmx(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def create_hivm_load_mx_scale(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class Semantic:

    def __init__(self):
        self.builder = Recorder()


def buf(shape, space="l1", dtype=None):
    dtype = dtype or (tl.float32 if space == "l0c" else tl.float8e5)
    return core.buffer(object(), core.buffer_type(shape, dtype, space))


def nz(rows, cols):
    return (cols // 16, rows // 16, 16, 16)


def zz(rows, k32):
    return (rows // 16, (k32 + 1) // 2, 16, 2)


def mx(a_shape=(32, 64), b_shape=(64, 128), acc_shape=(32, 128), **kwargs):
    """Build operands with scale shapes derived from the logical extents."""
    m, k = a_shape
    n = b_shape[1]
    a = buf(a_shape)
    b = buf(b_shape)
    sa = buf(zz(m, k // 32), dtype=tl.uint8)
    sb = buf(zz(n, k // 32), dtype=tl.uint8)
    acc = buf(acc_shape, "l0c")
    return a, b, sa, sb, acc, kwargs


def test_fp8_formats_are_inferred_and_emit_no_native_format():
    a, b, sa, sb, acc, _ = mx()
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, acc, clear_accum=True, _semantic=sem)
    assert sem.builder.calls == [((a.handle, b.handle, sa.handle, sb.handle, acc.handle),
                                  dict(init_condition=True, real_m=32, real_k=64, real_n=128, lhs_format=None,
                                       rhs_format=None, transpose_a=False, transpose_b=False, per_channel_bias=None))]


@pytest.mark.parametrize("fmt,code", [(cube.MxFormat.E5M2, 1), (cube.MxFormat.E4M3, 2), (cube.MxFormat.E2M1, 3)])
def test_integer_operands_require_and_forward_native_format(fmt, code):
    k = 64
    a = buf((32, k // 2 if fmt is cube.MxFormat.E2M1 else k), dtype=tl.int8)
    b = buf((k // 2 if fmt is cube.MxFormat.E2M1 else k, 128), dtype=tl.int8)
    sa = buf(zz(32, k // 32), dtype=tl.uint8)
    sb = buf(zz(128, k // 32), dtype=tl.uint8)
    acc = buf((32, 128), "l0c")
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, acc, format_a=fmt, format_b=fmt, _semantic=sem)
    args = sem.builder.calls[0][1]
    assert (args["lhs_format"], args["rhs_format"]) == (code, code)
    # real_k is the stored K extent; the template expands it for FP4.
    assert args["real_k"] == (k // 2 if fmt is cube.MxFormat.E2M1 else k)


def test_integer_operand_without_format_is_rejected():
    sem = Semantic()
    a = buf((32, 64), dtype=tl.int8)
    b = buf((64, 128), dtype=tl.int8)
    with pytest.raises(ValueError, match="format_a"):
        cube.mmad_mx(a, b, buf(zz(32, 2), dtype=tl.uint8), buf(zz(128, 2), dtype=tl.uint8), buf((32, 128), "l0c"),
                     _semantic=sem)
    assert not sem.builder.calls


def test_explicit_fp8_format_must_match_dtype():
    a, b, sa, sb, acc, _ = mx()
    with pytest.raises(ValueError, match="does not match"):
        cube.mmad_mx(a, b, sa, sb, acc, format_a=cube.MxFormat.E4M3, _semantic=Semantic())


def test_mixed_fp8_fp4_is_rejected():
    a = buf((32, 64))
    b = buf((32, 128), dtype=tl.int8)
    sa = buf(zz(32, 2), dtype=tl.uint8)
    sb = buf(zz(128, 2), dtype=tl.uint8)
    with pytest.raises(ValueError, match="mixed FP8/FP4"):
        cube.mmad_mx(a, b, sa, sb, buf((32, 128), "l0c"), format_b=cube.MxFormat.E2M1, _semantic=Semantic())


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_transpose_swaps_logical_extents(ta, tb):
    # Logical product is (M=32, K=64) x (K=64, N=128).
    a = buf((64, 32) if ta else (32, 64))
    b = buf((128, 64) if tb else (64, 128))
    sa = buf(zz(32, 2), dtype=tl.uint8)
    sb = buf(zz(128, 2), dtype=tl.uint8)
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, buf((32, 128), "l0c"), transpose_a=ta, transpose_b=tb, _semantic=sem)
    args = sem.builder.calls[0][1]
    assert (args["real_m"], args["real_k"], args["real_n"]) == (32, 64, 128)
    assert (args["transpose_a"], args["transpose_b"]) == (ta, tb)


def test_4d_fractal_operands_and_scale_buffers_are_accepted():
    a = buf(nz(32, 64))
    b = buf(nz(64, 128))
    sa = buf(zz(32, 2), dtype=tl.uint8)
    sb = buf(zz(128, 2), dtype=tl.uint8)
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, buf(nz(32, 128), "l0c"), _semantic=sem)
    assert sem.builder.calls[0][1]["real_m"] == 32


def test_2d_scale_buffers_are_accepted():
    a, b, _, _, acc, _ = mx()
    sa = buf((32, 2), dtype=tl.uint8)
    sb = buf((128, 2), dtype=tl.uint8)
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, acc, _semantic=sem)
    assert len(sem.builder.calls) == 1


def test_bias_is_forwarded_as_l1_operand():
    a, b, sa, sb, acc, _ = mx()
    bias = buf((1, 128), dtype=tl.float32)
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, acc, bias=bias, _semantic=sem)
    assert sem.builder.calls[0][1]["per_channel_bias"] == bias.handle


@pytest.mark.parametrize("bias", [buf((1, 128), dtype=tl.float16), buf((1, 128), "ub", dtype=tl.float32)])
def test_invalid_bias_is_rejected(bias):
    a, b, sa, sb, acc, _ = mx()
    with pytest.raises(ValueError, match="bias"):
        cube.mmad_mx(a, b, sa, sb, acc, bias=bias, _semantic=Semantic())


@pytest.mark.parametrize("scale", [buf((32, 2), dtype=tl.float32), buf((32, 2), "ub", dtype=tl.uint8)])
def test_invalid_scale_operand_is_rejected(scale):
    a, b, _, sb, acc, _ = mx()
    with pytest.raises(ValueError, match="scale_a"):
        cube.mmad_mx(a, b, scale, sb, acc, _semantic=Semantic())


def test_scale_shape_must_match_operand_extents():
    a, b, _, sb, acc, _ = mx()
    with pytest.raises(ValueError, match="scale_a"):
        cube.mmad_mx(a, b, buf(zz(48, 2), dtype=tl.uint8), sb, acc, _semantic=Semantic())


def test_odd_k_group_count_pads_only_the_4d_fractal_scale():
    # K=96 -> 3 K-groups; the zZ tile pads the odd count to 4.
    a = buf((32, 96))
    b = buf((96, 128))
    sa = buf(zz(32, 3), dtype=tl.uint8)
    sb = buf(zz(128, 3), dtype=tl.uint8)
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, buf((32, 128), "l0c"), _semantic=sem)
    assert len(sem.builder.calls) == 1
    # A 2-D ND scale must carry the exact logical K//32 extent.
    with pytest.raises(ValueError, match="scale_a"):
        cube.mmad_mx(a, b, buf((32, 4), dtype=tl.uint8), buf((128, 3), dtype=tl.uint8), buf((32, 128), "l0c"),
                     _semantic=Semantic())


def test_reduction_must_be_a_multiple_of_the_mx_group():
    a = buf((32, 48))
    b = buf((48, 128))
    with pytest.raises(ValueError, match="multiple of 32"):
        cube.mmad_mx(a, b, buf(zz(32, 2), dtype=tl.uint8), buf(zz(128, 2), dtype=tl.uint8), buf((32, 128), "l0c"),
                     _semantic=Semantic())


def test_reduction_mismatch_is_rejected():
    with pytest.raises(ValueError, match="reduction"):
        cube.mmad_mx(buf((32, 64)), buf((96, 128)), buf(zz(32, 2), dtype=tl.uint8), buf(zz(128, 3), dtype=tl.uint8),
                     buf((32, 128), "l0c"), _semantic=Semantic())


@pytest.mark.parametrize("axis,bound", [("real_m", 32), ("real_k", 64), ("real_n", 128)])
def test_invalid_real_extents_are_rejected(axis, bound):
    a, b, sa, sb, acc, _ = mx()
    with pytest.raises(ValueError, match=axis):
        cube.mmad_mx(a, b, sa, sb, acc, **{axis: bound + 16}, _semantic=Semantic())


def test_scale_shape_uses_full_tile_not_real_extents():
    # real_m/real_n restrict the accumulator only; scales cover the whole L1 tile.
    a, b, sa, sb, _, _ = mx()
    sem = Semantic()
    cube.mmad_mx(a, b, sa, sb, buf((16, 64), "l0c"), real_m=16, real_n=64, _semantic=sem)
    args = sem.builder.calls[0][1]
    assert (args["real_m"], args["real_k"], args["real_n"]) == (16, 64, 64)


def test_accumulator_must_match_actual_extents():
    a, b, sa, sb, _, _ = mx()
    with pytest.raises(ValueError, match="acc must"):
        cube.mmad_mx(a, b, sa, sb, buf((32, 128), "l0c"), real_n=64, _semantic=Semantic())


def test_operand_spaces_are_checked():
    with pytest.raises(ValueError, match="L1"):
        cube.mmad_mx(buf((32, 64), "ub"), buf((64, 128)), buf(zz(32, 2), dtype=tl.uint8),
                     buf(zz(128, 2), dtype=tl.uint8), buf((32, 128), "l0c"), _semantic=Semantic())
    with pytest.raises(ValueError, match="L0C"):
        cube.mmad_mx(buf((32, 64)), buf((64, 128)), buf(zz(32, 2), dtype=tl.uint8), buf(zz(128, 2), dtype=tl.uint8),
                     buf((32, 128), "ub"), _semantic=Semantic())


def test_scale_dma_emits_load_mx_scale():
    src = buf((32, 2), space="gm", dtype=tl.uint8)
    dst = buf(zz(32, 2), dtype=tl.uint8)
    sem = Semantic()
    dma.load_gm_to_l1_mx_scale(src, dst, _semantic=sem)
    assert sem.builder.calls == [((src.handle, dst.handle), dict(is_transposed=False))]


def test_scale_dma_rejects_configuration_and_wrong_spaces():
    src = buf((32, 2), space="gm", dtype=tl.uint8)
    dst = buf(zz(32, 2), dtype=tl.uint8)
    with pytest.raises(TypeError, match="no configuration"):
        dma.load_gm_to_l1_mx_scale(src, dst, params=object(), _semantic=Semantic())
    with pytest.raises(ValueError, match="gm"):
        dma.load_gm_to_l1_mx_scale(buf((32, 2), dtype=tl.uint8), dst, _semantic=Semantic())
