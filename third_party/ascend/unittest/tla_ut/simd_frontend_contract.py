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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Host-side contract tests for the TLA elementwise SIMD ops.

These run on any host with ``triton`` importable -- no AscendNPU-IR build, no
device.  They load ``language/tla/simd.py`` **from the source tree** and drive
it with a recording mock builder, so they check the frontend contract:

- signature/defaults and that internal callers (``math.compensated_rstd``)
  still work with the extended ``vdiv`` signature,
- which AVE op/kind each dtype selects (esp. unsigned vs signed),
- ``mode`` / ``precision`` rejection instead of silent degradation,
- mask dtype/lane validation and the default all-true mask,
- per-op element-type rejection,
- exact i64 scalar materialization (no double round-trip).

A second group statically cross-checks that every ``create_simd_*`` used from
Python exists in ``tla_builder.cc`` *and* is whitelisted in ``builder.py``,
which is the classic way a new binding silently breaks (AttributeError).
"""
import ast
import importlib
import pathlib
import re
import sys
import types

import pytest

pytest.importorskip("triton")

_TLA_DIR = pathlib.Path(__file__).resolve().parents[2] / "language" / "tla"

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


def _load_simd():
    """Import the in-tree ``language/tla/simd.py`` as a throwaway package."""
    if "tla_src" not in sys.modules:
        pkg = types.ModuleType("tla_src")
        pkg.__path__ = [str(_TLA_DIR)]
        sys.modules["tla_src"] = pkg
    return importlib.import_module("tla_src.simd")


simd = _load_simd()

# --------------------------------------------------------------------------
# mock builder
# --------------------------------------------------------------------------


class _Handle:

    def __init__(self, ty):
        self.ty = ty


class _Recorder:
    """Minimal stand-in for the pybind TLABuilder recording op selection."""

    def __init__(self):
        self.calls = []

    def _record(self, name, *args):
        self.calls.append((name, args))
        return _Handle(name)

    def create_simd_pge(self, pattern, lanes):
        return self._record("pge", pattern, lanes)

    # The trailing ``mode`` is the normalized builder spelling forwarded by
    # simd.py ("zeroing" / "merging"); recording it lets the contract tests
    # prove both modes reach the C++ layer instead of being flattened here.
    def create_simd_binary(self, kind, lhs, rhs, mask, mode):
        return self._record("binary", kind, lhs, rhs, mask, mode)

    def create_simd_binary_scalar(self, kind, vec, scalar, mask, mode):
        return self._record("binary_scalar", kind, vec, scalar, mask, mode)

    def create_simd_unary(self, kind, src, mask, mode):
        return self._record("unary", kind, src, mask, mode)

    def create_simd_vdup(self, scalar, elem_tag, lanes, mask, mode):
        return self._record("vdup", scalar, elem_tag, lanes, mask, mode)

    def create_simd_vbroadcast(self, src, mask, low, mode):
        return self._record("vbroadcast", src, mask, low, mode)

    def create_simd_vsplat(self, scalar, elem_tag, lanes):
        return self._record("vsplat", scalar, elem_tag, lanes)

    def create_simd_vsel(self, mask, true_val, false_val):
        return self._record("vsel", mask, true_val, false_val)

    def create_simd_scalar(self, elem_tag, value):
        return self._record("scalar", elem_tag, value)

    def create_simd_scalar_int(self, elem_tag, value):
        return self._record("scalar_int", elem_tag, value)

    def create_simd_vcvt(self, *args):
        return self._record("vcvt", *args)

    def create_simd_pred_pack(self, *args):
        return self._record("pred_pack", *args)

    def create_simd_pred_interleave(self, *args):
        self._record("pred_interleave", *args)
        return _Handle("lo"), _Handle("hi")

    def create_simd_vselr(self, *args):
        return self._record("vselr", *args)

    def create_simd_group_reduce(self, *args):
        return self._record("group_reduce", *args)

    def create_simd_vintlv(self, *args):
        self._record("vintlv", *args)
        return _Handle("low_data"), _Handle("high_data")

    def create_simd_vscatter(self, *args):
        self._record("vscatter", *args)

    def create_simd_vunpack(self, *args):
        return self._record("vunpack", *args)

    def create_simd_pld(self, *args):
        return self._record("pld", *args)

    def create_simd_pst(self, *args):
        return self._record("pst", *args)

    def create_simd_vgather(self, *args):
        return self._record("vgather", *args)

    def create_simd_gather_masks(self, mask):
        self._record("gather_masks", mask)
        return _Handle("low_mask"), _Handle("high_mask")

    def create_simd_vdintlv(self, *args):
        self._record("vdintlv", *args)
        return _Handle("even"), _Handle("odd")

    def create_simd_bitcast(self, *args):
        return self._record("bitcast", *args)


class _Semantic:

    def __init__(self):
        self.builder = _Recorder()

    def to_tensor(self, value):
        return tl.tensor(_Handle(value), tl.int32)


def _vec(lanes, dtype):
    return simd.vector(_Handle("vec"), simd.vector_type(lanes, dtype))


def _mask(lanes, dtype=None):
    return simd.vector(_Handle("mask"), simd.vector_type(lanes, dtype or tl.int1))


def _last(sem):
    assert sem.builder.calls, "no builder call recorded"
    return sem.builder.calls[-1]


def _buffer(dtype=None, shape=(128, ), space="ub", strides=None, offset=0):
    from tla_src.core import buffer_type
    return simd.buffer(_Handle("buffer"), buffer_type(shape, dtype or tl.uint8, space, strides, offset))


@pytest.mark.parametrize("width", (8, 16, 32))
@pytest.mark.parametrize("dist", ("NORM", "US", "DS"))
@pytest.mark.parametrize("access", (False, True))
def test_predicate_memory_dispatch(width, dist, access):
    buf, sem = _buffer(), _Semantic()
    pred = simd.pld(buf[32] if access else buf, None if access else [32], width=tl.constexpr(width), dist=dist,
                    _semantic=sem)
    name, args = _last(sem)
    assert pred.type == simd.vector_type(2048 // width, tl.int1)
    assert name == "pld" and args[0] is buf.handle and args[1][0].ty == 32
    assert args[2:] == (2048 // width, dist, f"pb{width}")
    if access:
        simd.pst(buf[32], pred, _semantic=sem)
    else:
        simd.pst(buf, [32], pred, _semantic=sem)
    name, args = _last(sem)
    assert name == "pst" and args[0] is buf.handle and args[1][0].ty == 32
    assert args[2:] == (pred.handle, f"pb{width}")


@pytest.mark.parametrize("buf", [
    _buffer(space="gm"),
    _buffer(dtype=tl.int1),
    _buffer(dtype=tl.int16),
    _buffer(shape=(16, 16)),
    _buffer(strides=(2, )),
    _buffer(offset=1),
])
@pytest.mark.parametrize("op", ("pld", "pst"))
def test_predicate_memory_rejects_invalid_buffers(buf, op):
    with pytest.raises((TypeError, ValueError)):
        if op == "pld":
            simd.pld(buf[0], _semantic=_Semantic())
        else:
            simd.pst(buf[0], _mask(64), _semantic=_Semantic())


@pytest.mark.parametrize("offset", (-32, 1, 128, 0.5, True))
@pytest.mark.parametrize("op", ("pld", "pst"))
def test_predicate_memory_rejects_invalid_offsets(offset, op):
    with pytest.raises((TypeError, ValueError)):
        if op == "pld":
            simd.pld(_buffer()[offset], _semantic=_Semantic())
        else:
            simd.pst(_buffer()[offset], _mask(64), _semantic=_Semantic())


@pytest.mark.parametrize("width", (1, 64, True, "8"))
def test_pld_rejects_invalid_density(width):
    with pytest.raises(ValueError, match="width"):
        simd.pld(_buffer()[0], width=width, _semantic=_Semantic())


def test_predicate_memory_bounds_and_layout_controls():
    with pytest.raises(ValueError):
        simd.pld(_buffer()[0], dist="NORM_B8", _semantic=_Semantic())
    with pytest.raises(ValueError, match="outside"):
        simd.pld(_buffer(shape=(32, ))[0], dist="DS", _semantic=_Semantic())
    for lanes in (64, 128, 256):
        # Exact-sized stores are valid; one byte less must be rejected.
        simd.pst(_buffer(shape=(lanes // 8, ))[0], _mask(lanes), _semantic=_Semantic())
        with pytest.raises(ValueError, match="outside"):
            simd.pst(_buffer(shape=(lanes // 8 - 1, ))[0], _mask(lanes), _semantic=_Semantic())
    for pred in (_mask(32), _mask(64, tl.int32)):
        with pytest.raises(ValueError, match="predicate"):
            simd.pst(_buffer()[0], pred, _semantic=_Semantic())


@pytest.mark.parametrize("op", (simd.vgather, simd.vld))
@pytest.mark.parametrize("dtype,index_dtype,lanes", [
    (tl.float32, tl.uint32, 64),
    (tl.float16, tl.uint16, 128),
    (tl.bfloat16, tl.uint16, 128),
    (tl.int32, tl.uint32, 64),
])
@pytest.mark.parametrize("mode", (simd.MODE.ZEROING, simd.MODE.MERGING))
def test_indexed_load_dispatch(op, dtype, index_dtype, lanes, mode):
    buf, index, mask, sem = _buffer(dtype), _vec(lanes, index_dtype), _mask(lanes), _Semantic()
    result = op(buf[3], index, mask, mode=mode, _semantic=sem)
    assert result.type == simd.vector_type(lanes, dtype)
    calls = [args for name, args in sem.builder.calls if name == "vgather"]
    assert len(calls) == 1
    assert calls[0][0] is buf.handle and calls[0][1][0].ty == 3
    assert calls[0][2] is index.handle and calls[0][3] is mask.handle
    assert calls[0][-1] == lanes
    assert all(name != "vsel" for name, _ in sem.builder.calls)


@pytest.mark.parametrize("buf,index,mask", [
    (_buffer(tl.int64), _vec(64, tl.uint32), None),
    (_buffer(tl.float32, space="gm"), _vec(64, tl.uint32), None),
    (_buffer(tl.float32, shape=(16, 16)), _vec(64, tl.uint32), None),
    (_buffer(tl.float32, strides=(2, )), _vec(64, tl.uint32), None),
    (_buffer(tl.float32), _vec(32, tl.uint32), None),
    (_buffer(tl.float32), _vec(64, tl.int32), None),
    (_buffer(tl.float16), _vec(128, tl.uint32), None),
    (_buffer(tl.float32), _vec(64, tl.uint32), _mask(128)),
    (_buffer(tl.uint8), _vec(128, tl.uint16), None),
    (_buffer(tl.uint8), (_vec(128, tl.uint16), ), None),
    (_buffer(tl.uint8), (_vec(128, tl.uint16), _vec(64, tl.uint32)), None),
])
def test_indexed_load_rejects_invalid_operands(buf, index, mask):
    with pytest.raises((TypeError, ValueError)):
        simd.vld(buf, index, mask, _semantic=_Semantic())


def test_indexed_load_rejects_bad_base_and_mode():
    for offset in (-1, 128, 0.5):
        with pytest.raises((TypeError, ValueError)):
            simd.vgather(_buffer(tl.float32)[offset], _vec(64, tl.uint32), _semantic=_Semantic())
    with pytest.raises(ValueError, match="invalid mode"):
        simd.vld(_buffer(tl.float32), _vec(64, tl.uint32), mode="unknown", _semantic=_Semantic())


def test_vunpack_rejects_ignored_selectors_and_partial_registers():
    for src, dst, kwargs in [(_vec(256, tl.int8), tl.int16, dict(pp="pp1")),
                             (_vec(256, tl.int8), tl.int32, dict(part=simd.PART.ODD)), (_vec(128,
                                                                                             tl.int8), tl.int16, {})]:
        with pytest.raises(ValueError):
            simd.vunpack(src, dst, _semantic=_Semantic(), **kwargs)


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("mode", (simd.MODE.ZEROING, simd.MODE.MERGING))
def test_vunpack_result_mask_and_selectors(signed, mode):
    sem = _Semantic()
    mask = _mask(64)
    result = simd.vunpack(_vec(256, tl.int8), tl.int32, mask, pp="pp3", signed=signed, mode=mode, _semantic=sem)
    name, args = _last(sem)
    assert result.type == simd.vector_type(64, tl.int32)
    assert name == "vcvt" and args[0] == ("extsi" if signed else "extui")
    assert args[2] is mask.handle
    assert args[-3:] == ("merging" if mode == simd.MODE.MERGING else "zeroing", "pp3", True)


@pytest.mark.parametrize("kwargs", [
    dict(pp="pp4"),
    dict(part="unknown"),
    dict(signed=1),
    dict(mode="unknown"),
    dict(mask=_mask(256)),
    dict(mask=_mask(64, tl.int32))
])
def test_vunpack_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        simd.vunpack(_vec(256, tl.int8), tl.int32, _semantic=_Semantic(), **kwargs)


@pytest.mark.parametrize("src,dst", [(tl.float16, tl.int32), (tl.int8, tl.float16), (tl.int16, tl.int8),
                                     (tl.int32, tl.int64)])
def test_vunpack_rejects_unsupported_widths(src, dst):
    with pytest.raises(ValueError, match="supports integer"):
        simd.vunpack(_vec(64, src), dst, _semantic=_Semantic())


# --------------------------------------------------------------------------
# exports / signatures
# --------------------------------------------------------------------------


def test_exports_and_module_mode_constants():
    for name in ("MODE", "POS", "vdiv", "vmax", "vmin", "vmaxs", "vexp", "vdup", "vdups", "vdupv", "vabs", "vneg",
                 "vnot", "vrelu", "vsqrt", "vabsdif", "vexpdif"):
        assert name in simd.__all__, f"{name} missing from simd.__all__"
        assert hasattr(simd, name), f"{name} missing from simd"
    assert simd.MODE.ZEROING == "MODE_ZEROING"
    assert simd.MODE.MERGING == "MODE_MERGING"


def test_signatures_match_api_list():
    import inspect

    p = inspect.signature(simd.vdiv).parameters
    assert list(p) == ["src0", "src1", "mask", "mode", "precision", "_semantic"]
    assert p["mask"].default is None
    assert p["mode"].default == simd.MODE.ZEROING
    assert p["precision"].default is None

    p = inspect.signature(simd.vdup).parameters
    assert list(p) == ["scalar", "dtype", "mask", "mode", "_semantic"]
    assert p["mode"].default == simd.MODE.ZEROING

    p = inspect.signature(simd.vdups).parameters
    assert list(p) == ["scalar", "dtype", "mask", "mode", "_semantic"]
    assert p["mode"].default == simd.MODE.ZEROING

    p = inspect.signature(simd.vdupv).parameters
    assert list(p) == ["src", "mask", "pos", "mode", "_semantic"]
    assert p["pos"].default == simd.POS.LOWEST
    assert p["mode"].default == simd.MODE.ZEROING
    assert simd.POS.LOWEST == "POS_LOWEST"
    assert simd.POS.HIGHEST == "POS_HIGHEST"

    p = inspect.signature(simd.vmaxs).parameters
    assert list(p) == ["src0", "scalar", "mask", "mode", "_semantic"]

    p = inspect.signature(simd.vexp).parameters
    assert list(p) == ["src", "mask", "mode", "precision", "_semantic"]

    for op in (simd.vmax, simd.vmin):
        assert list(inspect.signature(op).parameters) == ["src0", "src1", "mask", "mode", "_semantic"]


# --------------------------------------------------------------------------
# op selection
# --------------------------------------------------------------------------


def test_vdiv_float_and_signed_use_plain_vdiv():
    sem = _Semantic()
    simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0]) == ("binary", "div")

    sem = _Semantic()
    simd.vdiv(_vec(64, tl.int32), _vec(64, tl.int32), _semantic=sem)
    assert _last(sem)[1][0] == "div"


def test_vdiv_unsigned_selects_the_unsigned_cast():
    for dtype, tag in ((tl.uint16, "i16"), (tl.uint32, "i32"), (tl.uint64, "i64")):
        sem = _Semantic()
        simd.vdiv(_vec(64, dtype), _vec(64, dtype), _semantic=sem)
        name, args = _last(sem)
        assert (name, args[0]) == ("binary", "divu"), dtype


def test_vdiv_unsigned_masked_appends_explicit_zeroing_select():
    for dtype, tag in ((tl.uint16, "i16"), (tl.uint32, "i32")):
        sem = _Semantic()
        m = _mask(64)
        simd.vdiv(_vec(64, dtype), _vec(64, dtype), mask=m, _semantic=sem)
        calls = sem.builder.calls
        assert calls[0][0] == "binary" and calls[0][1][0] == "divu", dtype
        assert calls[0][1][3] is m.handle, dtype
        assert calls[0][1][4] == "zeroing", dtype
        assert calls[-1][0] == "vsel", (dtype, calls)
        assert calls[-1][1][0] is m.handle, dtype
        # false operand is an explicit zero vector (vsplat of integer zero)
        assert calls[-2][0] == "vsplat", (dtype, calls)
        assert calls[-2][1][1:] == (tag, 64), (dtype, calls)
        # MERGING is forwarded to the builder, and the Python zeroing
        # workaround must NOT stack a second select on top of the builder's
        # zero-init + vsel merge.
        sem_m = _Semantic()
        simd.vdiv(_vec(64, dtype), _vec(64, dtype), mask=m, mode=simd.MODE.MERGING, _semantic=sem_m)
        calls_m = sem_m.builder.calls
        assert calls_m[-1][0] == "binary", (dtype, calls_m)
        assert calls_m[-1][1][4] == "merging", (dtype, calls_m)
        assert all(c[0] != "vsel" for c in calls_m), (dtype, calls_m)

    # default all-true mask must not add a Python-side select
    for dtype in (tl.uint16, tl.uint32):
        sem = _Semantic()
        simd.vdiv(_vec(64, dtype), _vec(64, dtype), _semantic=sem)
        assert sem.builder.calls[-1][0] == "binary", dtype
        assert sem.builder.calls[-1][1][4] == "zeroing", dtype
    # signed/float masked division keeps the bare divide
    for dtype in (tl.int32, tl.float32):
        sem = _Semantic()
        simd.vdiv(_vec(64, dtype), _vec(64, dtype), mask=_mask(64), _semantic=sem)
        assert sem.builder.calls[-1][0] == "binary", dtype


def test_vdups_is_the_masked_vdup_alias():
    m = _mask(64)
    sem_a = _Semantic()
    simd.vdup(1.5, tl.float32, mask=m, _semantic=sem_a)
    sem_b = _Semantic()
    simd.vdups(1.5, tl.float32, mask=m, _semantic=sem_b)
    assert [c[0] for c in sem_a.builder.calls] == [c[0] for c in sem_b.builder.calls]
    assert sem_a.builder.calls[-1][1][1:] == sem_b.builder.calls[-1][1][1:]
    assert sem_a.builder.calls[-1][1][3] is m.handle
    assert sem_b.builder.calls[-1][1][3] is m.handle
    assert sem_a.builder.calls[-1][1][4] == "zeroing"

    sem = _Semantic()
    simd.vdups(7, tl.int32, _semantic=sem)
    name, args = _last(sem)
    assert name == "vdup"
    assert args[1:3] == ("i32", 64)
    assert sem.builder.calls[0] == ("pge", ("ALL", 64))


def test_vdupv_uses_vector_broadcast_low_and_high_lane():
    v = _vec(64, tl.float32)
    m = _mask(64)
    sem = _Semantic()
    simd.vdupv(v, mask=m, _semantic=sem)
    name, args = _last(sem)
    assert name == "vbroadcast"
    assert args == (v.handle, m.handle, True, "zeroing")
    assert sem.builder.calls[0][0] != "pge"  # explicit mask is not replaced

    sem = _Semantic()
    simd.vdupv(v, mask=m, pos=simd.POS.HIGHEST, _semantic=sem)
    assert _last(sem)[1] == (v.handle, m.handle, False, "zeroing")
    for spelling in ("POS_LOWEST", "LOWEST", "pos_lowest"):
        sem = _Semantic()
        simd.vdupv(v, mask=m, pos=spelling, _semantic=sem)
        assert _last(sem)[1][2] is True
    for spelling in ("POS_HIGHEST", "HIGHEST", "pos_highest"):
        sem = _Semantic()
        simd.vdupv(v, mask=m, pos=spelling, _semantic=sem)
        assert _last(sem)[1][2] is False

    # default mask is an all-true pge of the source lanes
    sem = _Semantic()
    simd.vdupv(_vec(128, tl.float16), _semantic=sem)
    assert sem.builder.calls[0] == ("pge", ("ALL", 128))
    assert _last(sem)[1][2] is True

    for bad in ("POS_MIDDLE", 0, None):
        with pytest.raises(ValueError, match="invalid pos"):
            simd.vdupv(v, mask=m, pos=bad, _semantic=_Semantic())
    # vector_broadcast has a real 64-lane library entry; i64/u64 are accepted.
    for dtype in (tl.int64, tl.uint64):
        sem = _Semantic()
        v64 = _vec(64, dtype)
        simd.vdupv(v64, mask=m, pos=simd.POS.HIGHEST, _semantic=sem)
        name, call = _last(sem)
        assert name == "vbroadcast"
        assert call == (v64.handle, m.handle, False, "zeroing")


def test_vmax_vmin_unsigned_selects_u_ops():
    for op, signed_kind, unsigned_kind in ((simd.vmax, "max", "umax"), (simd.vmin, "min", "umin")):
        sem = _Semantic()
        op(_vec(64, tl.float32), _vec(64, tl.float32), _semantic=sem)
        assert _last(sem)[1][0] == signed_kind

        sem = _Semantic()
        op(_vec(64, tl.uint8), _vec(64, tl.uint8), _semantic=sem)
        assert _last(sem)[1][0] == unsigned_kind

        sem = _Semantic()
        op(_vec(64, tl.int64), _vec(64, tl.int64), _semantic=sem)
        assert _last(sem)[1][0] == signed_kind


def test_vmaxs_signedness_and_scalar_operand():
    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.float32), 0.5, _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0]) == ("binary_scalar", "maxs")
    # scalar materialized as an f32 constant, not as an integer
    assert sem.builder.calls[-2][0] == "scalar"
    assert sem.builder.calls[-2][1] == ("f32", 0.5)

    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.uint32), 7, _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0]) == ("binary_scalar", "umaxs")
    assert sem.builder.calls[-2] == ("scalar_int", ("i32", 7))

    # int literal on a float vector is accepted (2 -> 2.0)
    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.float32), 2, _semantic=sem)
    assert sem.builder.calls[-2][1] == ("f32", 2.0)


def test_vmaxs_scalar_type_contract():
    sem = _Semantic()
    # float scalar on an integer vector is a hard error, not a truncation
    with pytest.raises(TypeError, match="is a float"):
        simd.vmaxs(_vec(64, tl.int32), 1.5, _semantic=sem)
    with pytest.raises(TypeError, match="got bool"):
        simd.vmaxs(_vec(64, tl.float32), True, _semantic=sem)
    with pytest.raises(ValueError, match="does not fit"):
        simd.vmaxs(_vec(64, tl.int8), 200, _semantic=sem)
    with pytest.raises(ValueError, match="does not fit"):
        simd.vmaxs(_vec(64, tl.uint8), -1, _semantic=sem)
    with pytest.raises(TypeError, match="must be a constexpr number"):
        simd.vmaxs(_vec(64, tl.float32), "nope", _semantic=sem)


def test_vexp_uses_vexp_op():
    sem = _Semantic()
    simd.vexp(_vec(64, tl.float32), _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0]) == ("unary", "exp")
    sem = _Semantic()
    simd.vexp(_vec(128, tl.float16), _semantic=sem)
    assert _last(sem)[1][0] == "exp"


def test_new_unary_and_binary_ops_use_expected_kinds():
    for op, kind, dtype in ((simd.vabs, "abs", tl.float32), (simd.vneg, "neg", tl.float32),
                            (simd.vnot, "not", tl.int32), (simd.vrelu, "relu", tl.int32), (simd.vsqrt, "sqrt",
                                                                                           tl.float32)):
        sem = _Semantic()
        op(_vec(64, dtype), _semantic=sem)
        name, args = _last(sem)
        assert (name, args[0]) == ("unary", kind), (op.__name__, sem.builder.calls)

    for op, kind in ((simd.vabsdif, "absdiff"), (simd.vexpdif, "expdif")):
        sem = _Semantic()
        op(_vec(64, tl.float32), _vec(64, tl.float32), _semantic=sem)
        name, args = _last(sem)
        assert (name, args[0]) == ("binary", kind), (op.__name__, sem.builder.calls)

    mask = _mask(64)
    for op in (simd.vabs, simd.vneg, simd.vnot, simd.vrelu, simd.vsqrt):
        sem = _Semantic()
        dtype = tl.int32 if op in (simd.vnot, simd.vrelu) else tl.float32
        op(_vec(64, dtype), mask=mask, mode=simd.MODE.MERGING, _semantic=sem)
        name, call = _last(sem)
        assert name == "unary"
        assert call[2] is mask.handle and call[3] == "merging", (op.__name__, call)

    for op in (simd.vabsdif, simd.vexpdif):
        sem = _Semantic()
        op(_vec(64, tl.float32), _vec(64, tl.float32), mask=mask, mode=simd.MODE.MERGING, _semantic=sem)
        name, call = _last(sem)
        assert name == "binary"
        assert call[3] is mask.handle and call[4] == "merging", (op.__name__, call)

    with pytest.raises(ValueError, match="vabs does not support element type bf16"):
        simd.vabs(_vec(128, tl.bfloat16), _semantic=_Semantic())
    with pytest.raises(ValueError, match="vrelu does not support element type i8"):
        simd.vrelu(_vec(256, tl.int8), _semantic=_Semantic())
    with pytest.raises(ValueError, match="vabsdif does not support element type i32"):
        simd.vabsdif(_vec(64, tl.int32), _vec(64, tl.int32), _semantic=_Semantic())
    with pytest.raises(ValueError, match="vexpdif does not support element type i32"):
        simd.vexpdif(_vec(64, tl.int32), _vec(64, tl.int32), _semantic=_Semantic())


def test_vdup_uses_masked_broadcast_with_tla_lane_convention():
    sem = _Semantic()
    simd.vdup(1.0, tl.float32, _semantic=sem)
    name, args = _last(sem)
    assert name == "vdup"
    scalar_tag, lanes, _mask_handle = args[1], args[2], args[3]
    assert (scalar_tag, lanes) == ("f32", 64)
    # default mask is the all-true pge with the register's lane count
    assert sem.builder.calls[0] == ("pge", ("ALL", 64))

    sem = _Semantic()
    simd.vdup(3, tl.int8, _semantic=sem)
    assert _last(sem)[1][1:3] == ("i8", 256)
    assert sem.builder.calls[-2] == ("scalar_int", ("i8", 3))

    # dtype names are accepted (repo convention) and resolve the same way
    sem = _Semantic()
    simd.vdup(0.0, "bf16", _semantic=sem)
    assert _last(sem)[1][1:3] == ("bf16", 128)

    # TLA short spellings keep unsignedness (not silently mapped to int8)
    sem = _Semantic()
    simd.vdup(200, "u8", _semantic=sem)
    assert _last(sem)[1][1:3] == ("i8", 256)
    assert sem.builder.calls[-2] == ("scalar_int", ("i8", 200))

    # result is a tla.vector, never a tl.tensor
    sem = _Semantic()
    out = simd.vdup(0.0, tl.float32, _semantic=sem)
    assert isinstance(out, simd.vector)


def test_vdup_dtype_aliases_resolve_to_canonical_names():
    cases = {
        "i8": ("i8", tl.int8),
        "int8": ("i8", tl.int8),
        "u8": ("i8", tl.uint8),
        "uint8": ("i8", tl.uint8),
        "i16": ("i16", tl.int16),
        "int16": ("i16", tl.int16),
        "u16": ("i16", tl.uint16),
        "uint16": ("i16", tl.uint16),
        "i32": ("i32", tl.int32),
        "int32": ("i32", tl.int32),
        "u32": ("i32", tl.uint32),
        "uint32": ("i32", tl.uint32),
        "i64": ("i64", tl.int64),
        "int64": ("i64", tl.int64),
        "u64": ("i64", tl.uint64),
        "uint64": ("i64", tl.uint64),
        "f16": ("f16", tl.float16),
        "fp16": ("f16", tl.float16),
        "float16": ("f16", tl.float16),
        "f32": ("f32", tl.float32),
        "fp32": ("f32", tl.float32),
        "float32": ("f32", tl.float32),
        "bf16": ("bf16", tl.bfloat16),
        "bfloat16": ("bf16", tl.bfloat16),
        "fp8e4nv": ("fp8e4nv", tl.float8e4nv),
        "float8e4nv": ("fp8e4nv", tl.float8e4nv),
        "fp8e4m3fn": ("fp8e4nv", tl.float8e4nv),
        "float8e4m3fn": ("fp8e4nv", tl.float8e4nv),
    }
    for name, (tag, dtype) in cases.items():
        got_tag, got_dtype = simd._resolve_element_dtype(name)
        assert (got_tag, got_dtype) == (tag, dtype), name
        assert simd._is_unsigned(got_dtype) == dtype.is_int_unsigned(), name

    # dtype objects are passed through unchanged (including unsigned)
    for dtype in (tl.int8, tl.uint8, tl.int32, tl.uint64, tl.float16, tl.bfloat16, tl.float8e4nv):
        got_tag, got_dtype = simd._resolve_element_dtype(dtype)
        assert got_dtype == dtype
        assert got_tag == simd._elem_tag(dtype)

    # invalid names/objects fail with a clear ValueError, not tl.dtype's assert
    for bad in ("nope", "fp64", "i128", None, 3):
        with pytest.raises(ValueError, match="unsupported vdup dtype"):
            simd._resolve_element_dtype(bad)
    with pytest.raises(ValueError, match="unsupported simd element dtype"):
        simd._resolve_element_dtype(tl.int1)


def test_u64_scalar_uses_exact_twos_complement():
    # [2**63, 2**64) must reach the int64 binding as negative two's-complement
    # bits; no float conversion is involved.
    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.uint64), 2**63, _semantic=sem)
    assert sem.builder.calls[-2] == ("scalar_int", ("i64", -(2**63)))
    assert all(c[0] != "scalar" for c in sem.builder.calls)

    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.uint64), 2**64 - 1, _semantic=sem)
    assert sem.builder.calls[-2] == ("scalar_int", ("i64", -1))

    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.uint64), 2**63 - 1, _semantic=sem)
    assert sem.builder.calls[-2] == ("scalar_int", ("i64", 2**63 - 1))

    # signed i64 boundaries are untouched
    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.int64), -(2**63), _semantic=sem)
    assert sem.builder.calls[-2] == ("scalar_int", ("i64", -(2**63)))

    # out-of-range u64 values are rejected by the frontend range check
    for bad in (2**64, -1):
        with pytest.raises(ValueError, match="does not fit"):
            simd.vmaxs(_vec(64, tl.uint64), bad, _semantic=_Semantic())


def test_exact_i64_scalar_has_no_double_roundtrip():
    big = 2**62 + 12345
    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.int64), big, _semantic=sem)
    tag, value = sem.builder.calls[-2][1]
    assert (tag, value) == ("i64", big)
    assert isinstance(value, int)
    # the double path would have lost precision -- prove it would differ
    assert float(big) != big

    # a 2**63-1 constexpr is the largest exact i64 and must survive as well
    biggest = 2**63 - 1
    sem = _Semantic()
    simd.vmaxs(_vec(64, tl.int64), biggest, _semantic=sem)
    assert sem.builder.calls[-2][1] == ("i64", biggest)
    with pytest.raises(ValueError, match="does not fit"):
        simd.vmaxs(_vec(64, tl.int64), 2**63, _semantic=_Semantic())


def test_vdup_rejects_i64():
    with pytest.raises(ValueError, match="vdup does not support element type i64"):
        simd.vdup(1, tl.int64, _semantic=_Semantic())
    with pytest.raises(ValueError, match="vdup does not support element type i64"):
        simd.vdup(1, tl.uint64, _semantic=_Semantic())


# --------------------------------------------------------------------------
# rejection paths
# --------------------------------------------------------------------------


def test_mode_zeroing_and_merging_reach_the_builder_distinctly():
    """Both modes forward the same op/mask but a different builder mode.

    ZEROING emits the bare op; MERGING emits the same op plus the builder-side
    zero-init + merge select.  The frontend must never rewrite MERGING to
    ZEROING, and a custom mask is forwarded unchanged for every spelling.
    """
    cases = (
        (simd.vdiv, (_vec(64, tl.float32), _vec(64, tl.float32))),
        (simd.vmax, (_vec(64, tl.float32), _vec(64, tl.float32))),
        (simd.vmin, (_vec(64, tl.float32), _vec(64, tl.float32))),
        (simd.vmaxs, (_vec(64, tl.float32), 1.0)),
        (simd.vexp, (_vec(64, tl.float32), )),
    )
    for op, args in cases:
        mask = _mask(64)
        calls = {}
        for mode in (simd.MODE.ZEROING, simd.MODE.MERGING, "ZEROING", "MERGING"):
            sem = _Semantic()
            op(*args, mask=mask, mode=mode, _semantic=sem)
            name, call = _last(sem)
            # mask stays in its original positional slot for these bindings
            assert mask.handle in call, (op.__name__, mode)
            assert all(c[0] != "pge" for c in sem.builder.calls)
            calls[str(mode).upper()] = (name, call[0], call[-1])
        # ZEROING and MERGING spellings normalize to exactly two builder modes
        z = {calls["MODE_ZEROING"], calls["ZEROING"]}
        m = {calls["MODE_MERGING"], calls["MERGING"]}
        assert len(z) == 1 and len(m) == 1, (op.__name__, calls)
        assert next(iter(z))[2] == "zeroing", (op.__name__, calls)
        assert next(iter(m))[2] == "merging", (op.__name__, calls)
        # op kind and mask wiring are identical; only the mode argument differs
        assert next(iter(z))[:2] == next(iter(m))[:2], (op.__name__, calls)

    # vdup has a scalar lane but the same trailing mode contract
    mask = _mask(64)
    modes = {}
    for mode in (simd.MODE.ZEROING, simd.MODE.MERGING, "ZEROING", "MERGING"):
        sem = _Semantic()
        simd.vdup(1.0, tl.float32, mask=mask, mode=mode, _semantic=sem)
        name, call = _last(sem)
        assert name == "vdup"
        assert call[3] is mask.handle, mode
        modes[str(mode).upper()] = call[4]
    assert modes == {"MODE_ZEROING": "zeroing", "ZEROING": "zeroing", "MODE_MERGING": "merging", "MERGING":
                     "merging"}, modes

    # vdups must share vdup's mode forwarding, not re-hardcode zeroing
    sem = _Semantic()
    simd.vdups(1.0, tl.float32, mask=mask, mode=simd.MODE.MERGING, _semantic=sem)
    assert _last(sem)[1][4] == "merging"

    # unknown modes are still rejected (never treated as zeroing)
    for op, args in cases:
        with pytest.raises(ValueError, match="invalid mode"):
            op(*args, mode="MODE_WHATEVER", _semantic=_Semantic())
    with pytest.raises(ValueError, match="invalid mode"):
        simd.vdup(1.0, tl.float32, mode="MODE_WHATEVER", _semantic=_Semantic())


def test_precision_tiers():
    # hardware tiers are accepted and change nothing
    for precision in (None, "ftz_true", "FTZ_TRUE", tl.constexpr("ftz_true")):
        for op, args in ((simd.vdiv, (_vec(64, tl.float32), _vec(64, tl.float32))), (simd.vexp, (_vec(64,
                                                                                                      tl.float32), ))):
            sem = _Semantic()
            op(*args, precision=precision, _semantic=sem)
            assert "divfhp" not in str(sem.builder.calls[-1])
    # the undocumented TileLang-free 'intrinsic' alias is not exposed
    with pytest.raises(ValueError, match="invalid precision"):
        simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), precision="intrinsic", _semantic=_Semantic())
    with pytest.raises(ValueError, match="invalid precision"):
        simd.vexp(_vec(64, tl.float32), precision="intrinsic", _semantic=_Semantic())
    # vexp's subnormal-preserving algorithm has no committed AVE lowering:
    # explicit NotImplementedError, never a silent FTZ fallback.
    for precision in ("ftz_false", "vexp_1ulp_ftz_false"):
        with pytest.raises(NotImplementedError, match="vexp_1ulp_ftz_false"):
            simd.vexp(_vec(64, tl.float32), precision=precision, _semantic=_Semantic())
    with pytest.raises(ValueError, match="invalid precision"):
        simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), precision="0ulp", _semantic=_Semantic())


def test_vdiv_exact_uses_vdivfhp_and_forwards_mask():
    """f32 exact aliases map onto the real VFDivFHPOp, never VFDivOp."""
    mask = _mask(64)
    for precision in ("exact", "EXACT", "vdiv_0ulp_ftz_true"):
        sem = _Semantic()
        simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), mask=mask, precision=precision, _semantic=sem)
        name, call = _last(sem)
        assert (name, call[0]) == ("binary", "divfhp"), (precision, sem.builder.calls)
        assert call[3] is mask.handle, precision
        assert call[4] == "zeroing", precision
        assert all(c[0] != "pge" for c in sem.builder.calls), precision
    # MERGING on the precise path still reaches the builder while the dtype
    # restriction is unchanged.
    sem = _Semantic()
    simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), mask=mask, precision="exact", mode=simd.MODE.MERGING,
              _semantic=sem)
    assert _last(sem)[1][0] == "divfhp"
    assert _last(sem)[1][4] == "merging"
    # exact is f32-only, matching TileLang's fp32-only precise wrappers
    for dtype in (tl.float16, tl.int32, tl.uint32):
        with pytest.raises(ValueError, match="requires float32"):
            simd.vdiv(_vec(64, dtype), _vec(64, dtype), precision="exact", _semantic=_Semantic())
    # ftz_false is not a vdiv tier
    with pytest.raises(ValueError, match="invalid precision"):
        simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), precision="ftz_false", _semantic=_Semantic())


def test_all_eight_apis_forward_the_exact_custom_mask_handle():
    """A caller-supplied mask must reach every builder call unchanged."""
    mask = _mask(64)
    a = _vec(64, tl.float32)
    b = _vec(64, tl.float32)
    cases = (
        (simd.vdiv, (a, b), 3),
        (simd.vmax, (a, b), 3),
        (simd.vmin, (a, b), 3),
        (simd.vmaxs, (a, 1.0), 3),
        (simd.vexp, (a, ), 2),
        (simd.vdup, (1.5, tl.float32), 3),
        (simd.vdups, (1.5, tl.float32), 3),
        (simd.vdupv, (a, ), 1),
    )
    for op, args, index in cases:
        sem = _Semantic()
        op(*args, mask=mask, _semantic=sem)
        name, call = _last(sem)
        assert call[index] is mask.handle, (op.__name__, name, call)
        assert all(c[0] != "pge" for c in sem.builder.calls), op.__name__


def test_mask_validation_and_custom_mask():
    sem = _Semantic()
    custom = _mask(64)
    simd.vmax(_vec(64, tl.float32), _vec(64, tl.float32), mask=custom, _semantic=sem)
    name, args = _last(sem)
    assert args[3] is custom.handle or args[3] is custom, "custom mask must be forwarded"
    # no default pge was emitted when a mask is supplied
    assert all(c[0] != "pge" for c in sem.builder.calls)

    with pytest.raises(ValueError, match="mask must be vector"):
        simd.vmax(_vec(64, tl.float32), _vec(64, tl.float32), mask=_mask(32), _semantic=_Semantic())
    with pytest.raises(ValueError, match="mask must be vector"):
        simd.vmaxs(_vec(64, tl.float32), 1.0, mask=_mask(64, tl.float32), _semantic=_Semantic())
    with pytest.raises(TypeError, match="mask must be a tla.vector"):
        simd.vmax(_vec(64, tl.float32), _vec(64, tl.float32), mask=1, _semantic=_Semantic())


def test_operand_type_mismatch():
    with pytest.raises(ValueError, match="same-typed"):
        simd.vmax(_vec(64, tl.float32), _vec(64, tl.int32), _semantic=_Semantic())
    with pytest.raises(ValueError, match="same-typed"):
        simd.vdiv(_vec(64, tl.float32), _vec(32, tl.float32), _semantic=_Semantic())
    with pytest.raises(TypeError, match="must be a tla.vector"):
        simd.vmax(1.0, _vec(64, tl.float32), _semantic=_Semantic())


def test_dtype_support_is_enforced_per_op():
    cases = [
        (lambda s: simd.vexp(_vec(64, tl.bfloat16), _semantic=s), "vexp", "bf16"),
        (lambda s: simd.vexp(_vec(64, tl.int32), _semantic=s), "vexp", "i32"),
        (lambda s: simd.vdiv(_vec(64, tl.int8), _vec(64, tl.int8), _semantic=s), "vdiv", "i8"),
        (lambda s: simd.vdiv(_vec(64, tl.bfloat16), _vec(64, tl.bfloat16), _semantic=s), "vdiv", "bf16"),
        (lambda s: simd.vmaxs(_vec(64, tl.bfloat16), 0.0, _semantic=s), "vmaxs", "bf16"),
        (lambda s: simd.vdup(0.0, tl.int64, _semantic=s), "vdup", "i64"),
    ]
    for call, op, tag in cases:
        with pytest.raises(ValueError, match=f"{op} does not support element type {tag}"):
            call(_Semantic())
    # fp8 is supported by vdup (VdupsZ has an F8E4M3FN slot)
    sem = _Semantic()
    simd.vdup(0.0, tl.float8e4nv, _semantic=sem)
    assert _last(sem)[1][1] == "fp8e4nv"
    # bf16 is fine for vmax but not for vmaxs (no bf16 scalar slot)
    sem = _Semantic()
    simd.vmax(_vec(64, tl.bfloat16), _vec(64, tl.bfloat16), _semantic=sem)
    assert _last(sem)[1][0] == "max"


def test_existing_internal_callers_still_work():
    """math.compensated_rstd calls vdiv/vmul/vfma/vsplat with keyword _semantic."""
    src = (_TLA_DIR / "math.py").read_text()
    assert "S.vdiv(" in src
    # the call shape used there must still bind: vdiv(a, b, _semantic=...)
    sem = _Semantic()
    simd.vdiv(_vec(64, tl.float32), _vec(64, tl.float32), _semantic=sem)
    assert _last(sem)[0] == "binary"


# --------------------------------------------------------------------------
# static binding consistency (no build required)
# --------------------------------------------------------------------------


def test_every_python_builder_call_is_bound_and_whitelisted():
    # SIMD bindings live in tla/cpp/tla_builder.cc; the shared Ascend builder
    # (ascend_ir.cc) contributes the generic ones (create_annotation_mark, ...).
    cpp = (_TLA_DIR / "cpp" / "tla_builder.cc").read_text()
    shared_cpp = (_TLA_DIR.parents[1] / "ascend_ir.cc").read_text()
    bound = set(re.findall(r'\.def\(\s*"(create_[a-z0-9_]+)"', cpp + shared_cpp))

    builder_py = (_TLA_DIR / "builder.py").read_text()
    tree = ast.parse(builder_py)
    whitelisted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ascend_methods":
            whitelisted = {elt.value for elt in node.value.elts}

    used = set()
    for py in _TLA_DIR.glob("*.py"):
        used |= set(re.findall(r"builder\.(create_[a-z0-9_]+)\(", py.read_text()))

    assert used, "expected builder.create_* calls in the tla python frontend"
    assert used <= bound, f"used but not bound in tla_builder.cc: {sorted(used - bound)}"
    assert used <= whitelisted, f"used but missing from ascend_methods: {sorted(used - whitelisted)}"


def test_new_bindings_and_ave_ops_are_present():
    cpp = (_TLA_DIR / "cpp" / "tla_builder.cc").read_text()
    builder_py = (_TLA_DIR / "builder.py").read_text()
    for name in ("create_simd_binary_scalar", "create_simd_vdup", "create_simd_scalar_int"):
        assert f'"{name}"' in cpp, f"{name} not bound"
        assert f"'{name}'" in builder_py, f"{name} not whitelisted"
    # each API must map to its own AVE op, not to a reduce/splat stand-in
    for ave_op in ("hivmave::VFDivOp", "hivmave::VFDivFHPOp", "hivmave::VFExpOp", "hivmave::VFBroadcastScalarMaskOp",
                   "hivmave::VFMaxOp", "hivmave::VFMinOp", "hivmave::VMaxUIOp", "hivmave::VMinUIOp",
                   "hivmave::VFMaxsOp", "hivmave::VMaxsUIOp", "hivmave::VFAbsOp", "hivmave::VFNegOp",
                   "hivmave::VFNotOp", "hivmave::VFReluOp", "hivmave::VFAbsDiffOp", "hivmave::VFExpdifOp"):
        assert ave_op in cpp, f"{ave_op} missing from tla_builder.cc"
    assert '"divfhp"' in cpp, "VFDivFHPOp builder kind is not dispatched"
    # vdup must not be routed through the unmasked scalar_broadcast
    vdup_block = cpp[cpp.index('"create_simd_vdup"'):cpp.index('"create_simd_vcvt"')]
    assert "VFBroadcastScalarMaskOp" in vdup_block
    assert "VFBroadcastScalarOp" not in vdup_block.replace("VFBroadcastScalarMaskOp", "")
    # mode must be consumed by a real lowering, not a decorative attr or a
    # discarded argument: MERGING materializes a +0 merge source and selects it.
    assert "applySimdMode" in cpp
    mode_block = cpp[cpp.index("Value applySimdMode"):cpp.index("} // namespace")]
    assert '"zeroing"' in mode_block and '"merging"' in mode_block
    assert "IntegerAttr::get(elemTy, 0)" in mode_block
    assert "FloatAttr::get(elemTy, 0.0)" in mode_block
    assert "hivmave::VFBroadcastScalarOp" in mode_block
    assert "hivmave::VFSelectOp" in mode_block
    assert "(void)mode" not in cpp
    # every mode-taking binding exposes the default and forwards it
    for binding in ("create_simd_binary", "create_simd_binary_scalar", "create_simd_unary", "create_simd_vdup",
                    "create_simd_vbroadcast"):
        start = cpp.index(f'"{binding}"')
        block = cpp[start:cpp.index(".def(", start)]
        assert "std::string mode" in block, binding
        assert 'py::arg("mode") = "zeroing"' in block, binding


def test_public_names_are_defined_and_documented():
    src = (_TLA_DIR / "simd.py").read_text()
    tree = ast.parse(src)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("vdiv", "vmax", "vmin", "vmaxs", "vexp", "vdup", "vdups", "vdupv"):
        assert name in defined, f"{name} not defined in simd.py"
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.get_docstring(node), f"{name} has no docstring"
    for name in ("membar", "pge", "vci"):
        assert f'"{name}"' in src


@pytest.mark.parametrize("part,number", [(simd.PART.LOWER, 0), (simd.PART.HIGHER, 1)])
@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("mode", (simd.MODE.ZEROING, simd.MODE.MERGING))
def test_native_vunpack_unmasked_is_one_operation(part, number, signed, mode):
    sem = _Semantic()
    src = _vec(256, tl.int8)
    result = simd.vunpack(src, tl.int16, part=part, signed=signed, mode=mode, _semantic=sem)
    assert result.type == simd.vector_type(128, tl.int16)
    assert sem.builder.calls == [("vunpack", (src.handle, "i16", number, signed))]


def test_native_vunpack_mask_and_invalid_width():
    sem, mask = _Semantic(), _mask(128)
    simd.vunpack(_vec(256, tl.int8), tl.int16, mask, part=simd.PART.HIGHER, _semantic=sem)
    assert _last(sem)[0] == "vsel" and _last(sem)[1][0] is mask.handle
    with pytest.raises(ValueError):
        simd.vunpack(_vec(256, tl.int8), tl.int32, part=simd.PART.LOWER, _semantic=_Semantic())
    with pytest.raises(ValueError):
        simd.vunpack(_vec(256, tl.int8), tl.int16, _mask(64), part=simd.PART.LOWER, _semantic=_Semantic())


def test_unmasked_strided_vunpack_omits_result_select():
    sem = _Semantic()
    simd.vunpack(_vec(256, tl.int8), tl.int16, _semantic=sem)
    assert _last(sem)[0] == "vcvt" and _last(sem)[1][-1] is False


@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("mode", (simd.MODE.ZEROING, simd.MODE.MERGING))
def test_byte_gather_native_predicate_routing(masked, mode):
    sem = _Semantic()
    mask = _mask(256) if masked else None
    indices = (_vec(128, tl.uint16), _vec(128, tl.uint16))
    result = simd.vgather(_buffer(tl.uint8, shape=(1024, ))[0], indices, mask, mode=mode, _semantic=sem)
    assert result.type == simd.vector_type(256, tl.uint8)
    calls = sem.builder.calls
    splits = [args for name, args in calls if name == "gather_masks"]
    gathers = [args for name, args in calls if name == "vgather"]
    assert len(gathers) == 2
    assert gathers[0][2] is indices[0].handle and gathers[1][2] is indices[1].handle
    if masked:
        assert splits == [(mask.handle, )]
        assert [args[3].ty for args in gathers] == ["low_mask", "high_mask"]
    else:
        assert not splits and gathers[0][3] is gathers[1][3]
    assert all(name != "vsel" for name, _ in calls)


@pytest.mark.parametrize("op,lanes,out_lanes", [(simd.ppack, 64, 128), (simd.ppack, 128, 256), (simd.punpack, 256, 128),
                                                (simd.punpack, 128, 64)])
@pytest.mark.parametrize("part,number", [(simd.PART.LOWER, 0), (simd.PART.HIGHER, 1)])
def test_native_predicate_pack_contract(op, lanes, out_lanes, part, number):
    sem, src = _Semantic(), _mask(lanes)
    result = op(src, part=part, _semantic=sem)
    assert result.type == simd.vector_type(out_lanes, tl.int1)
    assert _last(sem) == ("pred_pack", (src.handle, op is simd.punpack, number))


@pytest.mark.parametrize("op", (simd.ppack, simd.punpack))
def test_native_predicate_pack_invalid(op):
    for src, part in ((_mask(32), simd.PART.LOWER), (_vec(128, tl.int16), simd.PART.LOWER),
                      (_mask(128), simd.PART.EVEN), (_mask(256 if op is simd.ppack else 64), simd.PART.LOWER)):
        with pytest.raises((ValueError, TypeError)):
            op(src, part=part, _semantic=_Semantic())


@pytest.mark.parametrize("op", (simd.pintlv, simd.pdintlv))
@pytest.mark.parametrize("lanes", (64, 128, 256))
def test_native_predicate_interleave_contract(op, lanes):
    sem, a, b = _Semantic(), _mask(lanes), _mask(lanes)
    lo, hi = op(a, b, _semantic=sem)
    assert lo.type == hi.type == a.type
    assert _last(sem) == ("pred_interleave", (a.handle, b.handle, op is simd.pdintlv))
    with pytest.raises(ValueError):
        op(a, _mask(128 if lanes != 128 else 64), _semantic=_Semantic())


@pytest.mark.parametrize("op,kind", [(simd.vcgadd, "add"), (simd.vcgmax, "max"), (simd.vcgmin, "min")])
@pytest.mark.parametrize("dtype,unsigned", [(tl.int16, False), (tl.uint16, True), (tl.float16, False)])
def test_native_group_reduce_contract(op, kind, dtype, unsigned):
    sem, src, mask = _Semantic(), _vec(128, dtype), _mask(128)
    result = op(src, mask, mode=simd.MODE.MERGING, _semantic=sem)
    assert result.type == src.type
    assert sem.builder.calls == [("group_reduce", (kind, src.handle, mask.handle, unsigned))]
    for kwargs in (dict(mask=_mask(256)), dict(mode="invalid")):
        with pytest.raises(ValueError):
            op(src, _semantic=_Semantic(), **kwargs)
    for bad in (_vec(128, tl.bfloat16), _vec(64, tl.int64), _vec(64, tl.float16)):
        with pytest.raises(ValueError):
            op(bad, _semantic=_Semantic())


def test_vselr_contract_and_validation():
    sem, src, index = _Semantic(), _vec(256, tl.int8), _vec(256, tl.uint8)
    result = simd.vselr(src, index, _semantic=sem)
    assert result.type == src.type and sem.builder.calls == [("vselr", (src.handle, index.handle))]
    for idx in (_vec(256, tl.int8), _vec(128, tl.uint16), _vec(128, tl.uint8)):
        with pytest.raises(ValueError):
            simd.vselr(src, idx, _semantic=_Semantic())
    with pytest.raises(ValueError):
        simd.vselr(src, index, _mask(128), _semantic=_Semantic())
    with pytest.raises(ValueError):
        simd.vselr(src, index, mode="bad", _semantic=_Semantic())


def test_vscatter_contract_and_validation():
    sem, src, index, mask = _Semantic(), _vec(64, tl.float32), _vec(64, tl.uint32), _mask(64)
    buf = _buffer(tl.float32, shape=(1024, ))
    assert simd.vscatter(buf[32], index, src, mask, _semantic=sem) is None
    name, args = _last(sem)
    assert name == "vscatter" and args[0] is buf.handle and args[1].ty == 32
    assert args[2:] == (src.handle, index.handle, mask.handle)
    for b, i, s, m in ((_buffer(tl.float32, space="gm"), index, src, mask), (buf, _vec(128, tl.uint16), src, mask),
                       (buf, index, src, _mask(128)), (buf, index, _vec(64, tl.int32),
                                                       mask), (_buffer(tl.float32, strides=(2, )), index, src, mask)):
        with pytest.raises((TypeError, ValueError)):
            simd.vscatter(b[0], i, s, m, _semantic=_Semantic())


@pytest.mark.parametrize("masked", (False, True))
def test_byte_scatter_native_predicate_routing(masked):
    sem = _Semantic()
    mask = _mask(256) if masked else None
    src = _vec(256, tl.uint8)
    indices = (_vec(128, tl.uint16), _vec(128, tl.uint16))
    simd.vscatter(_buffer(tl.uint8, shape=(1024,))[0], indices, src, mask, _semantic=sem)
    scatters = [args for name, args in sem.builder.calls if name == "vscatter"]
    assert len(scatters) == 2
    assert [args[2].ty for args in scatters] == ["low_data", "high_data"]
    assert [args[3] for args in scatters] == [idx.handle for idx in indices]
    splits = [args for name, args in sem.builder.calls if name == "gather_masks"]
    if masked:
        assert splits == [(mask.handle,)]
        assert [args[4].ty for args in scatters] == ["low_mask", "high_mask"]
    else:
        assert not splits and scatters[0][4] is scatters[1][4]
