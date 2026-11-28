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


def _source_package():
    """Register the in-tree ``language/tla`` directory as a throwaway package."""
    if "tla_src" not in sys.modules:
        pkg = types.ModuleType("tla_src")
        pkg.__path__ = [str(_TLA_DIR)]
        sys.modules["tla_src"] = pkg
    return importlib.import_module


def _load_simd():
    """Import the in-tree ``language/tla/simd.py`` as a throwaway package."""
    return _source_package()("tla_src.simd")


def _load_selection():
    """Import the in-tree ``language/tla/selection.py`` (not site-packages)."""
    return _source_package()("tla_src.selection")


simd = _load_simd()
selection = _load_selection()
math_src = _source_package()("tla_src.math")
aux_src = _source_package()("tla_src.aux_ops")
tla_core = importlib.import_module("tla_src.core")


def _triton_is_builtin(fn):
    """Call triton's own ``is_builtin`` on whichever module actually has it.

    The helper lives in ``triton.language.core`` upstream but must not be
    hard-coded, since the Ascend build can move/alias it.
    """
    for mod_name in ("triton.language.core", "triton.language"):
        probe = getattr(importlib.import_module(mod_name), "is_builtin", None)
        if callable(probe):
            return probe(fn)
    raise AssertionError("triton exposes no is_builtin(); cannot verify @builtin")


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

    def create_simd_shift(self, kind, lhs, rhs, mask, sign, mode):
        return self._record("shift", kind, lhs, rhs, mask, sign, mode)

    def create_simd_shift_scalar(self, kind, vec, scalar, mask, sign, mode):
        return self._record("shift_scalar", kind, vec, scalar, mask, sign, mode)

    def create_simd_vpack(self, src, part, out_elem_tag, out_lanes):
        return self._record("vpack", src, part, out_elem_tag, out_lanes)

    def create_simd_unary(self, kind, src, mask, mode="zeroing"):
        return self._record("unary", kind, src, mask, mode)

    def create_simd_vfma(self, acc, lhs, rhs, mask, mode):
        return self._record("vfma", acc, lhs, rhs, mask, mode)

    def create_simd_vcvt(self, kind, src, mask, out_elem_tag, part, round="round", sat=True, mode="zeroing"):
        return self._record("vcvt", kind, src, mask, out_elem_tag, part, round, sat, mode)

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

    def create_simd_pld(self, *args):
        return self._record("pld", *args)

    def create_simd_pst(self, *args):
        return self._record("pst", *args)

    def create_simd_vgather(self, *args):
        return self._record("vgather", *args)

    def create_simd_vunpack(self, *args):
        return self._record("vunpack", *args)


class _Semantic:

    def __init__(self):
        self.builder = _Recorder()

    def to_tensor(self, value):
        return tl.tensor(_Handle(value), tl.int32)


class _ScalarHandle:

    def __init__(self, op, value):
        self.op, self.value = op, value

    def __repr__(self):
        return f"{self.op}({self.value!r})"


class _ScalarBuilder:
    """Recording builder good enough for the real ``TritonSemantic``."""

    def __init__(self):
        self.calls = []

    def _make(self, op, value):
        self.calls.append((op, value))
        return _ScalarHandle(op, value)

    def get_int1(self, value):
        return self._make("get_int1", value)

    def get_int32(self, value):
        return self._make("get_int32", value)

    def get_uint32(self, value):
        return self._make("get_uint32", value)

    def get_int64(self, value):
        return self._make("get_int64", value)

    def get_fp32(self, value):
        return self._make("get_fp32", value)

    def get_float_ty(self):
        return self._make("get_float_ty", None)

    def get_fp64(self, value):
        return self._make("get_fp64", value)

    def create_sqrt(self, value):
        return self._make("create_sqrt", value)

    def get_null_value(self, _ty):
        return self._make("get_null_value", None)

    def create_select(self, cond, x, y):
        self.calls.append(("create_select", (cond, x, y)))
        return _ScalarHandle("select", (cond, x, y))

    def get_int8(self, value):
        return self._make("get_int8", value)

    def create_srem(self, lhs, rhs):
        return self._make("srem", (lhs, rhs))

    def create_assume(self, cond):
        self.calls.append(("create_assume", (cond, )))

    def create_urem(self, lhs, rhs):
        return self._make("urem", (lhs, rhs))


def _scalar_semantic():
    from triton.language.semantic import TritonSemantic

    builder = _ScalarBuilder()
    return TritonSemantic(builder), builder


class _NoBuilder:
    """Invalid inputs must fail before any builder operation is reached."""

    @property
    def builder(self):
        raise AssertionError("invalid tla.where input reached the builder")


def _vec(lanes, dtype):
    return simd.vector(_Handle("vec"), simd.vector_type(lanes, dtype))


def _mask(lanes, dtype=None):
    return simd.vector(_Handle("mask"), simd.vector_type(lanes, dtype or tl.int1))


def _last(sem):
    assert sem.builder.calls, "no builder call recorded"
    return sem.builder.calls[-1]


# --------------------------------------------------------------------------
# exports / signatures
# --------------------------------------------------------------------------


def test_exports_and_module_mode_constants():
    for name in ("MODE", "POS", "vdiv", "vmax", "vmin", "vmaxs", "vexp", "vdup", "vdups", "vdupv", "vabs", "vneg",
                 "vnot", "vrelu", "vsqrt", "vabsdif", "vexpdif", "vpack", "vshl", "vshls", "vshr", "vlrelu", "vprelu"):
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

    p = inspect.signature(simd.vpack).parameters
    assert list(p) == ["src", "part", "_semantic"]
    assert p["part"].default == "part_even"

    p = inspect.signature(simd.vshl).parameters
    assert list(p) == ["src0", "src1", "mask", "mode", "_semantic"]
    assert p["mode"].default == simd.MODE.ZEROING

    p = inspect.signature(simd.vshls).parameters
    assert list(p) == ["src", "scalar", "mask", "mode", "_semantic"]

    p = inspect.signature(simd.vlrelu).parameters
    assert list(p) == ["src", "alpha", "mask", "_semantic"]

    p = inspect.signature(simd.vprelu).parameters
    assert list(p) == ["src0", "src1", "mask", "_semantic"]

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


def test_phase1_extended_ops_use_expected_builder_calls():
    sem = _Semantic()
    out = simd.vpack(_vec(128, tl.int16), part="part_odd", _semantic=sem)
    name, args = _last(sem)
    assert (name, args[1], args[2], args[3]) == ("vpack", 1, "i8", 256)
    assert out.shape[0] == 256 and out.dtype == tl.int8

    sem = _Semantic()
    simd.vpack(_vec(64, tl.uint32), _semantic=sem)
    name, args = _last(sem)
    assert (name, args[1], args[2], args[3]) == ("vpack", 0, "i16", 128)

    with pytest.raises(ValueError, match="vpack supports only"):
        simd.vpack(_vec(64, tl.float32), _semantic=_Semantic())
    with pytest.raises(ValueError, match="part"):
        simd.vpack(_vec(128, tl.int16), part="part_middle", _semantic=_Semantic())

    for op, kind, dtype in ((simd.vshl, "shl", tl.int32), (simd.vshr, "shr", tl.int32)):
        sem = _Semantic()
        op(_vec(64, dtype), _vec(64, dtype), _semantic=sem)
        name, args = _last(sem)
        assert (name, args[0], args[4]) == ("shift", kind, True)

    sem = _Semantic()
    simd.vshr(_vec(64, tl.uint32), _vec(64, tl.uint32), _semantic=sem)
    assert _last(sem)[1][4] is False

    sem = _Semantic()
    simd.vshls(_vec(64, tl.int32), 3, _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0], args[4]) == ("shift_scalar", "shls", True)

    for value in (32767, -32768):
        sem = _Semantic()
        simd.vshls(_vec(64, tl.int32), value, _semantic=sem)
        assert _last(sem)[0] == "shift_scalar"
    for value in (32768, -32769):
        with pytest.raises(ValueError, match="does not fit int16"):
            simd.vshls(_vec(64, tl.int32), value, _semantic=_Semantic())

    sem = _Semantic()
    simd.vlrelu(_vec(64, tl.float32), 0.25, _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0]) == ("binary_scalar", "lrelus")

    sem = _Semantic()
    simd.vprelu(_vec(64, tl.float16), _vec(64, tl.float16), _semantic=sem)
    name, args = _last(sem)
    assert (name, args[0]) == ("binary", "prelu")

    for call in (
            lambda s: simd.vshl(_vec(64, tl.int64), _vec(64, tl.int64), _semantic=s),
            lambda s: simd.vshr(_vec(64, tl.float32), _vec(64, tl.float32), _semantic=s),
            lambda s: simd.vlrelu(_vec(64, tl.bfloat16), 0.25, _semantic=s),
            lambda s: simd.vprelu(_vec(64, tl.int32), _vec(64, tl.int32), _semantic=s),
    ):
        with pytest.raises(ValueError, match="does not support element type"):
            call(_Semantic())


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


@pytest.mark.parametrize("dtype", [tl.int64, tl.uint64])
def test_vmaxs_i64_zeroing_masked_selects_zero(dtype):
    src = _vec(64, dtype)
    mask = _mask(64)
    sem = _Semantic()
    simd.vmaxs(src, 7, mask=mask, mode=simd.MODE.ZEROING, _semantic=sem)
    op_call = next(args for name, args in sem.builder.calls if name == "binary_scalar")
    assert op_call[3] is mask.handle and op_call[4] == "zeroing"
    select = next(args for name, args in sem.builder.calls if name == "vsel")
    assert select[0] is mask.handle and select[2] is not src.handle

    sem = _Semantic()
    simd.vmaxs(src, 7, mode=simd.MODE.ZEROING, _semantic=sem)
    assert all(name != "vsel" for name, _ in sem.builder.calls)

    with pytest.raises(ValueError, match="invalid mode"):
        simd.vmaxs(src, 7, mode="MODE_WHATEVER", _semantic=_Semantic())


@pytest.mark.parametrize("dtype,scalar", [
    (tl.float16, 1.0),
    (tl.float32, 1.0),
    (tl.int8, 7),
    (tl.int16, 7),
    (tl.int32, 7),
    (tl.uint8, 7),
    (tl.uint16, 7),
    (tl.uint32, 7),
])
def test_vmaxs_non64_zeroing_masked_has_no_extra_select(dtype, scalar):
    sem = _Semantic()
    simd.vmaxs(_vec(64, dtype), scalar, mask=_mask(64), mode=simd.MODE.ZEROING, _semantic=sem)
    assert all(name != "vsel" for name, _ in sem.builder.calls)


def test_merging_composes_vsel_against_the_first_operand():
    """MODE.MERGING on the accumulate/vector-scalar family selects src0/acc.

    The op itself is always built under ZEROING; MERGING appends a vsel
    against the op's first vector operand (acc for the accumulate ops),
    mirroring CANN's V920 software-merge lowering.  The remaining register
    ops keep forwarding "merging" to the C++ builder (covered above).
    """
    mask = _mask(64)
    a = _vec(64, tl.float32)
    b = _vec(64, tl.float32)
    for op, args, merge in (
        (simd.vand, (a, b), a.handle),
        (simd.vmaxs, (a, 2.0), a.handle),
        (simd.vmuls, (a, 2.0), a.handle),
        (simd.vmula, (a, b, b), a.handle),
        (simd.vaxpy, (a, b, 1.5), a.handle),
        (simd.vln, (a, ), a.handle),
    ):
        sem = _Semantic()
        op(*args, mask=mask, mode=simd.MODE.MERGING, _semantic=sem)
        calls = sem.builder.calls
        assert calls[-1][0] == "vsel", (op.__name__, calls)
        assert calls[-1][1][0] is mask.handle, (op.__name__, calls)
        assert calls[-1][1][2] is merge, (op.__name__, calls)
        op_call = next(c for c in calls if c[0] in ("binary", "binary_scalar", "vfma", "unary"))
        assert op_call[1][-1] == "zeroing", (op.__name__, calls)

        sem = _Semantic()
        op(*args, mask=mask, mode=simd.MODE.ZEROING, _semantic=sem)
        assert all(c[0] != "vsel" for c in sem.builder.calls), op.__name__


def test_vcvt_masks_follow_the_source_register():
    """Width-changing vcvt predicates at the source register's granularity.

    Measured on Ascend950PR: f16->f32 takes the f16 register's 128-lane b16
    mask (gating the source lane each output reads); f32->f16 takes the f32
    register's 64-lane b32 mask (gating whole lane pairs).  Same-width
    conversions share the 64-lane rule.
    """
    # f16 -> f32 widening: the f16 source register's 128-lane mask
    sem = _Semantic()
    simd.vcvt(_vec(128, tl.float16), tl.float32, _semantic=sem)
    assert sem.builder.calls[0] == ("pge", ("ALL", 128))
    assert _last(sem)[0] == "vcvt"
    # f32 -> f16 narrowing: the f32 source register's 64-lane mask
    sem = _Semantic()
    simd.vcvt(_vec(64, tl.float32), tl.float16, _semantic=sem)
    assert sem.builder.calls[0] == ("pge", ("ALL", 64))
    # same-width conversions keep the shared 64-lane rule
    sem = _Semantic()
    simd.vcvt(_vec(64, tl.int32), tl.float32, _semantic=sem)
    assert sem.builder.calls[0] == ("pge", ("ALL", 64))
    # the mask must match the source register in either direction
    with pytest.raises(ValueError, match="mask must be vector<128xint1>"):
        simd.vcvt(_vec(128, tl.float16), tl.float32, mask=_mask(64), _semantic=_Semantic())
    with pytest.raises(ValueError, match="mask must be vector<64xint1>"):
        simd.vcvt(_vec(64, tl.float32), tl.float16, mask=_mask(128), _semantic=_Semantic())


def test_vcvt_rejects_merging():
    """vcvt changes dtype, so no result-typed merge source exists."""
    with pytest.raises(ValueError, match="does not support MODE.MERGING"):
        simd.vcvt(_vec(64, tl.float32), tl.float16, mode=simd.MODE.MERGING, _semantic=_Semantic())
    with pytest.raises(ValueError, match="invalid mode"):
        simd.vcvt(_vec(64, tl.float32), tl.float16, mode="MODE_WHATEVER", _semantic=_Semantic())


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
# tla.truncmod (math.py)
# --------------------------------------------------------------------------


def test_truncmod_export_and_builtin():
    assert "truncmod" in math_src.__all__
    assert _triton_is_builtin(math_src.truncmod)
    with pytest.raises(ValueError, match="Did you forget to add @triton.jit"):
        math_src.truncmod(7, 3)
    name = "tla_truncmod_export"
    spec = importlib.util.spec_from_file_location(name, _TLA_DIR / "__init__.py",
                                                  submodule_search_locations=[str(_TLA_DIR)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    try:
        spec.loader.exec_module(pkg)
        assert "truncmod" in pkg.__all__
        assert pkg.truncmod is pkg.math.truncmod
    finally:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]


@pytest.mark.parametrize("a,b", [(7, 3), (-7, 3), (7, -3), (-7, -3)])
def test_truncmod_literals_emit_signed_remainder(a, b):
    for args in ((a, b), (tl.constexpr(a), tl.constexpr(b))):
        sem, _ = _scalar_semantic()
        result = math_src.truncmod(*args, _semantic=sem)
        assert isinstance(result, tl.tensor) and result.type == tl.int32
        name, (lhs, rhs) = _last(sem)
        assert name == "srem"
        assert (lhs.op, lhs.value) == ("get_int32", a)
        assert (rhs.op, rhs.value) == ("get_int32", b)


@pytest.mark.parametrize("dtype,op", [(tl.int8, "srem"), (tl.int32, "srem"), (tl.int64, "srem"), (tl.uint32, "urem"),
                                      (tl.uint64, "urem")])
def test_truncmod_tensor_signedness(dtype, op):
    sem, _ = _scalar_semantic()
    a = tl.tensor(_ScalarHandle("arg", "a"), dtype)
    b = tl.tensor(_ScalarHandle("arg", "b"), dtype)
    result = math_src.truncmod(a, b, _semantic=sem)
    assert result.type == dtype
    assert _last(sem) == (op, (a.handle, b.handle))


@pytest.mark.parametrize("dtype,op,constant", [(tl.int8, "srem", "get_int8"), (tl.uint32, "urem", "get_uint32")])
def test_truncmod_literal_preserves_tensor_type(dtype, op, constant):
    sem, _ = _scalar_semantic()
    a = tl.tensor(_ScalarHandle("arg", "a"), dtype)
    result = math_src.truncmod(a, 3, _semantic=sem)
    assert result.type == dtype
    name, (lhs, rhs) = _last(sem)
    assert name == op and lhs is a.handle
    assert (rhs.op, rhs.value) == (constant, 3)


def test_truncmod_rejects_mixed_signedness():
    for lhs_ty, rhs_ty in ((tl.int32, tl.uint32), (tl.uint64, tl.int64)):
        sem, builder = _scalar_semantic()
        a = tl.tensor(_ScalarHandle("arg", "a"), lhs_ty)
        b = tl.tensor(_ScalarHandle("arg", "b"), rhs_ty)
        with pytest.raises(TypeError, match="different signedness"):
            math_src.truncmod(a, b, _semantic=sem)
        assert not builder.calls


@pytest.mark.parametrize("divisor", [0, tl.constexpr(0)])
def test_truncmod_rejects_literal_zero(divisor):
    with pytest.raises(ValueError, match="divisor must be nonzero"):
        math_src.truncmod(7, divisor, _semantic=_NoBuilder())


def test_truncmod_invalid_operands():
    values = [
        True, 1.5,
        tl.constexpr(True),
        tl.constexpr(1.0),
        tl.tensor("block", tl.block_type(tl.int32, [64])),
        tl.tensor(_ScalarHandle("arg", "i1"), tl.int1),
        tl.tensor(_ScalarHandle("arg", "f32"), tl.float32),
        _vec(64, tl.int32),
        tla_core.buffer("buf", tla_core.buffer_type((64, ), tl.int32, "ub"))
    ]
    for value in values:
        with pytest.raises(TypeError, match="truncmod a"):
            math_src.truncmod(value, 2, _semantic=_NoBuilder())
        with pytest.raises(TypeError, match="truncmod b"):
            math_src.truncmod(2, value, _semantic=_NoBuilder())


# --------------------------------------------------------------------------
# tla.sqrt (math.py)
# --------------------------------------------------------------------------


def test_sqrt_export_and_builtin():
    assert "sqrt" in math_src.__all__
    assert _triton_is_builtin(math_src.sqrt)
    with pytest.raises(ValueError, match="Did you forget to add @triton.jit"):
        math_src.sqrt(4.0)
    name = "tla_sqrt_export"
    spec = importlib.util.spec_from_file_location(name, _TLA_DIR / "__init__.py",
                                                  submodule_search_locations=[str(_TLA_DIR)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    try:
        spec.loader.exec_module(pkg)
        assert "sqrt" in pkg.__all__
        assert pkg.sqrt is pkg.math.sqrt
    finally:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]


@pytest.mark.parametrize("value", [4.0, tl.constexpr(4.0), -1.0, 0.0, float("inf"), float("nan")])
def test_sqrt_scalar_emits_expression(value):
    sem, builder = _scalar_semantic()
    result = math_src.sqrt(value, _semantic=sem)
    assert isinstance(result, tl.tensor) and result.type == tl.float32
    assert builder.calls[-1][0] == "create_sqrt"


def test_sqrt_runtime_scalar_and_large_literal():
    sem, builder = _scalar_semantic()
    value = tl.tensor(_ScalarHandle("arg", "x"), tl.float32)
    result = math_src.sqrt(value, _semantic=sem)
    assert result.type == value.type
    assert builder.calls[-1] == ("create_sqrt", value.handle)
    sem, builder = _scalar_semantic()
    with pytest.raises(TypeError, match="float32 scalars"):
        math_src.sqrt(1e300, _semantic=sem)
    assert not any(call[0] == "create_sqrt" for call in builder.calls)


@pytest.mark.parametrize("dtype,lanes", [(tl.float16, 128), (tl.float32, 64)])
def test_sqrt_vector_preserves_type(dtype, lanes):
    sem = _Semantic()
    value = _vec(lanes, dtype)
    result = math_src.sqrt(value, _semantic=sem)
    assert isinstance(result, simd.vector) and result.type == value.type
    assert sem.builder.calls[0] == ("pge", ("ALL", lanes))
    name, args = _last(sem)
    assert name == "unary" and args[0] == "sqrt"
    assert args[1] is value.handle and args[2].ty == "pge"
    assert args[3] == "zeroing"


def test_sqrt_rejects_invalid_inputs_before_builder():
    values = [
        1, True, 1 + 2j,
        tl.constexpr(1),
        tl.constexpr(True),
        tl.tensor("block", tl.block_type(tl.float32, [64])),
        tla_core.buffer("buf", tla_core.buffer_type((64, ), tl.float32, "ub"))
    ]
    values += [
        tl.tensor(_ScalarHandle("arg", "x"), dtype)
        for dtype in (tl.float16, tl.bfloat16, tl.float64, tl.int32, tl.int1)
    ]
    values += [_vec(64, dtype) for dtype in (tl.int32, tl.uint32, tl.bfloat16, tl.float64, tl.int1)]
    for value in values:
        with pytest.raises(TypeError, match="sqrt"):
            math_src.sqrt(value, _semantic=_NoBuilder())


# --------------------------------------------------------------------------
# tla.where (selection.py)
# --------------------------------------------------------------------------


def test_where_exported_from_source_package():
    assert "where" in selection.__all__
    name = "tla_source_export"
    spec = importlib.util.spec_from_file_location(name, _TLA_DIR / "__init__.py",
                                                  submodule_search_locations=[str(_TLA_DIR)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    try:
        spec.loader.exec_module(pkg)
        assert pkg.where.__name__ == "where"
        assert "where" in pkg.__all__
        assert _triton_is_builtin(pkg.where)
    finally:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]


def test_where_signature():
    import inspect

    params = inspect.signature(selection.where).parameters
    assert list(params) == ["condition", "x", "y", "_semantic"]
    assert params["_semantic"].default is None
    # ``where`` must stay a triton builtin so @triton.jit injects _semantic.
    assert _triton_is_builtin(selection.where)


def test_where_without_semantic_hits_builtin_guard():
    # Outside @triton.jit no _semantic is injected; the @builtin wrapper must
    # reject the call instead of running the body with _semantic=None.
    with pytest.raises(ValueError, match="Did you forget to add @triton.jit"):
        selection.where(True, 1, 2)


def test_where_scalar_bool_delegates_to_tl_where_with_promotion():
    sem, builder = _scalar_semantic()
    result = selection.where(True, 1, 2.5, _semantic=sem)
    assert result.type == tl.float32
    name, (cond, x, y) = builder.calls[-1]
    assert name == "create_select"
    assert cond.op == "get_int1" and cond.value is True
    # the two arms were promoted by TritonSemantic before the select, not
    # returned raw: the int arm materializes as an f32 handle.
    assert (x.op, x.value) == ("get_fp32", 1)
    assert (y.op, y.value) == ("get_fp32", 2.5)


def test_where_constexpr_bool_scalar_keeps_type_promotion():
    sem, builder = _scalar_semantic()
    result = selection.where(tl.constexpr(False), 1, 2.5, _semantic=sem)
    assert result.type == tl.float32
    assert any(call[0] == "create_select" for call in builder.calls)
    _, (_, x, y) = builder.calls[-1]
    assert x.op == y.op == "get_fp32"


def test_where_runtime_scalar_bool_condition_delegates():
    sem, builder = _scalar_semantic()
    cond = tl.tensor(_ScalarHandle("arg", "cond"), tl.int1)
    result = selection.where(cond, 1, 2, _semantic=sem)
    assert result.type == tl.int32
    name, (got_cond, x, y) = builder.calls[-1]
    assert name == "create_select"
    assert got_cond is cond.handle
    assert (x.value, y.value) == (1, 2)


def test_where_vector_mask_uses_vsel():
    sem = _Semantic()
    mask, a, b = _mask(64), _vec(64, tl.float32), _vec(64, tl.float32)
    result = selection.where(mask, a, b, _semantic=sem)
    assert result.type == a.type
    assert result is not a and result is not b
    name, args = _last(sem)
    assert name == "vsel"
    assert args == (mask.handle, a.handle, b.handle)


@pytest.mark.parametrize("condition, want", [(True, "x"), (False, "y"), (tl.constexpr(True), "x"),
                                             (tl.constexpr(False), "y")])
def test_where_const_bool_folds_vector_arms_without_ir(condition, want):
    sem = _Semantic()
    a, b = _vec(64, tl.float32), _vec(64, tl.float32)
    result = selection.where(condition, a, b, _semantic=sem)
    assert result is (a if want == "x" else b)
    assert sem.builder.calls == []


def test_where_invalid_inputs_never_reach_the_builder():
    vec = _vec(64, tl.float32)
    other = _vec(64, tl.int32)
    mask = _mask(64)
    block = tl.tensor("block", tl.block_type(tl.float32, [64]))
    block_cond = tl.tensor("block_cond", tl.block_type(tl.int1, [64]))
    int_cond = tl.tensor("int_cond", tl.int32)
    bool_cond = tl.tensor("bool_cond", tl.int1)
    buf = tla_core.buffer("buf", tla_core.buffer_type((64, ), tl.float32, "ub"))
    sem = _NoBuilder()
    cases = [
        (TypeError, "condition must be", lambda: selection.where(1, 1.0, 2.0, _semantic=sem)),
        (TypeError, "int1 dtype", lambda: selection.where(int_cond, 1.0, 2.0, _semantic=sem)),
        (TypeError, "block tensor", lambda: selection.where(block_cond, 1.0, 2.0, _semantic=sem)),
        (TypeError, "block tensor", lambda: selection.where(True, block, 1.0, _semantic=sem)),
        (TypeError, "scalar value", lambda: selection.where(True, buf, 1.0, _semantic=sem)),
        (TypeError, "vector condition requires", lambda: selection.where(mask, 1.0, 2.0, _semantic=sem)),
        (ValueError, "same-typed", lambda: selection.where(mask, vec, other, _semantic=sem)),
        (TypeError, "mix scalar and tla.vector", lambda: selection.where(True, vec, 1.0, _semantic=sem)),
        (TypeError, "runtime scalar bool", lambda: selection.where(bool_cond, vec, vec, _semantic=sem)),
        (ValueError, "mask must be", lambda: selection.where(_mask(32), vec, vec, _semantic=sem)),
        (ValueError, "mask must be", lambda: selection.where(_mask(64, tl.float32), vec, vec, _semantic=sem)),
    ]
    for exc_type, match, call in cases:
        with pytest.raises(exc_type, match=match):
            call()


# --------------------------------------------------------------------------
# tla.assume (aux_ops.py)
# --------------------------------------------------------------------------


def test_assume_export_and_builtin():
    assert "assume" in aux_src.__all__
    assert _triton_is_builtin(aux_src.assume)
    with pytest.raises(ValueError, match="Did you forget to add @triton.jit"):
        aux_src.assume(True)
    name = "tla_assume_export"
    spec = importlib.util.spec_from_file_location(name, _TLA_DIR / "__init__.py",
                                                  submodule_search_locations=[str(_TLA_DIR)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    try:
        spec.loader.exec_module(pkg)
        assert "assume" in pkg.__all__
        assert pkg.assume is pkg.aux_ops.assume
    finally:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]


@pytest.mark.parametrize("cond,value", [(True, True), (False, False), (tl.constexpr(True), True),
                                        (tl.constexpr(False), False)])
def test_assume_constants_emit(cond, value):
    sem, builder = _scalar_semantic()
    assert aux_src.assume(cond, _semantic=sem) is None
    assert builder.calls[0] == ("get_int1", value)
    assert len(builder.calls) == 2
    name, (handle, ) = builder.calls[-1]
    assert name == "create_assume"
    assert (handle.op, handle.value) == ("get_int1", value)


def test_assume_runtime_condition_emits_with_debug_disabled():
    sem, builder = _scalar_semantic()
    builder.options = types.SimpleNamespace(debug=False)
    cond = tl.tensor(_ScalarHandle("arg", "cond"), tl.int1)
    assert aux_src.assume(cond, _semantic=sem) is None
    assert builder.calls == [("create_assume", (cond.handle, ))]


def test_assume_invalid_inputs_never_reach_builder():
    values = [
        1, 0, 1.0, None,
        tl.constexpr(1),
        tl.constexpr(1.0),
        tl.tensor("block", tl.block_type(tl.int1, [64])),
        tl.tensor("integer", tl.int32),
        tl.tensor("float", tl.float32),
        _vec(64, tl.int1),
        tla_core.buffer("buf", tla_core.buffer_type((64, ), tl.float32, "ub"))
    ]
    for cond in values:
        with pytest.raises(TypeError, match="assume requires"):
            aux_src.assume(cond, _semantic=_NoBuilder())


# --------------------------------------------------------------------------
# static binding consistency (no build required)
# --------------------------------------------------------------------------


def test_every_python_builder_call_is_bound_and_whitelisted():
    # SIMD bindings live in tla/cpp/tla_builder.cc; the shared Ascend builder
    # (ascend_ir.cc) contributes the generic ones (create_annotation_mark, ...).
    cpp = (_TLA_DIR / "cpp" / "tla_builder.cc").read_text()
    shared_cpp = (_TLA_DIR.parents[1] / "ascend_ir.cc").read_text()
    bound = set(re.findall(r'\.def\(\s*"(create_[a-z0-9_]+)"', cpp + shared_cpp))
    # Scalar expressions also use the underlying Triton builder directly;
    # those methods already exist and must not be rebound through Ascend.
    base_cpp = (_TLA_DIR.parents[3] / "python" / "src" / "ir.cc").read_text()
    base_bound = set(re.findall(r'\.def\(\s*"(create_[a-z0-9_]+)"', base_cpp))

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
    assert used <= bound | base_bound, f"used but not bound: {sorted(used - bound - base_bound)}"
    assert used <= whitelisted | base_bound, (
        f"used but missing from ascend_methods: {sorted(used - whitelisted - base_bound)}")


def test_new_bindings_and_ave_ops_are_present():
    cpp = (_TLA_DIR / "cpp" / "tla_builder.cc").read_text()
    builder_py = (_TLA_DIR / "builder.py").read_text()
    for name in ("create_simd_binary_scalar", "create_simd_vdup", "create_simd_scalar_int", "create_simd_shift",
                 "create_simd_shift_scalar", "create_simd_vpack", "create_simd_pld", "create_simd_pst",
                 "create_simd_vgather", "create_simd_vunpack"):
        assert f'"{name}"' in cpp, f"{name} not bound"
        assert f"'{name}'" in builder_py, f"{name} not whitelisted"
    # each API must map to its own AVE op, not to a reduce/splat stand-in
    for ave_op in ("hivmave::VFDivOp", "hivmave::VFDivFHPOp", "hivmave::VFExpOp", "hivmave::VFBroadcastScalarMaskOp",
                   "hivmave::VFMaxOp", "hivmave::VFMinOp", "hivmave::VMaxUIOp", "hivmave::VMinUIOp",
                   "hivmave::VFMaxsOp", "hivmave::VMaxsUIOp", "hivmave::VFAbsOp", "hivmave::VFNegOp",
                   "hivmave::VFNotOp", "hivmave::VFReluOp", "hivmave::VFAbsDiffOp", "hivmave::VFExpdifOp",
                   "hivmave::VFVpackOp", "hivmave::VFShlOp", "hivmave::VFShrOp", "hivmave::VFShlsOp",
                   "hivmave::VFLRelusOp", "hivmave::VFPReluOp", "hivmave::VFGatherOp", "hivmave::VFExtSIOp",
                   "hivmave::VFExtUIOp"):
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
    for name in ("vdiv", "vmax", "vmin", "vmaxs", "vexp", "vdup", "vdups", "vdupv", "pld", "pst", "vgather", "vld",
                 "vunpack"):
        assert name in defined, f"{name} not defined in simd.py"
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.get_docstring(node), f"{name} has no docstring"
    for name in ("membar", "pge", "vci"):
        assert f'"{name}"' in src


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
    (_buffer(tl.uint8), (_vec(128, tl.uint16), _vec(128, tl.uint16)), None),
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
@pytest.mark.parametrize("masked", (False, True))
def test_vunpack_result_mask_and_selectors(signed, mode, masked):
    sem = _Semantic()
    mask = _mask(64) if masked else None
    src = _vec(256, tl.int8)
    result = simd.vunpack(src, tl.int32, mask, pp="pp3", signed=signed, mode=mode, _semantic=sem)
    assert result.type == simd.vector_type(64, tl.int32)
    assert sem.builder.calls[0] == ("vunpack", (src.handle, "i32", simd.PART.EVEN, "pp3", signed))
    selects = [args for name, args in sem.builder.calls if name == "vsel"]
    assert len(selects) == int(masked)
    if masked:
        assert selects[0][0] is mask.handle
        assert ("scalar", ("i32", 0.0)) in sem.builder.calls
        assert sem.builder.calls[-2][0] == "vsplat"


@pytest.mark.parametrize("kwargs", [
    dict(pp="pp4"),
    dict(part="unknown"),
    dict(part="lower"),
    dict(part="higher"),
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


@pytest.mark.parametrize("dtype", (tl.int8, tl.uint8, tl.float8e4nv, tl.float8e5))
@pytest.mark.parametrize("mask", (None, _mask(256)))
def test_byte_gather_is_explicitly_unsupported(dtype, mask):
    with pytest.raises(ValueError, match="only 16/32-bit"):
        simd.vgather(_buffer(dtype), (_vec(128, tl.uint16), _vec(128, tl.uint16)), mask, _semantic=_Semantic())
