"""A5 MMAD L0 frontend contracts."""
import importlib
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("triton")
import triton.language as tl

TLA_DIR = Path(__file__).resolve().parents[3] / "language" / "tla"
if "mmad_l0_tla_src" not in sys.modules:
    package = types.ModuleType("mmad_l0_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["mmad_l0_tla_src"] = package
cube = importlib.import_module("mmad_l0_tla_src.cube")
core = importlib.import_module("mmad_l0_tla_src.core")
dma = importlib.import_module("mmad_l0_tla_src.dma")


class Recorder:

    def __init__(self):
        self.calls = []

    def create_hivm_mmadl0(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class Semantic:

    def __init__(self):
        self.builder = Recorder()


def buf(shape, space, dtype, strides=None, offset=0):
    return core.buffer(object(), core.buffer_type(shape, dtype, space, strides, offset))


def operands(m=32, n=64, k=48, dtype=tl.float16):
    return dict(a=buf((k // 16, m // 16, 16, 16), "l0a", dtype), b=buf(
        (k // 16, n // 16, 16, 16), "l0b", dtype), acc=buf((n // 16, m // 16, 16, 16), "l0c", tl.float32), m=m,
                n=n, k=k)


@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("unit", [None, 0, dma.UnitFlagMode.DISABLED])
@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
def test_native_order_and_defaults(clear, unit, dtype):
    args = operands(dtype=dtype)
    sem = Semantic()
    result = cube.mmad_l0(**args, clear_accum=tl.constexpr(clear), unit_flag_ctrl=tl.constexpr(unit), _semantic=sem)
    assert result is None
    assert sem.builder.calls == [((args["a"].handle, args["b"].handle, args["acc"].handle),
                                  dict(init_condition=clear, real_m=32, real_k=48, real_n=64, bias=None,
                                       disable_gemv=True, unit_flag_ctrl=None if unit is None else 0))]


def test_runtime_i1_is_forwarded_without_python_bool_conversion():
    clear = tl.tensor(object(), tl.int1)
    sem = Semantic()
    cube.mmad_l0(**operands(), clear_accum=clear, _semantic=sem)
    assert sem.builder.calls[0][1]["init_condition"] is clear.handle


@pytest.mark.parametrize("axis", ["m", "n", "k"])
@pytest.mark.parametrize("extent", [0, -16, 17, 2064, True, 32.0, "32", None])
def test_invalid_extents_fail_before_emission(axis, extent):
    args = operands()
    args[axis] = tl.constexpr(extent)
    sem = Semantic()
    with pytest.raises(ValueError, match=axis + " must be a compile-time"):
        cube.mmad_l0(**args, _semantic=sem)
    assert not sem.builder.calls


def test_runtime_extent_is_rejected():
    args = operands()
    args["k"] = tl.tensor(object(), tl.int32)
    with pytest.raises(ValueError, match="k must be a compile-time"):
        cube.mmad_l0(**args, _semantic=Semantic())


@pytest.mark.parametrize("name", ["a", "b", "acc"])
@pytest.mark.parametrize("fault", ["operand", "space", "dtype", "shape", "strides"])
def test_invalid_buffers_fail_before_emission(name, fault):
    args = operands()
    original = args[name]
    if fault == "operand":
        args[name] = object()
    else:
        shape, space, dtype, strides = original.shape, original.space, original.dtype, original.strides
        if fault == "space":
            space = "l1"
        elif fault == "dtype":
            dtype = tl.int8
        elif fault == "shape":
            shape = (shape[0] * 2, ) + shape[1:]
        else:
            strides = (strides[0] * 2, ) + strides[1:]
        args[name] = buf(shape, space, dtype, strides)
    sem = Semantic()
    with pytest.raises((TypeError, ValueError), match=name):
        cube.mmad_l0(**args, _semantic=sem)
    assert not sem.builder.calls


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
def test_mixed_operand_dtypes_are_rejected(dtype):
    args = operands(dtype=dtype)
    other = tl.bfloat16 if dtype == tl.float16 else tl.float16
    args["b"] = buf(args["b"].shape, "l0b", other)
    sem = Semantic()
    with pytest.raises((TypeError, ValueError), match="b"):
        cube.mmad_l0(**args, _semantic=sem)
    assert not sem.builder.calls


def test_l1_b_layout_is_not_accepted_as_l0b():
    args = operands()
    args["b"] = buf((4, 3, 16, 16), "l0b", tl.float16)
    with pytest.raises(ValueError, match="b must have exact physical shape"):
        cube.mmad_l0(**args, _semantic=Semantic())


def test_singleton_dimension_still_requires_compact_stride():
    args = operands(m=16, n=16, k=16)
    args["a"] = buf((1, 1, 16, 16), "l0a", tl.float16, strides=(512, 256, 16, 1))
    with pytest.raises(ValueError, match="compact physical strides"):
        cube.mmad_l0(**args, _semantic=Semantic())


@pytest.mark.parametrize("name,offset", [("a", 256), ("b", 512), ("acc", 256), ("a", None), ("b", None), ("acc", None)])
def test_compact_offsets_keep_the_original_buffer_handle(name, offset):
    args = operands()
    original = args[name]
    args[name] = buf(original.shape, original.space, original.dtype, offset=offset)
    sem = Semantic()
    cube.mmad_l0(**args, _semantic=sem)
    positional, _ = sem.builder.calls[0]
    emitted = positional[("a", "b", "acc").index(name)]
    assert emitted is args[name].handle


@pytest.mark.parametrize("name", ["a", "b", "acc"])
@pytest.mark.parametrize("offset", [-256, 1, 1 << 20])
def test_invalid_static_offset_is_rejected(name, offset):
    args = operands()
    original = args[name]
    args[name] = buf(original.shape, original.space, original.dtype, offset=offset)
    with pytest.raises(ValueError, match=name + " offset"):
        cube.mmad_l0(**args, _semantic=Semantic())


@pytest.mark.parametrize("m,n,k,name", [(2048, 16, 32, "a"), (16, 2048, 32, "b"), (512, 256, 16, "acc")])
def test_tile_capacity_is_checked(m, n, k, name):
    args = operands(m, n, k)
    with pytest.raises(ValueError, match=name + " exceeds"):
        cube.mmad_l0(**args, _semantic=Semantic())


@pytest.mark.parametrize(
    "clear", [0, 1, None, "false",
              tl.tensor(object(), tl.int32),
              tl.tensor(object(), tl.block_type(tl.int1, [1]))])
def test_clear_requires_bool_or_scalar_i1(clear):
    with pytest.raises(TypeError, match="clear_accum"):
        cube.mmad_l0(**operands(), clear_accum=clear, _semantic=Semantic())


@pytest.mark.parametrize("unit", [1, 2, 3, -1, True])
def test_unsupported_unit_flag_is_rejected(unit):
    with pytest.raises(ValueError, match="unit_flag_ctrl"):
        cube.mmad_l0(**operands(), unit_flag_ctrl=unit, _semantic=Semantic())


@pytest.mark.parametrize("disable", [False, 0, None])
def test_a5_requires_gemv_disabled(disable):
    with pytest.raises((TypeError, ValueError), match="disable_gemv"):
        cube.mmad_l0(**operands(), disable_gemv=disable, _semantic=Semantic())
