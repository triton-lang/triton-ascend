"""Host-only shape/alias contracts for the FlashAttention TLA primitives."""
import importlib
import pathlib
import sys
import types

import pytest

pytest.importorskip("triton")
import triton.language as tl

TLA_DIR = pathlib.Path(__file__).resolve().parents[2] / "language" / "tla"
if "fa_tla_src" not in sys.modules:
    package = types.ModuleType("fa_tla_src")
    package.__path__ = [str(TLA_DIR)]
    sys.modules["fa_tla_src"] = package
cube = importlib.import_module("fa_tla_src.cube")
view = importlib.import_module("fa_tla_src.view")
core = importlib.import_module("fa_tla_src.core")


class Recorder:

    def create_hivm_mmadl1(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def get_bf16_ty(self):
        return "bf16"

    def create_tla_buffer_type(self, *args):
        return args

    def create_tla_reinterpret_cast(self, *args):
        self.alias = args
        return "alias"


class Semantic:

    def __init__(self):
        self.builder = Recorder()

    def to_tensor(self, value):
        return types.SimpleNamespace(handle=value)


def buf(shape, space="l1", **kwargs):
    dtype = tl.float32 if space == "l0c" else tl.bfloat16
    return core.buffer(object(), core.buffer_type(shape, dtype, space, **kwargs))


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("nz", [False, True])
def test_transpose_checks_physical_shape_and_emits_flags(ta, tb, nz):

    def shape(rows, cols):
        return (cols // 16, rows // 16, 16, 16) if nz else (rows, cols)

    a = buf(shape(64, 32) if ta else shape(32, 64))
    b = buf(shape(128, 64) if tb else shape(64, 128))
    acc = buf(shape(32, 128), "l0c")
    sem = Semantic()
    cube.mmad(a, b, acc, transpose_a=ta, transpose_b=tb, clear_accum=True, _semantic=sem)
    assert sem.builder.kwargs == dict(init_condition=True, real_m=32, real_k=64, real_n=128, transpose_a=ta,
                                      transpose_b=tb, unit_flag_ctrl=None, unit_flag_group_id=None)


def test_partial_tile_uses_actual_l0c_shape():
    sem = Semantic()
    cube.mmad(buf((64, 128)), buf((64, 256)), buf((32, 64), "l0c"), transpose_a=True, real_m=32, real_k=16, real_n=64,
              _semantic=sem)
    assert [sem.builder.kwargs[k] for k in ("real_m", "real_k", "real_n")] == [32, 16, 64]
    with pytest.raises(ValueError, match="acc must"):
        cube.mmad(buf((64, 128)), buf((128, 64)), buf((64, 64), "l0c"), real_m=32, _semantic=sem)


@pytest.mark.parametrize("extent", [0, -16, 17, 80, True, 16.0])
def test_partial_extent_must_be_aligned_and_bounded(extent):
    with pytest.raises(ValueError, match="real_m"):
        cube.mmad(buf((64, 128)), buf((128, 64)), buf((64, 64), "l0c"), real_m=extent, _semantic=Semantic())


def test_bad_reduction_dimension_is_not_hidden_by_real_k():
    with pytest.raises(ValueError, match="reduction"):
        cube.mmad(buf((64, 128)), buf((256, 64)), buf((64, 64), "l0c"), real_k=32, _semantic=Semantic())


def test_prefix_alias_preserves_origin_and_changes_pitch():
    src = buf((8, 2064), "ub")
    sem = Semantic()
    dst = view.prefix_view(src, (8, 1040), _semantic=sem)
    assert sem.builder.alias[:4] == (src.handle, 0, [8, 1040], [1040, 1])
    assert dst.space == "ub" and dst.dtype == tl.bfloat16 and dst.offset is None
    with pytest.raises(ValueError, match="zero-offset"):
        view.prefix_view(dst, (8, 512), _semantic=sem)


@pytest.mark.parametrize("kwargs", [{"offset": 1}, {"offset": None}, {"strides": (65, 1)}])
def test_prefix_alias_rejects_non_identity_origin(kwargs):
    with pytest.raises(ValueError, match="zero-offset"):
        view.prefix_view(buf((64, 64), "ub", **kwargs), (32, 64), _semantic=Semantic())


def test_prefix_alias_rejects_oversize():
    with pytest.raises(ValueError, match="fit"):
        view.prefix_view(buf((64, 64), "ub"), (65, 64), _semantic=Semantic())
