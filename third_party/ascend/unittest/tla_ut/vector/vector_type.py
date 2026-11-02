"""Regression coverage for SIMD value types; IR tests require a TLA-enabled build."""
import pytest
import triton as tr
import triton.language as tl
from triton.language.extra import tla

from utils.tla_test_utils import compile_offline, lower_offline


def test_type_protocol():
    ty = tla.vector_type(tl.constexpr(64), tl.constexpr(tl.float32))
    assert ty == tla.vector_type(64, tl.float32)
    assert ty != tla.vector_type(128, tl.float32)
    assert ty != tl.block_type(tl.float32, [64])
    assert tla.vector_type(64, tl.int32) != tla.vector_type(64, tl.uint32)
    assert tla.vector_type(64, tl.int32).mangle() != tla.vector_type(64, tl.uint32).mangle()
    assert tla.vector_type(64, tl.int1).shape == (64, )
    handle = object()
    value, cursor = ty._unflatten_ir([None, handle], 1)
    assert cursor == 2 and value.handle is handle
    assert value.dtype == tl.float32 and value.shape == (64, )
    assert not isinstance(value, tl.tensor)
    handles = []
    value._flatten_ir(handles)
    assert handles == [handle]
    with pytest.raises(TypeError, match="scalar condition"):
        bool(value)


@pytest.mark.parametrize("lanes", [0, -1, True, 1.5])
def test_invalid_lanes(lanes):
    with pytest.raises(ValueError, match="positive compile-time integer"):
        tla.vector_type(lanes, tl.float32)


@pytest.mark.parametrize("dtype", [tl.pointer_type(tl.float32), tl.block_type(tl.float32, [64]), "fp32"])
def test_invalid_element_type(dtype):
    with pytest.raises(TypeError, match="scalar integer or floating-point"):
        tla.vector_type(64, dtype)


class NoBuilder:
    """Invalid inputs must fail before reaching any builder operation."""

    @property
    def builder(self):
        raise AssertionError("invalid SIMD input reached the builder")


def test_operand_validation():
    sem = NoBuilder()
    vec = tla.vector(None, tla.vector_type(64, tl.float32))
    tensor = tl.tensor(None, tl.block_type(tl.float32, [64]))
    with pytest.raises(TypeError, match="tla.vector"):
        tla.simd.vadd(tensor, tensor, _semantic=sem)
    with pytest.raises(ValueError, match="same-typed"):
        tla.simd.vadd(vec, tla.vector(None, tla.vector_type(128, tl.float32)), _semantic=sem)
    for mask in (vec, tla.vector(None, tla.vector_type(128, tl.int1))):
        with pytest.raises(ValueError, match="mask must be"):
            tla.simd.vsel(mask, vec, vec, _semantic=sem)
    with pytest.raises(TypeError, match="tla.vector"):
        tla.simd.vreduce_max([vec, vec, tensor], _semantic=sem)


@tr.jit
def _vector_control_flow(out, n):
    with tla.scope.simd():
        acc = tla.simd.vfull(0.0, tl.float32, 64)
        step = tla.simd.vfull(1.0, tl.float32, 64)
        for i in range(n):
            acc, mask = tla.simd.vadd(acc, step), tla.simd.vcmp(acc, step, op=tla.simd.CMP.GE)
            acc = tla.simd.vsel(mask, acc, step)
        pair = (acc, step)
        for i in range(n):
            pair = (tla.simd.vadd(pair[0], pair[1]), pair[1])
        acc = pair[0]
        j = 0
        while j < n:
            acc = tla.simd.vadd(acc, step)
            j += 1
        if n > 0:
            acc = tla.simd.vmul(acc, step)
        else:
            acc = tla.simd.vneg(acc)
        narrow = tla.simd.vtruncf(acc, tl.bfloat16)
        acc = tla.simd.vextf(narrow, tl.float32)
        bits = tla.simd.vbitcast(acc, "bf16")
        acc = tla.simd.vbitcast(bits, "f32")
        lo, hi = tla.simd.vintlv(acc, step)
        lo, hi = tla.simd.vdintlv(lo, hi)
        acc = tla.simd.vreduce_max([lo, hi])
        tla.simd.vsts(out[0], acc)


@tr.jit
def _scalar_where(out, cond):
    with tla.scope.simd():
        base = tla.where(cond, 0, 32)  # scalar int1 cond -> arith.select
        tla.simd.vsts(out[0], tla.simd.vci(base), dist="NORM_B32")


@tr.jit
def _scalar_truncmod(out, a, b, DTYPE: tl.constexpr):
    rem = tla.truncmod(a, b)
    with tla.scope.simd():
        tla.simd.vsts(out[0], tla.simd.vdup(rem, DTYPE), dist="NORM_B32")


def test_truncmod_scalar_ir(output_dir=None):
    for kind, dtype, op in (("i32", tl.int32, "arith.remsi"), ("u32", tl.uint32, "arith.remui")):
        module, ctx = compile_offline(_scalar_truncmod, {
            "out": tla.buffer_type((64, ), dtype, "ub"),
            "a": kind,
            "b": kind,
        }, {"DTYPE": dtype}, output_dir)
        text = str(module)
        assert op in text, text
        assert "tensor<" not in text, text
        lower_offline(module, ctx, output_dir)


@tr.jit
def _scalar_sqrt(out, x):
    root = tla.sqrt(x)
    with tla.scope.simd():
        tla.simd.vsts(out[0], tla.simd.vdup(root, tl.float32), dist="NORM_B32")


@tr.jit
def _vector_sqrt(out):
    with tla.scope.simd():
        x = tla.simd.vlds(out[0])
        tla.simd.vsts(out[0], tla.sqrt(x), dist="NORM_B32")


def test_sqrt_ir(output_dir=None):
    for kernel, signature in (
        (_scalar_sqrt, {"out": tla.buffer_type((64, ), tl.float32, "ub"), "x": "fp32"}),
        (_vector_sqrt, {"out": tla.buffer_type((64, ), tl.float32, "ub")}),
    ):
        module, ctx = compile_offline(kernel, signature, {}, output_dir)
        text = str(module)
        assert ("math.sqrt" if kernel is _scalar_sqrt else "ave.hir.vsqrt") in text, text
        lower_offline(module, ctx, output_dir)


def test_scalar_where_ir(output_dir=None):
    # vci materializes an i32 vector, so the UB view must be int32 (f32 here
    # made the vsts dtype mismatch).
    module, ctx = compile_offline(_scalar_where, {"out": tla.buffer_type((64, ), tl.int32, "ub"), "cond": "i1"}, {},
                                  output_dir)
    text = str(module)
    assert "arith.select" in text, text
    assert "tensor<" not in text
    lower_offline(module, ctx, output_dir)


@tr.jit
def _scalar_assume(cond):
    tla.assume(cond)


def test_assume_scalar_ir(output_dir=None):
    module, ctx = compile_offline(_scalar_assume, {"cond": "i1"}, {}, output_dir)
    text = str(module)
    assert "llvm.intr.assume" in text, text
    lowered = lower_offline(module, ctx, output_dir)
    assert "llvm.intr.assume" in str(lowered), str(lowered)


def test_ir(output_dir=None):
    module, ctx = compile_offline(_vector_control_flow, {"out": tla.buffer_type((64, ), tl.float32, "ub"), "n": "i32"},
                                  {}, output_dir)
    text = str(module)
    assert "vector<64xf32>" in text
    assert "vector<64xi1>" in text
    assert "vector<128xbf16>" in text
    assert "tensor<" not in text
    assert "scf.for" in text and "scf.while" in text and "scf.if" in text
    lower_offline(module, ctx, output_dir)
