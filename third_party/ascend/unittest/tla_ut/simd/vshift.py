"""TLA 3-in-1 test: vshl, vshr and vshls masks and sign selection (ZEROING only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32)
SIGNATURE_I = {"X": "*i32", "Y": "*i32", "O": "*i32"}
SIGNATURE_U = {"X": "*u32", "Y": "*u32", "O": "*u32"}
SHIFT = 3


@tla.jit
def vshl_i32(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    y = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((3, N), tl.int32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        b = tla.simd.vlds(y[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshl(a, b))
        tla.simd.vsts(o[1, 0], tla.simd.vshl(a, b, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vshl(a, b, none))
    tla.copy(o, tla.view(O, (3, N)))


@tla.jit
def vshr_i32(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    y = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((3, N), tl.int32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        b = tla.simd.vlds(y[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshr(a, b))
        tla.simd.vsts(o[1, 0], tla.simd.vshr(a, b, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vshr(a, b, none))
    tla.copy(o, tla.view(O, (3, N)))


@tla.jit
def vshr_u32(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    y = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    o = tla.alloc_buffer((3, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        b = tla.simd.vlds(y[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshr(a, b))
        tla.simd.vsts(o[1, 0], tla.simd.vshr(a, b, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vshr(a, b, none))
    tla.copy(o, tla.view(O, (3, N)))


@tla.jit
def vshls_i32(X, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((3, N), tl.int32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshls(a, 3))
        tla.simd.vsts(o[1, 0], tla.simd.vshls(a, 3, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vshls(a, 3, none))
    tla.copy(o, tla.view(O, (3, N)))


@tla.jit
def vshls_tensor_i32(X, O, N: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((N, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        s = tl.full((), 3, tl.int32)
        tla.simd.vsts(o[0], tla.simd.vshls(a, s))
    tla.copy(o, tla.view(O, (N, )))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    for fn, sig, op_name in ((vshl_i32, SIGNATURE_I, "ave.hir.vshl"), (vshr_i32, SIGNATURE_I, "ave.hir.vshr"),
                             (vshr_u32, SIGNATURE_U, "ave.hir.vshr"), (vshls_i32, {"X": "*i32", "O":
                                                                                   "*i32"}, "ave.hir.vshls")):
        module, ctx = compile_offline(fn, sig, constexprs(), output_dir=output_dir)
        text = str(module)
        expected = 1 if fn is vshls_tensor_i32 else 3
        assert text.count(op_name) == expected, (fn.__name__, text.count(op_name))
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        loads = 1 if fn in (vshls_i32, vshls_tensor_i32) else 2
        check_lowered_ir(text, loads=loads, stores=1, buffers=None, extra=(op_name, ))
    print("IR PASS: vshl/vshr/vshls signed, unsigned and masks")


def _ref_rows(value, prefix):
    zero = torch.zeros_like(value)
    # torch.where on CPU has no uint32 kernel; mask through int64 and cast back.
    masked = (value.to(torch.int64) * prefix.to(torch.int64)).to(value.dtype)
    return torch.stack([
        value,
        masked,
        zero,
    ])


def _masks(s):
    idx = torch.arange(s["lanes"])
    return idx < s["active"]


def _launch():
    torch.manual_seed(123)
    s = SHAPE
    prefix = _masks(s)
    x_small = torch.randint(-(1 << 20), 1 << 20, (s["lanes"], ), dtype=torch.int32)
    x_full = torch.randint(-(1 << 31), 1 << 31, (s["lanes"], ), dtype=torch.int32)
    u_full = torch.randint(0, 1 << 32, (s["lanes"], ), dtype=torch.int64).to(torch.uint32)
    b = torch.randint(0, 8, (s["lanes"], ), dtype=torch.int32)

    out_shl = torch.empty((3, s["lanes"]), dtype=torch.int32, device="npu")
    out_shr = torch.empty((3, s["lanes"]), dtype=torch.int32, device="npu")
    out_shru = torch.empty((3, s["lanes"]), dtype=torch.uint32, device="npu")
    out_shls = torch.empty((3, s["lanes"]), dtype=torch.int32, device="npu")
    out_shls_tensor = torch.empty((s["lanes"], ), dtype=torch.int32, device="npu")

    vshl_i32[(1, )](x_small.npu(), b.npu(), out_shl, **constexprs())
    vshr_i32[(1, )](x_full.npu(), b.npu(), out_shr, **constexprs())
    vshr_u32[(1, )](u_full.npu(), b.npu(), out_shru, **constexprs())
    vshls_i32[(1, )](x_small.npu(), out_shls, **constexprs())
    vshls_tensor_i32[(1, )](x_small.npu(), out_shls_tensor, N=s["lanes"])
    torch.npu.synchronize()

    return [
        (out_shl.cpu(), _ref_rows(x_small << b, prefix)),
        (out_shr.cpu(), _ref_rows(x_full >> b, prefix)),
        (out_shru.cpu(), _ref_rows((u_full.to(torch.int64) >> b.to(torch.int64)).to(torch.uint32), prefix)),
        (out_shls.cpu(), _ref_rows(x_small << SHIFT, prefix)),
        (out_shls_tensor.cpu().unsqueeze(0), (x_small << SHIFT).unsqueeze(0)),
    ]


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vshift generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for got, ref in _launch():
        if got.dtype == torch.uint32:
            torch.testing.assert_close(got.view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
        else:
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
    print("VSHIFT OK: vshl/vshr signed+unsigned/vshls constexpr+tensor, ZEROING")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vshift", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
