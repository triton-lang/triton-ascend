"""TLA 3-in-1 test: vshl/vshr/vshls on i8/u8/i16/u16 (ZEROING only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPES = {"i8": (256, 128), "u8": (256, 128), "i16": (128, 64), "u16": (128, 64)}
SIGS = {
    "i8": {"X": "*i8", "R": "*i8", "Y": "*i8", "O": "*i8"},
    "u8": {"X": "*u8", "R": "*u8", "Y": "*u8", "O": "*u8"},
    "i16": {"X": "*i16", "R": "*i16", "Y": "*i16", "O": "*i16"},
    "u16": {"X": "*u16", "R": "*u16", "Y": "*u16", "O": "*u16"},
}


@tla.jit
def vshift_i8(X, R, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int8, space="ub")
    r = tla.alloc_buffer((N, ), tl.int8, space="ub")
    y = tla.alloc_buffer((N, ), tl.int8, space="ub")
    o = tla.alloc_buffer((9, N), tl.int8, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(R, (N, )), r)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        rv = tla.simd.vlds(r[0])
        yv = tla.simd.vlds(y[0])
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(xv, xv, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshl(xv, yv), dist="NORM_B8")
        tla.simd.vsts(o[1, 0], tla.simd.vshl(xv, yv, prefix), dist="NORM_B8")
        tla.simd.vsts(o[2, 0], tla.simd.vshl(xv, yv, none), dist="NORM_B8")
        tla.simd.vsts(o[3, 0], tla.simd.vshr(rv, yv), dist="NORM_B8")
        tla.simd.vsts(o[4, 0], tla.simd.vshr(rv, yv, prefix), dist="NORM_B8")
        tla.simd.vsts(o[5, 0], tla.simd.vshr(rv, yv, none), dist="NORM_B8")
        tla.simd.vsts(o[6, 0], tla.simd.vshls(xv, 3), dist="NORM_B8")
        tla.simd.vsts(o[7, 0], tla.simd.vshls(xv, 3, prefix), dist="NORM_B8")
        tla.simd.vsts(o[8, 0], tla.simd.vshls(xv, 3, none), dist="NORM_B8")
    tla.copy(o, tla.view(O, (9, N)))


@tla.jit
def vshift_u8(X, R, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint8, space="ub")
    r = tla.alloc_buffer((N, ), tl.uint8, space="ub")
    y = tla.alloc_buffer((N, ), tl.uint8, space="ub")
    o = tla.alloc_buffer((9, N), tl.uint8, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(R, (N, )), r)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        rv = tla.simd.vlds(r[0])
        yv = tla.simd.vlds(y[0])
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(xv, xv, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshl(xv, yv), dist="NORM_B8")
        tla.simd.vsts(o[1, 0], tla.simd.vshl(xv, yv, prefix), dist="NORM_B8")
        tla.simd.vsts(o[2, 0], tla.simd.vshl(xv, yv, none), dist="NORM_B8")
        tla.simd.vsts(o[3, 0], tla.simd.vshr(rv, yv), dist="NORM_B8")
        tla.simd.vsts(o[4, 0], tla.simd.vshr(rv, yv, prefix), dist="NORM_B8")
        tla.simd.vsts(o[5, 0], tla.simd.vshr(rv, yv, none), dist="NORM_B8")
        tla.simd.vsts(o[6, 0], tla.simd.vshls(xv, 3), dist="NORM_B8")
        tla.simd.vsts(o[7, 0], tla.simd.vshls(xv, 3, prefix), dist="NORM_B8")
        tla.simd.vsts(o[8, 0], tla.simd.vshls(xv, 3, none), dist="NORM_B8")
    tla.copy(o, tla.view(O, (9, N)))


@tla.jit
def vshift_i16(X, R, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int16, space="ub")
    r = tla.alloc_buffer((N, ), tl.int16, space="ub")
    y = tla.alloc_buffer((N, ), tl.int16, space="ub")
    o = tla.alloc_buffer((9, N), tl.int16, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(R, (N, )), r)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        rv = tla.simd.vlds(r[0])
        yv = tla.simd.vlds(y[0])
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(xv, xv, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshl(xv, yv), dist="NORM_B16")
        tla.simd.vsts(o[1, 0], tla.simd.vshl(xv, yv, prefix), dist="NORM_B16")
        tla.simd.vsts(o[2, 0], tla.simd.vshl(xv, yv, none), dist="NORM_B16")
        tla.simd.vsts(o[3, 0], tla.simd.vshr(rv, yv), dist="NORM_B16")
        tla.simd.vsts(o[4, 0], tla.simd.vshr(rv, yv, prefix), dist="NORM_B16")
        tla.simd.vsts(o[5, 0], tla.simd.vshr(rv, yv, none), dist="NORM_B16")
        tla.simd.vsts(o[6, 0], tla.simd.vshls(xv, 3), dist="NORM_B16")
        tla.simd.vsts(o[7, 0], tla.simd.vshls(xv, 3, prefix), dist="NORM_B16")
        tla.simd.vsts(o[8, 0], tla.simd.vshls(xv, 3, none), dist="NORM_B16")
    tla.copy(o, tla.view(O, (9, N)))


@tla.jit
def vshift_u16(X, R, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint16, space="ub")
    r = tla.alloc_buffer((N, ), tl.uint16, space="ub")
    y = tla.alloc_buffer((N, ), tl.uint16, space="ub")
    o = tla.alloc_buffer((9, N), tl.uint16, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(R, (N, )), r)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        rv = tla.simd.vlds(r[0])
        yv = tla.simd.vlds(y[0])
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(xv, xv, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vshl(xv, yv), dist="NORM_B16")
        tla.simd.vsts(o[1, 0], tla.simd.vshl(xv, yv, prefix), dist="NORM_B16")
        tla.simd.vsts(o[2, 0], tla.simd.vshl(xv, yv, none), dist="NORM_B16")
        tla.simd.vsts(o[3, 0], tla.simd.vshr(rv, yv), dist="NORM_B16")
        tla.simd.vsts(o[4, 0], tla.simd.vshr(rv, yv, prefix), dist="NORM_B16")
        tla.simd.vsts(o[5, 0], tla.simd.vshr(rv, yv, none), dist="NORM_B16")
        tla.simd.vsts(o[6, 0], tla.simd.vshls(xv, 3), dist="NORM_B16")
        tla.simd.vsts(o[7, 0], tla.simd.vshls(xv, 3, prefix), dist="NORM_B16")
        tla.simd.vsts(o[8, 0], tla.simd.vshls(xv, 3, none), dist="NORM_B16")
    tla.copy(o, tla.view(O, (9, N)))


KERNELS = {
    "i8": vshift_i8,
    "u8": vshift_u8,
    "i16": vshift_i16,
    "u16": vshift_u16,
}


def constexprs(kind):
    lanes, active = SHAPES[kind]
    return {"N": lanes, "ACTIVE": active}


def test_ir(output_dir=None):
    for kind, fn in KERNELS.items():
        module, ctx = compile_offline(fn, SIGS[kind], constexprs(kind), output_dir=output_dir)
        text = str(module)
        assert text.count("ave.hir.vshl ") == 3
        assert text.count("ave.hir.vshr ") == 3
        assert text.count("ave.hir.vshls ") == 3
        assert text.count("ave.hir.vsel") == 0
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        check_lowered_ir(text, loads=3, stores=1, buffers=None, extra=("ave.hir.vshl", "ave.hir.vshr", "ave.hir.vshls"))
    print("IR PASS: vshl/vshr/vshls i8/u8/i16/u16")


def _rows(bits, prefix):
    zero = torch.zeros_like(bits)
    return torch.stack([bits, torch.where(prefix, bits, zero), zero])


def _bits(t):
    return t.view(torch.uint8) if t.dtype.itemsize == 1 else t.view(torch.uint16)


def _ref(kind, x_left, r_full, b):
    width = x_left.dtype.itemsize * 8
    mask = (1 << width) - 1
    bit_dtype = torch.uint8 if width == 8 else torch.uint16
    left = ((x_left.to(torch.int64) << b.to(torch.int64)) & mask).to(bit_dtype)
    shls = ((x_left.to(torch.int64) << 3) & mask).to(bit_dtype)
    if kind in ("i8", "i16"):
        right = _bits(r_full >> b)
    else:
        right = (r_full.to(torch.int64) >> b.to(torch.int64)).to(bit_dtype)
    prefix = torch.arange(x_left.numel()) < SHAPES[kind][1]
    return torch.cat([_rows(left, prefix), _rows(right, prefix), _rows(shls, prefix)])


def _launch():
    torch.manual_seed(123)
    results = []
    for kind, fn in KERNELS.items():
        lanes, _ = SHAPES[kind]
        width = 8 if kind in ("i8", "u8") else 16
        x_left = torch.randint(0, 8, (lanes, ), dtype=torch.int64)
        if width == 8:
            r_full = torch.randint(-(1 << 7), 1 << 7, (lanes, ), dtype=torch.int64)
        else:
            r_full = torch.randint(-(1 << 15), 1 << 15, (lanes, ), dtype=torch.int64)
        b_cpu = (torch.arange(lanes) % width).to(torch.int64)
        if kind.startswith("u"):
            x_t = x_left.to(torch.uint8 if width == 8 else torch.uint16)
            r_t = (r_full & ((1 << width) - 1)).to(torch.uint8 if width == 8 else torch.uint16)
            b_t = b_cpu.to(torch.uint8 if width == 8 else torch.uint16)
        else:
            x_t = x_left.to(torch.int8 if width == 8 else torch.int16)
            r_t = r_full.to(torch.int8 if width == 8 else torch.int16)
            b_t = b_cpu.to(torch.int8 if width == 8 else torch.int16)
        out = torch.empty((9, lanes), dtype=x_t.dtype, device="npu")
        fn[(1, )](x_t.npu(), r_t.npu(), b_t.npu(), out, **constexprs(kind))
        torch.npu.synchronize()
        got = _bits(out.cpu())
        ref = _ref(kind, x_t, r_t, b_t)
        results.append((kind, got, ref))
    return results


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vshift narrow generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for kind, got, ref in _launch():
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    print("VSHIFT NARROW OK: i8/u8/i16/u16, ZEROING and sign selection")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vshift_narrow", "ir" if do_ir else "",
          "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
