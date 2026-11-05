"""TLA 3-in-1 test: vpack b16->b8 and b32->b16 for signed/unsigned sources."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE_I16 = dict(n=128)
SHAPE_I32 = dict(n=64)


@tla.jit
def vpack_i16(X, O, N: tl.constexpr, PART: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int16, space="ub")
    o = tla.alloc_buffer((2 * N, ), tl.int8, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        v = tla.simd.vlds(x[0])
        if PART == 0:
            packed = tla.simd.vpack(v, part="part_even")
        else:
            packed = tla.simd.vpack(v, part="part_odd")
        tla.simd.vsts(o[0], packed, dist="NORM_B8")
    tla.copy(o, tla.view(O, (2 * N, )))


@tla.jit
def vpack_u16(X, O, N: tl.constexpr, PART: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint16, space="ub")
    o = tla.alloc_buffer((2 * N, ), tl.uint8, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        v = tla.simd.vlds(x[0])
        if PART == 0:
            packed = tla.simd.vpack(v, part="part_even")
        else:
            packed = tla.simd.vpack(v, part="part_odd")
        tla.simd.vsts(o[0], packed, dist="NORM_B8")
    tla.copy(o, tla.view(O, (2 * N, )))


@tla.jit
def vpack_i32(X, O, N: tl.constexpr, PART: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((2 * N, ), tl.int16, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        v = tla.simd.vlds(x[0])
        if PART == 0:
            packed = tla.simd.vpack(v, part="part_even")
        else:
            packed = tla.simd.vpack(v, part="part_odd")
        tla.simd.vsts(o[0], packed, dist="NORM_B16")
    tla.copy(o, tla.view(O, (2 * N, )))


@tla.jit
def vpack_u32(X, O, N: tl.constexpr, PART: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    o = tla.alloc_buffer((2 * N, ), tl.uint16, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        v = tla.simd.vlds(x[0])
        if PART == 0:
            packed = tla.simd.vpack(v, part="part_even")
        else:
            packed = tla.simd.vpack(v, part="part_odd")
        tla.simd.vsts(o[0], packed, dist="NORM_B16")
    tla.copy(o, tla.view(O, (2 * N, )))


CASES = {
    "i16": (vpack_i16, {"X": "*i16", "O": "*i8"}, SHAPE_I16["n"], torch.int16, torch.int8, torch.uint8),
    "u16": (vpack_u16, {"X": "*u16", "O": "*u8"}, SHAPE_I16["n"], torch.uint16, torch.uint8, torch.uint8),
    "i32": (vpack_i32, {"X": "*i32", "O": "*i16"}, SHAPE_I32["n"], torch.int32, torch.int16, torch.uint16),
    "u32": (vpack_u32, {"X": "*u32", "O": "*u16"}, SHAPE_I32["n"], torch.uint32, torch.uint16, torch.uint16),
}


def _constexprs(n, part):
    return {"N": n, "PART": part}


def test_ir(output_dir=None):
    for kind, (fn, sig, n, _, _, _) in CASES.items():
        for part in (0, 1):
            module, ctx = compile_offline(fn, sig, _constexprs(n, part), output_dir=output_dir)
            assert str(module).count("ave.hir.vpack") == 1
            text = str(lower_offline(module, ctx, output_dir=output_dir))
            check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vpack", ))
    print("IR PASS: vpack i16/u16/i32/u32 part_even/part_odd")


def _reference(x, part, out_view_dtype):
    n = x.numel()
    if x.dtype.itemsize == 2:
        narrowed = x.contiguous().view(torch.uint8)[::2]
    else:
        narrowed = x.contiguous().view(torch.uint16)[::2]
    out = torch.zeros((2 * n, ), dtype=out_view_dtype)
    if part == 0:
        out[:n] = narrowed
    else:
        out[n:] = narrowed
    return out


def _launch():
    torch.manual_seed(2026)
    results = []
    for kind, (fn, _, n, torch_in, torch_out, view_dtype) in CASES.items():
        if torch_in is torch.int16:
            x = torch.randint(-(1 << 15), 1 << 15, (n, ), dtype=torch_in)
        elif torch_in is torch.uint16:
            x = torch.randint(0, 1 << 16, (n, ), dtype=torch.int64).to(torch_in)
        elif torch_in is torch.int32:
            x = torch.randint(-(1 << 31), 1 << 31, (n, ), dtype=torch_in)
        else:
            x = torch.randint(0, 1 << 32, (n, ), dtype=torch.int64).to(torch_in)
        for part in (0, 1):
            out = torch.empty((2 * n, ), dtype=torch_out, device="npu")
            fn[(1, )](x.npu(), out, **_constexprs(n, part))
            torch.npu.synchronize()
            got = out.cpu().view(view_dtype)
            ref = _reference(x, part, view_dtype)
            results.append((kind, part, got, ref))
    return results


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vpack generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for kind, part, got, ref in _launch():
        n = len(ref) // 2
        sl = slice(0, n) if part == 0 else slice(n, 2 * n)
        # The unselected half is not part of the AVE contract; only the
        # selected half is meaningful for a single part_even/part_odd call.
        torch.testing.assert_close(got[sl], ref[sl], rtol=0, atol=0)
    print("VPACK OK: signed/unsigned b16->b8 and b32->b16, both parts")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vpack", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
