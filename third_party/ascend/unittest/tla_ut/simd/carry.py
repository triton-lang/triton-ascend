"""Tests for AVE carry/borrow arithmetic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, get_stages, lower_offline, check_compile_artifacts)

MASK32 = 0xFFFFFFFF


@tla.jit
def vaddcs_kernel(X, Y, C, M, D, B, T: tl.constexpr):
    x = tla.alloc_buffer((64, ), T, space="ub")
    y = tla.alloc_buffer((64, ), T, space="ub")
    c = tla.alloc_buffer((64, ), tl.int32, space="ub")
    m = tla.alloc_buffer((64, ), tl.int32, space="ub")
    d = tla.alloc_buffer((64, ), T, space="ub")
    b = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    tla.copy(tla.view(Y, (64, )), y)
    tla.copy(tla.view(C, (64, )), c)
    tla.copy(tla.view(M, (64, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        carry = tla.simd.vcmps(tla.simd.vlds(c[0]), 0, op=tla.simd.CMP.NE)
        total, carry_out = tla.simd.vaddcs(tla.simd.vlds(x[0]), tla.simd.vlds(y[0]), carry, mask)
        tla.simd.vsts(d[0], total)
        bits = tla.simd.vsel(carry_out, tla.simd.vdup(1, tl.int32), tla.simd.vdup(0, tl.int32))
        tla.simd.vsts(b[0], bits)
    tla.copy(d, tla.view(D, (64, )))
    tla.copy(b, tla.view(B, (64, )))


@tla.jit
def vsubcs_kernel(X, Y, C, M, D, B, T: tl.constexpr):
    x = tla.alloc_buffer((64, ), T, space="ub")
    y = tla.alloc_buffer((64, ), T, space="ub")
    c = tla.alloc_buffer((64, ), tl.int32, space="ub")
    m = tla.alloc_buffer((64, ), tl.int32, space="ub")
    d = tla.alloc_buffer((64, ), T, space="ub")
    b = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    tla.copy(tla.view(Y, (64, )), y)
    tla.copy(tla.view(C, (64, )), c)
    tla.copy(tla.view(M, (64, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        borrow = tla.simd.vcmps(tla.simd.vlds(c[0]), 0, op=tla.simd.CMP.NE)
        diff, borrow_out = tla.simd.vsubcs(tla.simd.vlds(x[0]), tla.simd.vlds(y[0]), borrow, mask)
        tla.simd.vsts(d[0], diff)
        bits = tla.simd.vsel(borrow_out, tla.simd.vdup(1, tl.int32), tla.simd.vdup(0, tl.int32))
        tla.simd.vsts(b[0], bits)
    tla.copy(d, tla.view(D, (64, )))
    tla.copy(b, tla.view(B, (64, )))


CASES = [
    ("addcs", vaddcs_kernel, {"X": "*i32", "Y": "*i32", "C": "*i32", "M": "*i32", "D": "*i32", "B":
                              "*i32"}, ("ave.hir.vaddcs", )),
    ("subcs", vsubcs_kernel, {"X": "*i32", "Y": "*i32", "C": "*i32", "M": "*i32", "D": "*i32", "B":
                              "*i32"}, ("ave.hir.vsubcs", )),
]


def test_ir():
    for dtype, short in [(tl.int32, "i32"), (tl.uint32, "u32")]:
        for name, fn, signature, ops in CASES:
            sig = {key: val.replace("i32", short) for key, val in signature.items()}
            mod, ctx = compile_offline(fn, sig, {"T": dtype})
            text = str(lower_offline(mod, ctx))
            for op in ops:
                assert op in text, f"{name} did not lower to {op}"
    print("IR PASS: vaddcs/vsubcs")


def _u32(x):
    return x.to(torch.int64) & MASK32


def _reference(name, x, y, carry, mask):
    ux, uy = _u32(x), _u32(y)
    active = mask.bool()
    cin = carry.bool()
    if name == "addcs":
        partial = (ux + uy) & MASK32
        carry0 = (ux + uy) > MASK32
        value = (partial + cin.to(torch.int64)) & MASK32
        carry1 = (partial == MASK32) & cin
        flag = (carry0 | carry1) & active
    else:
        partial = (ux - uy) & MASK32
        borrow0 = ux < uy
        value = (partial - cin.to(torch.int64)) & MASK32
        borrow1 = partial < cin.to(torch.int64)
        flag = (borrow0 | borrow1) & active
    value = torch.where(active, value, torch.zeros_like(value))
    return value.to(torch.int32), flag.to(torch.int32)


def _run(fn, x, y, carry, mask):
    d = torch.empty(64, dtype=torch.int32, device="npu")
    b = torch.empty(64, dtype=torch.int32, device="npu")
    args = [x.npu(), y.npu(), carry.to(torch.int32).npu(), mask.to(torch.int32).npu(), d, b]
    fn[(1, )](*args, T=tl.int32)
    torch.npu.synchronize()
    return d.cpu(), b.cpu()


def _launch():
    _, fn, _, _ = CASES[0]
    x = torch.arange(64, dtype=torch.int32)
    y = torch.arange(64, dtype=torch.int32) + 1
    carry = torch.zeros(64, dtype=torch.int32)
    mask = torch.ones(64, dtype=torch.int32)
    return _run(fn, x, y, carry, mask)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: carry generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    torch.manual_seed(11)
    edge = torch.tensor([0, 1, -1, -2, 0x7FFFFFFF, -0x80000000, 0x7FFFFFFE, -0x7FFFFFFF], dtype=torch.int32)
    x = torch.cat([edge, torch.randint(-(1 << 31), 1 << 31, (64 - edge.numel(), ), dtype=torch.int32)])
    y = torch.cat([edge.flip(0), torch.randint(-(1 << 31), 1 << 31, (64 - edge.numel(), ), dtype=torch.int32)])
    masks = [torch.ones(64, dtype=torch.int32), (torch.arange(64) % 3 != 0).to(torch.int32)]
    carries = [
        torch.zeros(64, dtype=torch.int32),
        torch.ones(64, dtype=torch.int32),
        (torch.arange(64) % 2).to(torch.int32),
    ]
    for name, fn, _, _ in CASES:
        for mask in masks:
            for carry in carries:
                actual_d, actual_b = _run(fn, x, y, carry, mask)
                ref_d, ref_b = _reference(name, x, y, carry, mask)
                torch.testing.assert_close(actual_d, ref_d, rtol=0, atol=0)
                torch.testing.assert_close(actual_b, ref_b, rtol=0, atol=0)
        print("RUN PASS:", name)
    print("RUN PASS: vaddcs/vsubcs all")


if __name__ == "__main__":
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
