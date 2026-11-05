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
"""tla.alloc address-space contract: IR, compilation and device checks.

Each backend-supported allocation space gets a real kernel that reads and
writes it, so the allocation is not dead code:

* ``ub``  -- GM -> UB -> GM round trip, ``versions=1`` and ``versions=2``.
* ``l1``  -- GM ND -> L1 NZ -> UB ND -> GM round trip (``hivm.hir.nd2nz`` +
  ``hivm.hir.l12ub``).
* ``l0c`` -- Cube GEMM with an L0C accumulator, then fixpipe back to GM.

The rejection contract for ``gm``/``l0a``/``l0b``/``bt``/``fb`` and unknown
spaces is covered by ``alloc_frontend_contract.py`` (host only).  The fuller
Cube regression for L0C (accumulate, unit flags, bias) stays in
``cube/gemm_basic.py``.

``versions=2`` is only asserted at frontend-IR level for UB: that is the only
space the existing multi-buffer tests exercise, so this file does not assume
the backend supports multi-buffering for L1/L0C.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary, matrix

UB_N = 256
L1_M, L1_N = 16, 32
MM_M, MM_N, MM_K = 16, 16, 16

CONFIGS = [
    dict(name="ub_v1", space="ub", versions=1, launch=True, kernel="ub"),
    dict(name="ub_enum", space="ub", versions=1, launch=True, kernel="ub_enum"),
    dict(name="ub_v2", space="ub", versions=2, launch=False, kernel="ub"),
    dict(name="l1", space="l1", versions=1, launch=True, kernel="l1"),
    dict(name="l0c", space="l0c", versions=1, launch=True, kernel="l0c"),
]


@tla.jit
def ub_roundtrip(X, Y, N: tl.constexpr, VERSIONS: tl.constexpr):
    with tla.scope.vector():
        buf = tla.alloc((N, ), tl.float32, space="ub", versions=VERSIONS)
        tla.load_gm_to_ub(tla.view(X, (N, )), buf)
        tla.store_ub_to_gm(buf, tla.view(Y, (N, )))


@tla.jit
def ub_roundtrip_enum(X, Y, N: tl.constexpr):
    """Same as ``ub_roundtrip`` but through the public ``tla.AddressSpace``."""
    with tla.scope.vector():
        buf = tla.alloc((N, ), tl.float32, space=tla.AddressSpace.UB)
        tla.load_gm_to_ub(tla.view(X, (N, )), buf)
        tla.store_ub_to_gm(buf, tla.view(Y, (N, )))


@tla.jit
def l1_roundtrip(X, Y, M: tl.constexpr, N: tl.constexpr):
    out = tla.alloc((M, N), tl.float16, space="ub")
    with tla.scope.cube():
        l1 = tla.alloc((N // 16, M // 16, 16, 16), tl.float16, space="l1")
        tla.load_gm_to_l1(tla.view(X, (M, N)), l1)
        tla.copy_l1_to_ub(l1, out)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 0)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        # L12UB writes the first AIV's UB; the second AIV must not overwrite GM.
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out, tla.view(Y, (M, N)))


@tla.jit
def l0c_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and K % 16 == 0)
    with tla.scope.cube():
        a = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (M, K)), a)
        tla.load_gm_to_l1(tla.view(B, (K, N)), b)
        tla.cube.mmad(a, b, acc, clear_accum=True)
        tla.store_l0c_to_gm(acc, tla.view(C, (M, N)), tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


def compile_case(case):
    """Return (module, context) for the requested space."""
    if case["kernel"] == "ub_enum":
        return compile_offline(ub_roundtrip_enum, {"X": "*fp32", "Y": "*fp32"}, {"N": UB_N})
    if case["space"] == "ub":
        return compile_offline(ub_roundtrip, {"X": "*fp32", "Y": "*fp32"}, {"N": UB_N, "VERSIONS": case["versions"]})
    if case["space"] == "l1":
        return compile_offline(l1_roundtrip, {"X": "*fp16", "Y": "*fp16"}, {"M": L1_M, "N": L1_N})
    if case["space"] == "l0c":
        return compile_offline(l0c_gemm, {"A": "*fp16", "B": "*fp16", "C": "*fp32"}, {"M": MM_M, "N": MM_N, "K": MM_K})
    raise AssertionError(f"unknown case space {case['space']!r}")


def ir_expectations(case):
    """(memref.alloc count, required address spaces, multi_buffer marks)."""
    if case["space"] == "ub":
        return 1, ("#hivm.address_space<ub>", ), 1 if case["versions"] > 1 else 0
    if case["space"] == "l1":
        return 2, ("#hivm.address_space<ub>", "#hivm.address_space<cbuf>"), 0
    return 3, ("#hivm.address_space<cbuf>", "#hivm.address_space<cc>"), 0


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_case(case)
    ast = str(module)
    allocs, spaces, marks = ir_expectations(case)
    assert ast.count("memref.alloc()") == allocs, "one memref.alloc per tla.alloc"
    for space in spaces:
        assert space in ast, space
    # versions=1 opts out: no annotation.mark is emitted at all.
    assert ast.count("hivm.multi_buffer") == marks, "unexpected multi_buffer annotation count"
    if marks:
        assert "hivm.multi_buffer = 2 : i32" in ast

    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert text.count("memref.alloc()") == allocs, "allocation must survive lowering"
    for space in spaces:
        assert space in text, space
    assert "!tt.ptr" not in text and "tensor<" not in text and "bufferization." not in text
    print("IR PASS: alloc", case["name"])


def _launch(case, seed=7):
    generator = torch.Generator().manual_seed(seed)
    if case["kernel"] == "ub_enum":
        x = torch.randint(-127, 128, (UB_N, ), generator=generator).to(torch.float32)
        y = torch.full_like(x, float("nan"))
        args = (x.npu(), y.npu())
        return ub_roundtrip_enum, args, {"N": UB_N}
    if case["space"] == "ub":
        x = torch.randint(-127, 128, (UB_N, ), generator=generator).to(torch.float32)
        y = torch.full_like(x, float("nan"))
        args = (x.npu(), y.npu())
        return ub_roundtrip, args, {"N": UB_N, "VERSIONS": case["versions"]}
    if case["space"] == "l1":
        x = torch.randint(-127, 128, (L1_M, L1_N), generator=generator).to(torch.float16)
        y = torch.full_like(x, float("nan"))
        args = (x.npu(), y.npu())
        return l1_roundtrip, args, {"M": L1_M, "N": L1_N}
    a = torch.randn((MM_M, MM_K), generator=generator, dtype=torch.float16)
    b = torch.randn((MM_K, MM_N), generator=generator, dtype=torch.float16)
    out = torch.full((MM_M, MM_N), float("nan"), dtype=torch.float32)
    args = (a.npu(), b.npu(), out.npu())
    return l0c_gemm, args, {"M": MM_M, "N": MM_N, "K": MM_K}


@matrix(CONFIGS)
def test_compile(case):
    if not case["launch"]:
        # versions=2 contract is frontend-only until a later multi-buffer
        # regression proves backend support for this shape.
        pytest.skip("versions>1 frontend contract only")
    kernel, args, consts = _launch(case)
    check_kernel_binary(kernel.warmup(*args, grid=(1, ), **consts))


@matrix(CONFIGS)
def test_run(case):
    if not case["launch"]:
        pytest.skip("versions>1 frontend contract only")
    for seed in (7, 19):
        kernel, args, consts = _launch(case, seed)
        cpu_args = tuple(t.detach().cpu() for t in args)
        kernel[(1, )](*args, **consts)
        torch.npu.synchronize()
        actual = args[-1].cpu()
        expected = cpu_args[0].float() @ cpu_args[1].float() if case["space"] == "l0c" else cpu_args[0]
        if case["space"] == "l0c":
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-2)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print("ACCURACY PASS: alloc", case["name"], "seed", seed)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
