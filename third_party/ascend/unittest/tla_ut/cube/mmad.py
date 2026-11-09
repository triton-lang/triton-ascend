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
"""MMAD transpose and leading-submatrix coverage: IR, binary, and accuracy.

Non-square inputs distinguish NN/NT/TN/TT. Truncated inputs retain nonzero
values outside real extents, so silently computing a full tile is observable.
References use CPU FP32 arithmetic on exactly the quantized FP16/BF16 inputs.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, check_kernel_binary, get_stages, matrix

# Mathematical dimensions before optional real extents; physical input shapes
# are reversed independently when the corresponding transpose is requested.
SHAPE = dict(M=48, K=64, N=80)
CONFIGS = [
    dict(name=f"{dtype}_{'t' if ta else 'n'}{'t' if tb else 'n'}", dtype=dtype, ta=ta, tb=tb, real=(None, None, None),
         accumulate=False) for dtype in ("float16", "bfloat16") for ta in (False, True) for tb in (False, True)
] + [
    dict(name="real_m", dtype="bfloat16", ta=False, tb=True, real=(32, None, None), accumulate=False),
    dict(name="real_k", dtype="bfloat16", ta=True, tb=False, real=(None, 32, None), accumulate=False),
    dict(name="real_n", dtype="bfloat16", ta=False, tb=True, real=(None, None, 48), accumulate=False),
    dict(name="real_mkn_tn", dtype="bfloat16", ta=True, tb=False, real=(32, 32, 48), accumulate=False),
    dict(name="real_mkn_tt_accumulate", dtype="float16", ta=True, tb=True, real=(32, 32, 48), accumulate=True),
    dict(name="default_accumulate", dtype="bfloat16", ta=False, tb=False, real=(None, None, None), accumulate=True),
]


def constants(case):
    rm, rk, rn = case["real"]
    return dict(SHAPE, TA=case["ta"], TB=case["tb"], RM=rm, RK=rk, RN=rn, ACCUMULATE=case["accumulate"])


def actual_shape(case):
    return tuple(extent if extent is not None else SHAPE[axis] for axis, extent in zip(("M", "K", "N"), case["real"]))


def signature(case):
    element = "fp16" if case["dtype"] == "float16" else "bf16"
    return dict(A="*" + element, B="*" + element, C="*fp32")


@tla.jit
def gemm_extents(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, TA: tl.constexpr, TB: tl.constexpr,
                 RM: tl.constexpr, RK: tl.constexpr, RN: tl.constexpr, ACCUMULATE: tl.constexpr):
    A_ROWS: tl.constexpr = K if TA else M
    A_COLS: tl.constexpr = M if TA else K
    B_ROWS: tl.constexpr = N if TB else K
    B_COLS: tl.constexpr = K if TB else N
    OUT_M: tl.constexpr = M if RM is None else RM
    OUT_N: tl.constexpr = N if RN is None else RN
    with tla.scope.cube():
        a = tla.alloc((A_COLS // 16, A_ROWS // 16, 16, 16), A.dtype.element_ty, space="l1")
        b = tla.alloc((B_COLS // 16, B_ROWS // 16, 16, 16), B.dtype.element_ty, space="l1")
        acc = tla.alloc((OUT_N // 16, OUT_M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (A_ROWS, A_COLS)), a)
        tla.load_gm_to_l1(tla.view(B, (B_ROWS, B_COLS)), b)
        tla.cube.mmad(a, b, acc, transpose_a=TA, transpose_b=TB, clear_accum=True, real_m=RM, real_k=RK, real_n=RN)
        if ACCUMULATE:
            tla.cube.mmad(a, b, acc, transpose_a=TA, transpose_b=TB, clear_accum=False, real_m=RM, real_k=RK, real_n=RN)
        tla.store_l0c_to_gm(acc, tla.view(C, (OUT_M, OUT_N)),
                            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(gemm_extents, signature(case), constants(case), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    ops = re.findall(r"hivm\.hir\.mmadL1[^\n]*", text)
    assert len(ops) == (2 if case["accumulate"] else 1)
    values = {name: int(value) for name, value in re.findall(r"(%[\w]+) = arith\.constant (\d+) : index", text)}
    for op in ops:
        assert ("a_transpose" in op) == case["ta"]
        assert ("b_transpose" in op) == case["tb"]
        operands = re.search(r"ins\((.*?) :", op).group(1).split(", ")
        assert tuple(values[value] for value in operands[-3:]) == actual_shape(case)
    assert text.count("hivm.hir.nd2nz ") == 2
    assert text.count("hivm.hir.fixpipe ") == 1
    assert text.count("memref.alloc()") == 3
    assert "memref.transpose" not in text and "hivm.hir.vtranspose" not in text
    assert "enable_HF32" not in text
    assert 'mix_mode = "mix"' in text
    assert "!tt.ptr" not in text and "tensor<" not in text
    print("IR PASS: MMAD", case["name"])


def _launch(case, seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    m, k, n = (SHAPE[x] for x in ("M", "K", "N"))
    a = torch.randn((k, m) if case["ta"] else (m, k), generator=generator).to(dtype)
    b = torch.randn((n, k) if case["tb"] else (k, n), generator=generator).to(dtype)
    rm, rk, rn = actual_shape(case)
    x, y = a.npu(), b.npu()
    out = torch.full((rm, rn), float("nan"), dtype=torch.float32, device="npu")
    if compile_only:
        kernel = gemm_extents.warmup(x, y, out, grid=(1, ), **constants(case))
        return kernel, None, None
    kernel = gemm_extents[(1, )](x, y, out, **constants(case))
    torch.npu.synchronize()
    af = a.float().T if case["ta"] else a.float()
    bf = b.float().T if case["tb"] else b.float()
    expected = af[:rm, :rk] @ bf[:rk, :rn]
    if case["accumulate"]:
        expected = expected + expected
    return kernel, out.cpu(), expected


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _ = _launch(case, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    for seed in (7, 19):
        _, actual, expected = _launch(case, seed)
        assert torch.isfinite(actual).all(), "MMAD left unwritten or non-finite output"
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        print("ACCURACY PASS: MMAD", case["name"], "seed", seed, "max_abs_error",
              (actual - expected).abs().max().item())


@tla.jit
def split_qk(Q, K, QR, KR, OUT, M: tl.constexpr):
    with tla.scope.cube():
        q = tla.alloc((32, M // 16, 16, 16), tl.bfloat16, space="l1")
        k = tla.alloc((32, 8, 16, 16), tl.bfloat16, space="l1")
        qr = tla.alloc((4, M // 16, 16, 16), tl.bfloat16, space="l1")
        kr = tla.alloc((4, 8, 16, 16), tl.bfloat16, space="l1")
        acc = tla.alloc((8, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(Q, (M, 512)), q)
        tla.load_gm_to_l1(tla.view(K, (128, 512)), k)
        tla.load_gm_to_l1(tla.view(QR, (M, 64)), qr)
        tla.load_gm_to_l1(tla.view(KR, (128, 64)), kr)
        tla.cube.mmad(q, k, acc, transpose_b=True, clear_accum=True)
        tla.cube.mmad(qr, kr, acc, transpose_b=True, clear_accum=False)
        tla.store_l0c_to_gm(acc, tla.view(OUT, (M, 128)), tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


@pytest.mark.parametrize("m", (16, 32, 64))
class TestTransposeAccumulate:
    """Different K=512/64 products share L0C; H128 also uses M=64 tiles."""

    def test_ir(self, m, output_dir=None):
        module, ctx = compile_offline(split_qk, dict(Q="*bf16", K="*bf16", QR="*bf16", KR="*bf16", OUT="*fp32"),
                                      dict(M=m), output_dir=output_dir)
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        ops = re.findall(r"hivm\.hir\.mmadL1[^\n]*", text)
        assert len(ops) == 2
        values = {name: int(value) for name, value in re.findall(r"(%[\w]+) = arith\.constant (\d+) : index", text)}
        flags = dict(re.findall(r"(%[\w]+) = arith\.constant (true|false)", text))
        outputs = []
        for op, reduction, clear in zip(ops, (512, 64), ("true", "false")):
            assert "b_transpose" in op and "a_transpose" not in op
            operands = re.search(r"ins\((.*?) :", op).group(1).split(", ")
            assert flags[operands[2]] == clear
            outputs.append(re.search(r"outs\((%[\w]+) :", op).group(1))
            assert tuple(values[value] for value in operands[-3:]) == (m, reduction, 128)
        assert outputs[0] == outputs[1]
        assert text.count("hivm.hir.nd2nz ") == 4
        assert text.count("hivm.hir.fixpipe ") == 1
        assert text.count("memref.alloc()") == 5
        assert "memref.transpose" not in text and "hivm.hir.vtranspose" not in text
        assert "!tt.ptr" not in text and "tensor<" not in text

    @staticmethod
    def _launch(m, seed=7, compile_only=False):
        generator = torch.Generator().manual_seed(seed)
        inputs = [
            torch.randn(shape, generator=generator).to(torch.bfloat16)
            for shape in ((m, 512), (128, 512), (m, 64), (128, 64))
        ]
        device_inputs = [x.npu() for x in inputs]
        out = torch.full((m, 128), float("nan"), dtype=torch.float32, device="npu")
        if compile_only:
            return split_qk.warmup(*device_inputs, out, M=m, grid=(1, )), None, None
        kernel = split_qk[(1, )](*device_inputs, out, M=m)
        torch.npu.synchronize()
        q, k, qr, kr = [x.float() for x in inputs]
        return kernel, out.cpu(), q @ k.T + qr @ kr.T

    def test_compile(self, m):
        kernel, _, _ = self._launch(m, compile_only=True)
        check_kernel_binary(kernel)

    def test_run(self, m):
        for seed in (7, 19):
            _, actual, expected = self._launch(m, seed)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)

    split = TestTransposeAccumulate()
    for m in (16, 32, 64):
        if do_ir:
            split.test_ir(m)
        if do_compile:
            split.test_compile(m)
        if do_run:
            split.test_run(m)


if __name__ == "__main__":
    main()
