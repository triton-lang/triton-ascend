# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""JIT diagnostics and numerical coverage for type-dependent vcvt rounding."""
import os
import sys

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from triton.compiler.errors import CompilationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tla_test_utils import compile_offline, get_stages


@tla.jit
def convert_int(X, Y, N: tl.constexpr, SRC: tl.constexpr, DST: tl.constexpr, RND: tl.constexpr, SAT: tl.constexpr):
    x = tla.alloc((N, ), SRC, space="ub")
    y = tla.alloc((N, ), DST, space="ub")
    with tla.scope.vector():
        tla.load_gm_to_ub(tla.view(X, (N, )), x)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        with tla.scope.simd():
            v = tla.simd.vcvt(tla.simd.vlds(x[0]), DST, round=RND, sat=SAT)
            if DST == tl.int32:
                tla.simd.vsts(y[0], v, dist="NORM_B32")
            else:
                tla.simd.vsts(y[0], v, dist="NORM_B16")
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.store_ub_to_gm(y, tla.view(Y, (N, )))
        tla.sync.barrier(tla.sync.PIPE.ALL)


@tla.jit
def narrow_odd(X, Y):
    x = tla.alloc((64, ), tl.float32, space="ub")
    y = tla.alloc((128, ), tl.float16, space="ub")
    with tla.scope.vector():
        tla.load_gm_to_ub(tla.view(X, (64, )), x)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        with tla.scope.simd():
            v = tla.simd.vcvt(tla.simd.vlds(x[0]), tl.float16, round=tla.simd.ROUND.O)
            tla.simd.vsts(y[0], v, dist="NORM_B16")
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.store_ub_to_gm(y, tla.view(Y, (128, )))
        tla.sync.barrier(tla.sync.PIPE.ALL)


@pytest.mark.parametrize("bits", [16, 32])
def test_ir_rejects_integer_round_odd(bits):
    constants = dict(N=2048 // bits, SRC=getattr(tl, f"float{bits}"), DST=getattr(tl, f"int{bits}"),
                     RND=tla.simd.ROUND.O, SAT=True)
    signature = {"X": f"*fp{bits}", "Y": f"*i{bits}"}
    diagnostic = rf"tla.simd.vcvt f{bits} -> i{bits} does not support round=ROUND\.O; supported round:"
    with pytest.raises(CompilationError, match=diagnostic):
        compile_offline(convert_int, signature, constants)


def test_ir_keeps_floating_round_odd():
    module, _ = compile_offline(narrow_odd, {"X": "*fp32", "Y": "*fp16"}, {})
    text = str(module)
    assert "ave.hir.vtruncf" in text and "odd" in text


@pytest.mark.parametrize("bits", [16, 32])
@pytest.mark.parametrize("sat", [False, True])
@pytest.mark.parametrize("rounding,expected", [
    (tla.simd.ROUND.R, [0, 2, 2, 0, -2, -2, 1, -1]),
    (tla.simd.ROUND.A, [1, 2, 3, -1, -2, -3, 1, -1]),
    (tla.simd.ROUND.F, [0, 1, 2, -1, -2, -3, 1, -2]),
    (tla.simd.ROUND.C, [1, 2, 3, 0, -1, -2, 2, -1]),
    (tla.simd.ROUND.Z, [0, 1, 2, 0, -1, -2, 1, -1]),
])
def test_run(bits, sat, rounding, expected):
    n = 2048 // bits
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1.25, -1.25] * (n // 8), dtype=getattr(torch, f"float{bits}"),
                     device="npu")
    y = torch.empty(n, dtype=getattr(torch, f"int{bits}"), device="npu")
    convert_int[(1, )](x, y, n, getattr(tl, f"float{bits}"), getattr(tl, f"int{bits}"), rounding, sat)
    torch.npu.synchronize()
    assert torch.equal(y.cpu(), torch.tensor(expected * (n // 8), dtype=y.dtype))


@pytest.mark.skipif(not get_stages()[2], reason="NPU run stage disabled")
def test_run_floating_round_odd():
    # Von Neumann rounding retains exact values; otherwise it truncates and
    # sets the retained LSB. These values distinguish it from nearest-ties-odd.
    x = torch.tensor([1.0, 1 + 2**-12, 1 + 3 * 2**-12, 1 + 2**-10, 1 + 9 * 2**-12, -(1 + 2**-12), -1.0, 0.0] * 8,
                     dtype=torch.float32, device="npu")
    y = torch.empty(128, dtype=torch.float16, device="npu")
    narrow_odd[(1, )](x, y)
    torch.npu.synchronize()
    expected = torch.tensor([1.0, 1 + 2**-10, 1 + 2**-10, 1 + 2**-10, 1 + 3 * 2**-10, -(1 + 2**-10), -1.0, 0.0] * 8,
                            dtype=torch.float16)
    # PART.EVEN places the converted f32 lane into every even f16 slot.
    assert torch.equal(y.cpu()[::2], expected)
