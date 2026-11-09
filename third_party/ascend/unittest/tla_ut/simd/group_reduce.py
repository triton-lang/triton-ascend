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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Native 32-byte group reductions, input masks and complete output registers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, lower_offline, compile_native_kernel


@tla.jit
def group_kernel(X, M, O, DT: tl.constexpr, W: tl.constexpr, OP: tl.constexpr, MASKED: tl.constexpr,
                 MODE: tl.constexpr):
    N: tl.constexpr = 2048 // W
    MT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    x = tla.alloc_buffer((N, ), DT, space="ub")
    m = tla.alloc_buffer((N, ), MT, space="ub")
    WIDEN: tl.constexpr = OP == "add" and (DT == tl.int16 or DT == tl.uint16)
    OUT_DT: tl.constexpr = (tl.uint32 if DT == tl.uint16 else tl.int32) if WIDEN else DT
    OUT_N: tl.constexpr = 64 if WIDEN else N
    OW: tl.constexpr = 32 if WIDEN else W
    out = tla.alloc_buffer((OUT_N, ), OUT_DT, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        if not MASKED:
            mask = None
        if OP == "add":
            result = tla.simd.vcgadd(a, mask, mode=MODE)
        elif OP == "max":
            result = tla.simd.vcgmax(a, mask, mode=MODE)
        else:
            result = tla.simd.vcgmin(a, mask, mode=MODE)
        tla.simd.vsts(out[0], result, dist="NORM_B16" if OW == 16 else "NORM_B32")
    tla.copy(out, tla.view(O, (OUT_N, )))


TYPES = [("int16", "i16", 16), ("uint16", "u16", 16), ("int32", "i32", 32), ("uint32", "u32", 32),
         ("float16", "fp16", 16), ("float32", "fp32", 32)]
CASES = [(op, typ) for op in ("add", "max", "min") for typ in TYPES]


def output_sig(case):
    op, (name, sig, _) = case
    return ("u32" if name == "uint16" else "i32") if op == "add" and name in ("int16", "uint16") else sig


def constants(case, masked, mode):
    op, (name, _, width) = case
    return dict(DT=getattr(tl, name), W=width, OP=op, MASKED=masked, MODE=mode)


def launch(case, pattern, mode="ZEROING"):
    op, (name, _, width) = case
    n = 2048 // width
    dtype = getattr(torch, name)
    lane = torch.arange(n)
    raw = (lane * 13 + 7) % 31 - (0 if name.startswith("uint") else 15)
    if pattern == "extreme":
        if name.startswith("float"):
            raw = torch.where(lane % 2 == 0, 128, -64)
        else:
            info = torch.iinfo(dtype)
            raw = torch.where(lane % 3 == 0, info.min, info.max)
    if pattern == "special":
        groups = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0, 0.0, 1.0, -1.0, 2.0], dtype=dtype)
        raw = groups.repeat_interleave(n // 8)
    elif pattern == "masked_nan":
        raw = raw.to(torch.float64)
        raw[lane % 2 == 0] = float("nan")
    data = raw.to(dtype)
    mask = {
        "all": lane >= 0, "none": lane < 0, "tail": lane < n - 3, "mixed": (lane * 7 + 3) % 11 < 5, "unmasked": lane
        >= 0, "extreme": lane >= 0, "special": lane >= 0, "masked_nan": lane >= 0
    }[pattern]
    if pattern == "masked_nan":
        mask = lane % 2 != 0
    widen = op == "add" and name in ("int16", "uint16")
    out_dtype = (torch.uint32 if name == "uint16" else torch.int32) if widen else dtype
    out_n = 64 if widen else n
    out = torch.empty(out_n, dtype=out_dtype, device="npu")
    compiled = group_kernel[(1, )](data.npu(), mask.to({8: torch.int8, 16: torch.int16, 32: torch.int32}[width]).npu(),
                                   out, **constants(case, pattern != "unmasked", mode))
    torch.npu.synchronize()
    values = data.to(torch.float64 if name.startswith("float") else torch.int64).reshape(8, -1)
    active = mask.reshape(8, -1)
    if op == "add":
        reduced = torch.where(active, values, 0).sum(1)
    else:
        if name.startswith("float"):
            identity = -float("inf") if op == "max" else float("inf")
        else:
            info = torch.iinfo(dtype)
            identity = info.min if op == "max" else info.max
        values = torch.where(active, values, identity)
        reduced = values.max(1).values if op == "max" else values.min(1).values
    expected = torch.cat((reduced.to(out_dtype).to(torch.float64), torch.zeros(out_n - 8, dtype=torch.float64)))
    return out.cpu().to(torch.float64), expected, compiled


@pytest.mark.parametrize("case", CASES)
def test_ir(case):
    op, (name, sig, w) = case
    module, ctx = compile_offline(group_kernel, dict(X=f"*{sig}", M=f"*i{w}", O=f"*{output_sig(case)}"),
                                  constants(case, True, "ZEROING"))
    text = str(module)
    assert f'ave.hir.group_reduction <{op}>' in text
    assert 'llvm.call_intrinsic' not in text
    assert "ave.hir.vdintlv" not in text
    if op == "add" and name in ("int16", "uint16"):
        assert "-> vector<64xi32>" in text
        assert "vector.bitcast" not in text
    lowered = str(lower_offline(module, ctx))
    assert 'ave.hir.group_reduction' in lowered
    assert 'llvm.call_intrinsic' not in lowered


@pytest.mark.parametrize("case", CASES)
def test_compile(case):
    _, (_, sig, w) = case
    compile_native_kernel(group_kernel, dict(X=f"*{sig}", M=f"*i{w}", O=f"*{output_sig(case)}"),
                          constants(case, True, "ZEROING"))


@pytest.mark.parametrize("case,pattern",
                         [(case, pattern)
                          for case in CASES
                          for pattern in (("unmasked", "all", "none", "tail", "mixed", "extreme") +
                                          (("special", "masked_nan") if case[1][0].startswith("float") else ()))])
@pytest.mark.parametrize("mode", ("ZEROING", "MERGING"))
def test_run(case, pattern, mode):
    actual, expected, _ = launch(case, pattern, mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if pattern == "special" and case[0] != "add":
        assert torch.signbit(actual[3]) == torch.signbit(expected[3])
