# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Scalar right shifts: signedness, masks, widths and dynamic shift amounts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, check_kernel_binary


@tla.jit
def shift_scalar(X, M, O, SHIFT, DT: tl.constexpr, W: tl.constexpr, DYNAMIC: tl.constexpr, MODE: tl.constexpr,
                 MASKED: tl.constexpr):
    N: tl.constexpr = 2048 // W
    x = tla.alloc_buffer((N, ), DT, space="ub")
    masks = tla.alloc_buffer((N, ), DT, space="ub")
    out = tla.alloc_buffer((N, ), DT, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(M, (N, )), masks)
    with tla.scope.simd():
        src = tla.simd.vlds(x[0])
        mask = tla.simd.vcmps(tla.simd.vlds(masks[0]), 0, op=tla.simd.CMP.NE)
        if not MASKED:
            mask = None
        if DYNAMIC:
            result = tla.simd.vshrs(src, SHIFT, mask, mode=MODE)
        else:
            result = tla.simd.vshrs(src, 3, mask, mode=MODE)
        tla.simd.vsts(out[0], result, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
    tla.copy(out, tla.view(O, (N, )))


TYPES = [("int8", "i8", 8), ("uint8", "u8", 8), ("int16", "i16", 16), ("uint16", "u16", 16), ("int32", "i32", 32),
         ("uint32", "u32", 32)]


@pytest.mark.parametrize("case", TYPES)
@pytest.mark.parametrize("dynamic", (False, True))
def test_ir(case, dynamic):
    name, sig, width = case
    module, _ = compile_offline(
        shift_scalar, dict(X=f"*{sig}", M=f"*{sig}", O=f"*{sig}", SHIFT="i32"),
        dict(DT=getattr(tl, name), W=width, DYNAMIC=dynamic, MODE=tla.simd.MODE.ZEROING, MASKED=True))
    text = str(module)
    assert "ave.hir.vshrs " in text
    assert "llvm." not in text


@pytest.mark.parametrize("case", TYPES)
@pytest.mark.parametrize("mode", (tla.simd.MODE.ZEROING, tla.simd.MODE.MERGING))
@pytest.mark.parametrize("pattern", ("default", "none", "tail", "mixed"))
@pytest.mark.parametrize("shift", (None, 0, 3, 7))
def test_run(case, mode, pattern, shift):
    name, _, width = case
    n = 2048 // width
    dtype = getattr(torch, name)
    raw = (torch.arange(n, dtype=torch.int64) * 97 + 13) % (1 << width)
    raw[:8] = torch.tensor([0, 1, 7, 8, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 2, (1 << width) - 1])
    if name.startswith("int"):
        raw = torch.where(raw >= (1 << (width - 1)), raw - (1 << width), raw)
    data = raw.to(dtype)
    lane = torch.arange(n)
    masks = {"default": lane >= 0, "none": lane < 0, "tail": lane < n - 7, "mixed": lane % 3 == 1}
    mask = masks[pattern]
    out = torch.empty((n, ), dtype=dtype, device="npu")
    amount = 3 if shift is None else shift
    kernel = shift_scalar[(1, )](data.npu(), mask.to(dtype).npu(), out, amount, DT=getattr(tl, name), W=width,
                                 DYNAMIC=shift is not None, MODE=mode, MASKED=pattern != "default")
    torch.npu.synchronize()
    check_kernel_binary(kernel)
    expected = torch.where(mask, raw >> amount, 0)
    torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)
