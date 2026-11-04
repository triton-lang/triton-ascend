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
"""Exact integer scalar constants through arithmetic and broadcast APIs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline


@tla.jit
def exact_scalar(X, O, DT: tl.constexpr, VALUE: tl.constexpr, OP: tl.constexpr):
    x = tla.alloc_buffer((64, ), DT, space="ub")
    out = tla.alloc_buffer((64, ), DT, space="ub")
    tla.copy(tla.view(X, (64, )), x)
    with tla.scope.simd():
        src = tla.simd.vlds(x[0])
        if OP == "add":
            value = tla.simd.vadds(src, VALUE)
        elif OP == "mul":
            value = tla.simd.vmuls(src, VALUE)
        elif OP == "full":
            value = tla.simd.vfull(VALUE, DT, 64)
        else:
            value = tla.simd.vsplat(VALUE, DT)
        tla.simd.vsts(out[0], value, dist="NORM_B64")
    tla.copy(out, tla.view(O, (64, )))


VALUES = [("int64", value) for value in (2**53 - 1, 2**53 + 1, -(2**53 + 1), -(2**63), 2**63 - 1)]
VALUES += [("uint64", value) for value in (2**53 + 1, 2**63 + 1, 2**64 - 1)]


@pytest.mark.parametrize("dtype,value", VALUES)
@pytest.mark.parametrize("op", ("add", "mul", "full", "splat"))
def test_ir(dtype, value, op):
    sig = "u64" if dtype == "uint64" else "i64"
    module, _ = compile_offline(exact_scalar, dict(X=f"*{sig}", O=f"*{sig}"),
                                dict(DT=getattr(tl, dtype), VALUE=value, OP=op))
    signed_bits = value if value < 2**63 else value - 2**64
    assert f"{signed_bits} : i64" in str(module)


@pytest.mark.parametrize("dtype,value", VALUES)
@pytest.mark.parametrize("op", ("add", "mul", "full", "splat"))
def test_run(dtype, value, op):
    torch_dtype = getattr(torch, dtype)
    data = torch.full((64, ), 1 if op == "mul" else 0, dtype=torch_dtype)
    out = torch.empty_like(data, device="npu")
    exact_scalar[(1, )](data.npu(), out, DT=getattr(tl, dtype), VALUE=value, OP=op)
    torch.npu.synchronize()
    # Compare raw bits: no CPU uint64 arithmetic or float reference conversion.
    expected = torch.full((64, ), value, dtype=torch_dtype)
    torch.testing.assert_close(out.cpu().view(torch.int64), expected.view(torch.int64), rtol=0, atol=0)
