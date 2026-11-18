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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Scalar unary math IR, compilation, and NPU accuracy tests."""
import sys
from pathlib import Path

# This directory contains select.py, which would shadow the standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import pytest
import triton.language as tl
from triton.language.extra import tla
from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline, matrix


@tla.jit
def unary_math(out, x, OP: tl.constexpr):
    ub = tla.alloc((64, ), tl.float32)
    with tla.scope.vector():
        y = OP(x)
        tl.static_assert(y.dtype == tl.float32)
        with tla.scope.simd():
            tla.simd.vsts(ub[0], tla.simd.vdup(y, tl.float32), dist="NORM_B32")
        tla.store_ub_to_gm(ub, tla.view(out, (64, )))


_SPECIAL = (float("-inf"), -0.0, 0.0, float("inf"), float("nan"))
_CONFIGS = [
    dict(name="exp", op=tla.exp, ir=("math.exp", ), values=(-100.0, -80.0, -1.5, 1.5, 80.0, 100.0)),
]


@matrix(_CONFIGS)
def test_ir(case):
    module, ctx = compile_offline(unary_math, {"out": "*fp32", "x": "fp32"}, {"OP": case["op"]})
    lowered = str(lower_offline(module, ctx))
    for op in case["ir"]:
        assert op in lowered
    assert lowered.count("memref.alloc()") == 1
    assert "hivm.hir.store" in lowered
    assert "tt." not in lowered and "tensor<" not in lowered


@matrix(_CONFIGS)
def test_unsupported_inputs(case):
    for x in (True, 2, tla.alloc, tl.tensor(None, tl.float16), tl.tensor(None, tl.float64),
              tl.tensor(None, tl.block_type(tl.float32, (64, )))):
        with pytest.raises(TypeError, match=rf"tla\.{case['name']} expects a float32 scalar"):
            case["op"](x, _semantic=object())


def _output():
    import torch
    import torch_npu  # noqa: F401

    return torch.empty(64, dtype=torch.float32, device="npu")


@matrix(_CONFIGS)
def test_compile(case):
    check_kernel_binary(unary_math.warmup(_output(), 0.25, OP=case["op"], grid=(1, )))


@matrix(_CONFIGS)
def test_run(case):
    import torch

    out = _output()
    reference = getattr(torch, case["name"])
    for x in (*_SPECIAL, *case["values"]):
        unary_math[(1, )](out, x, OP=case["op"])
        torch.npu.synchronize()
        expected = reference(torch.tensor(x, dtype=torch.float32)).expand_as(out)
        actual = out.cpu()
        # The target exponential may flush subnormal results to zero.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=torch.finfo(torch.float32).tiny, equal_nan=True)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in _CONFIGS:
        if do_ir:
            test_ir(case)
            test_unsupported_inputs(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
