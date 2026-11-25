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
"""TLA 3-in-1 test: scalar select IR, compilation, and NPU accuracy."""
import sys
from pathlib import Path

# Do not let this category's select.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import triton.language as tl
from triton.language.extra import tla
from utils.tla_test_utils import (get_stages, matrix, compile_offline, lower_offline, check_kernel_binary)


@tla.jit
def select_scalar(Out, condition, x, y, DTYPE: tl.constexpr):
    out = tla.view(Out, (64, ))
    local = tla.alloc((64, ), DTYPE)
    with tla.scope.vector():
        result = tla.select(condition, x, y)
        with tla.scope.simd():
            value = tla.simd.vdup(result, DTYPE)
            tla.simd.vsts(local[0], value)
        tla.store_ub_to_gm(local, out)


CONFIGS = [
    dict(name=name, left=left, right=right, condition=condition, dtype=dtype)
    for name, left, right, condition, dtype in (
        ("f32", "fp32", "fp32", "i1", tl.float32),
        ("i32", "i32", "i32", "i1", tl.int32),
        ("i32_condition", "fp32", "fp32", "i32", tl.float32),
        ("f32_condition", "fp32", "fp32", "fp32", tl.float32),
        ("f32_i32", "fp32", "i32", "i1", tl.float32),
        ("i32_f32", "i32", "fp32", "i1", tl.float32),
    )
]


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    output = "fp32" if case["dtype"] == tl.float32 else "i32"
    signature = {"Out": "*" + output, "condition": case["condition"], "x": case["left"], "y": case["right"]}
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(select_scalar, signature, {"DTYPE": case["dtype"]}, output_dir=dump)
    text = str(lower_offline(module, ctx, output_dir=dump))
    assert "arith.select" in text
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1
    print("IR PASS: select", case["name"])


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    dtype = torch.float32 if case["dtype"] == tl.float32 else torch.int32
    return torch.empty(64, dtype=dtype, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    out = _new_output(case)
    x = -2.5 if case["left"] == "fp32" else -7
    y = -3.5 if case["right"] == "fp32" else -3
    condition = {"i1": True, "i32": -2, "fp32": -0.5}[case["condition"]]
    compiled = select_scalar.warmup(out, condition, x, y, DTYPE=case["dtype"], grid=(1, ))
    check_kernel_binary(compiled)


@matrix(CONFIGS)
def test_run(case):
    import torch

    out = _new_output(case)
    values = ([-2.5, -0.0, float("inf"), -float("inf"), float("nan")]
              if case["left"] == "fp32" else [-2147483648, -7, 0, 2147483647])
    conditions = {"i1": (False, True), "i32": (0, -2), "fp32": (0.0, -0.5)}[case["condition"]]
    for x in values:
        for condition in conditions:
            y = -3.5 if case["right"] == "fp32" else -3
            select_scalar[(1, )](out, condition, x, y, DTYPE=case["dtype"])
            torch.npu.synchronize()
            reference = torch.tensor(x, dtype=out.dtype)
            expected = reference if condition else torch.tensor(y, dtype=out.dtype)
            actual = out.cpu()
            torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0, equal_nan=True)
            if out.dtype == torch.float32 and expected.item() == 0:
                assert torch.equal(torch.signbit(actual), torch.signbit(expected).expand_as(actual))
    print("SELECT OK:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    print("ALL PASS: select")


if __name__ == "__main__":
    main()
