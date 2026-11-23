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
"""TLA 3-in-1 test: scalar maximum IR, compilation, and NPU accuracy."""
import random
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
def maximum_scalar(Out, x, y, INPUT_DTYPE: tl.constexpr, RIGHT_DTYPE: tl.constexpr, DTYPE: tl.constexpr,
                   N: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), DTYPE)
    with tla.scope.vector():
        a = tla.cast(x, INPUT_DTYPE)
        b = tla.cast(y, RIGHT_DTYPE)
        result = tla.maximum(a, b)
        # Store unsigned results through their signed bit representation.
        stored = tla.cast(result, DTYPE)
        with tla.scope.simd():
            value = tla.simd.vdup(stored, DTYPE)
            tla.simd.vsts(local[0], value, dist="NORM_B32")
        tla.store_ub_to_gm(local, out)


CONFIGS = [
    dict(name=name, left=left, right=right, dtype=dtype) for name, left, right, dtype in (
        ("f32", tl.float32, tl.float32, tl.float32),
        ("i32", tl.int32, tl.int32, tl.int32),
        ("u32", tl.uint32, tl.uint32, tl.int32),
    )
]


def constexprs(case):
    return {
        "INPUT_DTYPE": case["left"], "RIGHT_DTYPE": case["right"], "DTYPE": case["dtype"], "N":
        2048 // case["dtype"].primitive_bitwidth
    }


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    types = {tl.float32: "fp32", tl.int32: "i32"}
    signature = {
        "Out": "*" + types[case["dtype"]], "x": "i32" if case["left"].is_int() else "fp32", "y":
        "i32" if case["right"].is_int() else "fp32"
    }
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(maximum_scalar, signature, constexprs(case), output_dir=dump)
    text = str(lower_offline(module, ctx, output_dir=dump))
    suffix = {"i32": "si", "u32": "ui"}.get(case["name"], "numf")
    assert "arith.max" + suffix in text
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1, "scalar math introduced an implicit buffer"
    print("IR PASS: maximum", case["name"])


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    dtype = {tl.float32: torch.float32, tl.int32: torch.int32}[case["dtype"]]
    return torch.empty(constexprs(case)["N"], dtype=dtype, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    tag = case["name"]
    out = _new_output(case)
    x = -2 if tag in ("i32", "u32") else -1.5
    integer_y = tag in ("i32", "u32")
    compiled = maximum_scalar.warmup(out, x, 3 if integer_y else 2.5, **constexprs(case), grid=(1, ))
    check_kernel_binary(compiled)


def _values(tag):
    rng = random.Random(0)
    if tag in ("i32", "u32"):
        return [(-2147483648, 2147483647), (-1, 1), (1, -1), (0, -1), (7, 7)]
    pairs = [
        (-2.5, 1.5),
        (1.5, -2.5),
        (1.0, 1.0),
        (-0.0, 0.0),
        (0.0, -0.0),
        (float("inf"), -float("inf")),
        # TODO: Restore single-NaN cases after CCE fixes scalar maxnum/minnum lowering.
        # TODO: Restore after Ascend tl.maximum/minimum double-NaN handling is fixed.
        # (float("nan"), float("nan")),
    ]
    return pairs + [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(16)]


@matrix(CONFIGS)
def test_run(case):
    import torch

    tag = case["name"]
    out = _new_output(case)
    for values in _values(tag):
        maximum_scalar[(1, )](out, *values, **constexprs(case))
        torch.npu.synchronize()
        if tag in ("i32", "u32"):
            a = values[0] & 0xFFFFFFFF if tag == "u32" else values[0]
            b = values[1] & 0xFFFFFFFF if tag == "u32" else values[1]
            result = max(a, b)
            result = (result + 2**31) % 2**32 - 2**31
            expected = torch.tensor(result, dtype=out.dtype)
        else:
            a = torch.tensor(values[0], dtype=out.dtype)
            b = torch.tensor(values[1], dtype=out.dtype)
            expected = torch.fmax(a, b).to(out.dtype)
        actual = out.cpu()
        torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0, equal_nan=True)
    print("MAXIMUM OK:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    print("ALL PASS: maximum")


if __name__ == "__main__":
    main()
