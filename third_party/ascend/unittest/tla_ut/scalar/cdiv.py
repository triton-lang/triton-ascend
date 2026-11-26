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
"""TLA scalar cdiv: kernel IR, target compilation, and NPU accuracy."""
import random
import sys
from pathlib import Path

# Do not let this category's select.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import triton.language as tl
from triton.language.extra import tla
from utils.tla_test_utils import get_stages, matrix, compile_offline, lower_offline, check_kernel_binary


@tla.jit
def cdiv_scalar(Out, x: tl.int64, div: tl.int64, LEFT: tl.constexpr, RIGHT: tl.constexpr, RESULT: tl.constexpr):
    out = tla.view(Out, (128, ))
    local = tla.alloc((128, ), tl.int32)
    with tla.scope.vector():
        dividend = tla.cast(x, LEFT)
        divisor = tla.cast(div, RIGHT)
        result = tla.cdiv(dividend, divisor)
        tl.static_assert(result.dtype == RESULT)
        low = tla.cast(result, tl.int32)
        high = tla.cast(tla.cast(result, tl.uint64) >> 32, tl.int32)
        with tla.scope.simd():
            tla.simd.vsts(local[0], tla.simd.vdup(low, tl.int32))
            tla.simd.vsts(local[64], tla.simd.vdup(high, tl.int32))
        tla.store_ub_to_gm(local, out)


INTEGER_TYPES = (tl.int8, tl.uint8, tl.int16, tl.uint16, tl.int32, tl.uint32, tl.int64, tl.uint64)
CONFIGS = [dict(name=str(dtype), left=dtype, right=dtype, result=dtype) for dtype in INTEGER_TYPES]
CONFIGS += [
    dict(name=name, left=left, right=right, result=result) for name, left, right, result in (
        ("i8_i16", tl.int8, tl.int16, tl.int16),
        ("u16_u32", tl.uint16, tl.uint32, tl.uint32),
        ("i32_i64", tl.int32, tl.int64, tl.int64),
        ("i32_u32", tl.int32, tl.uint32, tl.uint32),
    )
]


def constexprs(case):
    return dict(LEFT=case["left"], RIGHT=case["right"], RESULT=case["result"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(cdiv_scalar, {"Out": "*i32", "x": "i64", "div": "i64"}, constexprs(case))
    text = str(lower_offline(module, ctx, output_dir=dump))
    division = "arith.divsi" if case["result"].is_int_signed() else "arith.divui"
    assert division in text
    assert "arith.addi" in text
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1, "scalar cdiv introduced an implicit buffer"
    assert "cdiv" in tla.__all__
    print("IR PASS: cdiv", case["name"])


def _new_output():
    import torch
    import torch_npu  # noqa: F401

    return torch.empty(128, dtype=torch.int32, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    compiled = cdiv_scalar.warmup(_new_output(), 1, 1, **constexprs(case), grid=(1, ))
    check_kernel_binary(compiled)


def _values(case):
    limit = case["left"].get_int_max_value()
    divisor_limit = case["right"].get_int_max_value()
    result_limit = case["result"].get_int_max_value()
    pairs = [(0, 1), (1, 1), (1, 2), (7, 3), (16, 4), (17, 4), (limit, 1), (limit - 6, 7)]
    if limit > 2**32:
        pairs += [(2**32 + 5, 3), (2**40 + 7, 1)]
    rng = random.Random(0)
    pairs += [(rng.randint(0,
                           min(limit, result_limit) // 2),
               rng.randint(1, max(1, min(divisor_limit, result_limit // 4, 1024)))) for _ in range(16)]
    return [(x, div)
            for x, div in pairs
            if 0 <= x <= limit and 0 < div <= divisor_limit and x + div - 1 <= result_limit]


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value if value < 1 << (bits - 1) else value - (1 << bits)


@matrix(CONFIGS)
def test_run(case):
    import torch

    out = _new_output()
    for x, div in _values(case):
        cdiv_scalar[(1, )](out, _signed(x, 64), _signed(div, 64), **constexprs(case))
        torch.npu.synchronize()
        # Independent nonnegative ceiling reference, avoiding the addition.
        expected = x // div + int(x % div != 0)
        words = [_signed(expected >> shift, 32) for shift in (0, 32)]
        reference = torch.tensor(words, dtype=torch.int32).repeat_interleave(64)
        torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
    print("CDIV OK:", case["name"])


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
