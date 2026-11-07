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
"""TLA 3-in-1 test: scalar ^ IR, compilation, and NPU accuracy."""
import sys
from pathlib import Path
import random

# Do not let this category's select.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import triton.language as tl
from triton.language.extra import tla
from utils.tla_test_utils import (get_stages, matrix, compile_offline, lower_offline, check_kernel_binary)


@tla.jit
def xor_scalar(Out, x: tl.int64, y: tl.int64, LEFT: tl.constexpr, RIGHT: tl.constexpr, RESULT: tl.constexpr,
               MODE: tl.constexpr):
    out = tla.view(Out, (128, ))
    local = tla.alloc((128, ), tl.int32)
    with tla.scope.vector():
        a = tla.cast(x, LEFT)
        b = tla.cast(y, RIGHT)
        if MODE == "literal_left":
            result = 85 ^ a
        elif MODE == "literal_right":
            result = a ^ 85
        elif MODE == "augmented":
            result = a
            result ^= b
        else:
            result = a ^ b
        tl.static_assert(result.dtype == RESULT)
        low = tla.cast(result, tl.int32)
        high = tla.cast(tla.cast(result, tl.uint64) >> 32, tl.int32)
        with tla.scope.simd():
            tla.simd.vsts(local[0], tla.simd.vdup(low, tl.int32))
            tla.simd.vsts(local[64], tla.simd.vdup(high, tl.int32))
        tla.store_ub_to_gm(local, out)


INTEGER_TYPES = (tl.int1, tl.int8, tl.uint8, tl.int16, tl.uint16, tl.int32, tl.uint32, tl.int64, tl.uint64)
CONFIGS = [dict(name=str(dtype), left=dtype, right=dtype, result=dtype) for dtype in INTEGER_TYPES]
CONFIGS += [
    dict(name=name, left=left, right=right, result=result) for name, left, right, result in (
        ("i8_u16", tl.int8, tl.uint16, tl.uint16),
        ("u8_i16", tl.uint8, tl.int16, tl.int16),
        ("i32_u32", tl.int32, tl.uint32, tl.uint32),
        ("i64_u32", tl.int64, tl.uint32, tl.int64),
    )
]
CONFIGS += [
    dict(name=mode, left=tl.int32, right=tl.int32, result=tl.int32, mode=mode)
    for mode in ("literal_left", "literal_right", "augmented")
]


def _constants(case):
    return dict(LEFT=case["left"], RIGHT=case["right"], RESULT=case["result"], MODE=case.get("mode", "binary"))


def _new_output(dtype, n):
    import torch
    import torch_npu  # noqa: F401

    torch_dtype = {
        tl.float32: torch.float32, tl.float16: torch.float16, tl.bfloat16: torch.bfloat16, tl.int32: torch.int32
    }[dtype]
    return torch.empty(n, dtype=torch_dtype, device="npu")


def integer_value(value, dtype):
    """Interpret a Python integer in a fixed-width Triton scalar type."""
    if dtype == tl.int1:
        return int(value != 0)
    bits = dtype.primitive_bitwidth
    value = int(value) & ((1 << bits) - 1)
    if dtype.is_int_signed() and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


def integer_words(value):
    """Expose both halves of a scalar so i64/u64 device checks lose no bits."""
    return [integer_value(value >> shift, tl.int32) for shift in (0, 32)]


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(xor_scalar, {"Out": "*i32", "x": "i64", "y": "i64"}, _constants(case))
    text = str(lower_offline(module, ctx, output_dir=dump))
    assert "arith.xori" in text
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1
    print("IR PASS: xor", case["name"])


@matrix(CONFIGS)
def test_compile(case):
    out = _new_output(tl.int32, 128)
    check_kernel_binary(xor_scalar.warmup(out, -7, 85, **_constants(case), grid=(1, )))


@matrix(CONFIGS)
def test_run(case):
    import torch

    out = _new_output(tl.int32, 128)
    pairs = [(0, 0), (1, 1), (-1, 0), (-1, 1), (-(1 << 63), (1 << 63) - 1), (0x5555555555555555, -0x5555555555555556),
             (1 << 40, (1 << 48) + 1)]
    rng = random.Random(0)
    pairs += [(rng.randrange(-(1 << 63), 1 << 63), rng.randrange(-(1 << 63), 1 << 63)) for _ in range(16)]
    for x, y in pairs:
        xor_scalar[(1, )](out, x, y, **_constants(case))
        torch.npu.synchronize()
        a = integer_value(x, case["left"])
        b = 85 if case.get("mode", "binary").startswith("literal") else integer_value(y, case["right"])
        result = integer_value(a ^ b, case["result"])
        expected = torch.tensor(integer_words(result), dtype=torch.int32).repeat_interleave(64)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    print("XOR OK:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    print("ALL PASS: xor")


if __name__ == "__main__":
    main()
