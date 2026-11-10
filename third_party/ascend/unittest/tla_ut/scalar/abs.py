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
"""TLA 3-in-1 test: scalar abs writeback IR, compilation, and NPU accuracy."""
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
def abs_integer(Out, x: tl.int64, DTYPE: tl.constexpr):
    out = tla.view(Out, (128, ))
    local = tla.alloc((128, ), tl.int32)
    with tla.scope.vector():
        source = tla.cast(x, DTYPE)
        result = tla.abs(source)
        tl.static_assert(result.dtype == DTYPE)
        low = tla.cast(result, tl.int32)
        high = tla.cast(tla.cast(result, tl.uint64) >> 32, tl.int32)
        with tla.scope.simd():
            tla.simd.vsts(local[0], tla.simd.vdup(low, tl.int32))
            tla.simd.vsts(local[64], tla.simd.vdup(high, tl.int32))
        tla.store_ub_to_gm(local, out)


@tla.jit
def abs_float(Out, x, DTYPE: tl.constexpr, N: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), DTYPE)
    with tla.scope.vector():
        source = tla.cast(x, DTYPE)
        result = tla.abs(source)
        tl.static_assert(result.dtype == DTYPE)
        with tla.scope.simd():
            value = tla.simd.vdup(result, DTYPE)
            if DTYPE == tl.float16:
                tla.simd.vsts(local[0], value, dist="NORM_B16")
            else:
                tla.simd.vsts(local[0], value, dist="NORM_B32")
        tla.store_ub_to_gm(local, out)


CONFIGS = [
    dict(name=str(dtype), dtype=dtype) for dtype in (
        tl.int1,
        tl.int8,
        tl.uint8,
        tl.int16,
        tl.uint16,
        tl.int32,
        tl.uint32,
        tl.int64,
        tl.uint64,
        tl.float16,
        tl.float32,
    )
]


def _kernel(case):
    dtype = case["dtype"]
    if dtype.is_int():
        return abs_integer, {"Out": "*i32", "x": "i64"}, {"DTYPE": dtype}
    output_type = "fp16" if dtype == tl.float16 else "fp32"
    return abs_float, {"Out": "*" + output_type, "x": "fp32"}, {"DTYPE": dtype, "N": 2048 // dtype.primitive_bitwidth}


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    kernel, signature, constants = _kernel(case)
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(kernel, signature, constants, output_dir=dump)
    text = str(lower_offline(module, ctx, output_dir=dump))
    dtype = case["dtype"]
    if dtype.is_floating():
        assert "math.absf" in text
    elif dtype.is_int_signed():
        assert "math.absi" in text
    else:
        assert "math.absi" not in text and "math.absf" not in text
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1, "scalar abs introduced an implicit buffer"
    print("IR PASS: abs", case["name"])


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    dtype = case["dtype"]
    output_dtype = {tl.float16: torch.float16, tl.float32: torch.float32}.get(dtype, torch.int32)
    n = 2048 // dtype.primitive_bitwidth if dtype in (tl.float16, tl.float32) else 128
    return torch.empty(n, dtype=output_dtype, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, constants = _kernel(case)
    x = -7 if case["dtype"].is_int() else -1.5
    check_kernel_binary(kernel.warmup(_new_output(case), x, **constants, grid=(1, )))


def _signed_word(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def _integer_reference(value, dtype):
    bits = dtype.primitive_bitwidth
    value = int(value != 0) if dtype == tl.int1 else value % (1 << bits)
    if dtype.is_int_signed():
        if value >= 1 << (bits - 1):
            value -= 1 << bits
        if value != -(1 << (bits - 1)):
            value = abs(value)
    return value


def _values(dtype):
    rng = random.Random(0)
    if dtype.is_int():
        bits = dtype.primitive_bitwidth
        # Raw signed i64 values also supply unsigned high-bit patterns.
        boundary = 1 << (bits - 1)
        values = [
            0, 1, -1, -boundary, boundary - 1, -boundary + 1, -(1 << 63), (1 << 63) - 1, -(1 << 40) - 1, (1 << 48) + 1
        ]
        return values + [rng.randrange(-(1 << 63), 1 << 63) for _ in range(16)]
    tiny = 2.0**(-24 if dtype == tl.float16 else -149)
    limit = 65504.0 if dtype == tl.float16 else 3.4028234663852886e38
    return [0.0, -0.0, 1.5, -1.5, tiny, -tiny, limit, -limit,
            float("inf"), -float("inf"),
            float("nan"), -float("nan")] + [rng.uniform(-100, 100) for _ in range(16)]


@matrix(CONFIGS)
def test_run(case):
    import torch

    dtype = case["dtype"]
    kernel, _, constants = _kernel(case)
    out = _new_output(case)
    for x in _values(dtype):
        kernel[(1, )](out, x, **constants)
        torch.npu.synchronize()
        actual = out.cpu()
        if dtype.is_int():
            result = _integer_reference(x, dtype)
            words = [_signed_word(result >> shift) for shift in (0, 32)]
            expected = torch.tensor(words, dtype=torch.int32).repeat_interleave(64)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            expected = torch.tensor(x, dtype=out.dtype).abs().expand_as(actual)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            # Value equality alone does not detect a failure to clear -0 or NaN's sign.
            assert not torch.signbit(actual).any()
    print("ABS OK:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    print("ALL PASS: abs")


if __name__ == "__main__":
    main()
