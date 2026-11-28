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
"""TLA scalar max_value: kernel IR, target compilation, and NPU accuracy."""
import sys
from pathlib import Path

# Do not let this category's select.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import triton.language as tl
from triton.language.extra import tla
from utils.tla_test_utils import get_stages, matrix, compile_offline, lower_offline, check_kernel_binary

CONFIGS = [
    dict(name=name, dtype=dtype, value=value) for name, dtype, value in (
        ("bool", tl.int1, 1),
        ("int8", tl.int8, 127),
        ("uint8", tl.uint8, 255),
        ("int16", tl.int16, 32767),
        ("uint16", tl.uint16, 65535),
        ("int32", tl.int32, 2147483647),
        ("uint32", tl.uint32, 4294967295),
        ("int64", tl.int64, 9223372036854775807),
        ("uint64", tl.uint64, 18446744073709551615),
        ("float16", tl.float16, 65504.0),
        ("bfloat16", tl.bfloat16, float.fromhex("0x1.fep127")),
        ("float32", tl.float32, float.fromhex("0x1.fffffep127")),
        ("float64", tl.float64, sys.float_info.max),
    )
]


@tla.jit
def max_value_scalar(Out, DTYPE: tl.constexpr, N: tl.constexpr):
    value = tla.max_value(DTYPE)
    tl.static_assert(value.dtype == DTYPE)
    # Expose both 32-bit words so 64-bit maxima cannot pass after truncation.
    if DTYPE.is_int() or DTYPE == tl.float64:
        out = tla.view(Out, (128, ))
        local = tla.alloc((128, ), tl.int32)
        with tla.scope.vector():
            if DTYPE == tl.float64:
                bits = tla.cast(value, tl.uint64, bitcast=True)
            else:
                bits = tla.cast(value, tl.uint64)
            low = tla.cast(bits, tl.int32)
            high = tla.cast(bits >> 32, tl.int32)
            with tla.scope.simd():
                tla.simd.vsts(local[0], tla.simd.vdup(low, tl.int32))
                tla.simd.vsts(local[64], tla.simd.vdup(high, tl.int32))
            tla.store_ub_to_gm(local, out)
    else:
        out = tla.view(Out, (N, ))
        local = tla.alloc((N, ), DTYPE)
        with tla.scope.vector():
            with tla.scope.simd():
                register = tla.simd.vdup(value, DTYPE)
                if DTYPE == tl.float32:
                    tla.simd.vsts(local[0], register, dist="NORM_B32")
                else:
                    tla.simd.vsts(local[0], register, dist="NORM_B16")
            tla.store_ub_to_gm(local, out)


def constexprs(case):
    dtype = case["dtype"]
    return dict(DTYPE=dtype, N=2048 // dtype.primitive_bitwidth)


def _output_type(case):
    dtype = case["dtype"]
    return tl.int32 if dtype.is_int() or dtype == tl.float64 else dtype


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    output_type = "i32" if _output_type(case) == tl.int32 else _output_type(case).name
    module, ctx = compile_offline(max_value_scalar, {"Out": "*" + output_type}, constexprs(case), output_dir=dump)
    text = str(lower_offline(module, ctx, output_dir=dump))
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1, "max_value introduced an implicit buffer"
    assert "arith.extf" not in text and "arith.truncf" not in text
    assert "arith.uitofp" not in text and "arith.sitofp" not in text
    print("IR PASS: max_value", case["name"])


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    dtype = _output_type(case)
    torch_dtype = {
        tl.int32: torch.int32, tl.float16: torch.float16, tl.bfloat16: torch.bfloat16, tl.float32: torch.float32
    }[dtype]
    n = 128 if dtype == tl.int32 else constexprs(case)["N"]
    return torch.empty(n, dtype=torch_dtype, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    out = _new_output(case)
    compiled = max_value_scalar.warmup(out, **constexprs(case), grid=(1, ))
    check_kernel_binary(compiled)


@matrix(CONFIGS)
def test_run(case):
    import struct
    import torch

    out = _new_output(case)
    if _output_type(case) == tl.int32:
        bits = case["value"]
        if case["dtype"] == tl.float64:
            bits, = struct.unpack("<Q", struct.pack("<d", bits))
        words = [(bits >> shift) & 0xFFFFFFFF for shift in (0, 32)]
        expected = torch.tensor([word if word < 2**31 else word - 2**32 for word in words],
                                dtype=torch.int32).repeat_interleave(64)
    else:
        expected = torch.full(out.shape, case["value"], dtype=out.dtype)
    max_value_scalar[(1, )](out, **constexprs(case))
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    print("MAX_VALUE OK:", case["name"])


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
