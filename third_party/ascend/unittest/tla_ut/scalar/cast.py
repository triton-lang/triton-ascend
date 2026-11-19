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
"""TLA 3-in-1 test: scalar tla.cast IR, compilation, and NPU accuracy."""
import sys
from pathlib import Path
import random
import struct
import math

# Do not let this category's select.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import triton as tr
import triton.language as tl
from triton.language.extra import tla
from utils.tla_test_utils import (get_stages, matrix, compile_offline, lower_offline, check_kernel_binary)


def _integer_cast_kernel(src):

    @tla.jit
    def cast_to_integer(Out, value: src, DST: tl.constexpr, N: tl.constexpr):
        out = tla.view(Out, (N, ))
        local = tla.alloc((N, ), tl.int32)
        with tla.scope.vector():
            if isinstance(value, tl.tensor):
                tl.static_assert(value.dtype == src)
            result = tla.cast(value, DST)
            tl.static_assert(result.dtype == DST)
            # Expose both words so 64-bit integer results are checked in full.
            low = tla.cast(result, tl.int32)
            high = tla.cast(tla.cast(result, tl.uint64) >> 32, tl.int32)
            with tla.scope.simd():
                tla.simd.vsts(local[0], tla.simd.vdup(low, tl.int32))
                tla.simd.vsts(local[64], tla.simd.vdup(high, tl.int32))
            tla.store_ub_to_gm(local, out)

    return cast_to_integer


def _float_cast_kernel(src):

    @tla.jit
    def cast_to_float(Out, value: src, DST: tl.constexpr, N: tl.constexpr):
        out = tla.view(Out, (N, ))
        local = tla.alloc((N, ), DST)
        with tla.scope.vector():
            if isinstance(value, tl.tensor):
                tl.static_assert(value.dtype == src)
            result = tla.cast(value, DST)
            tl.static_assert(result.dtype == DST)
            with tla.scope.simd():
                register = tla.simd.vdup(result, DST)
                if DST == tl.float16:
                    tla.simd.vsts(local[0], register, dist="NORM_B16")
                else:
                    tla.simd.vsts(local[0], register, dist="NORM_B32")
            tla.store_ub_to_gm(local, out)

    return cast_to_float


@tla.jit
def cast_bool_to_integer(Out, value: tl.int1, DST: tl.constexpr, N: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), tl.int32)
    with tla.scope.vector():
        result = tla.cast(value, DST)
        tl.static_assert(result.dtype == tl.int32)
        with tla.scope.simd():
            tla.simd.vsts(local[0], tla.simd.vdup(result, tl.int32))
            tla.simd.vsts(local[64], tla.simd.vdup(0, tl.int32))
        tla.store_ub_to_gm(local, out)


@tla.jit
def bitcast_to_integer(Out, X, DST: tl.constexpr, N: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), tl.int32)
    with tla.scope.vector():
        source = tl.load(X)
        result = tla.cast(source, DST, bitcast=True)
        tl.static_assert(result.dtype == DST)
        low = tla.cast(result, tl.int32)
        high = tla.cast(tla.cast(result, tl.uint64) >> 32, tl.int32)
        with tla.scope.simd():
            tla.simd.vsts(local[0], tla.simd.vdup(low, tl.int32))
            tla.simd.vsts(local[64], tla.simd.vdup(high, tl.int32))
        tla.store_ub_to_gm(local, out)


@tla.jit
def bitcast_to_float(Out, X, DST: tl.constexpr, N: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), DST)
    with tla.scope.vector():
        source = tl.load(X)
        result = tla.cast(source, DST, bitcast=True)
        tl.static_assert(result.dtype == DST)
        with tla.scope.simd():
            value = tla.simd.vdup(result, DST)
            tla.simd.vsts(local[0], value, dist="NORM_B32")
        tla.store_ub_to_gm(local, out)


CONFIGS = [
    dict(name=str(src) + "_to_" + str(dst), src=src, dst=dst) for src, dst in (
        (tl.int8, tl.int32),
        (tl.uint8, tl.int32),
        (tl.int16, tl.int32),
        (tl.uint16, tl.int32),
        (tl.int32, tl.int8),
        (tl.int32, tl.uint8),
        (tl.int32, tl.int16),
        (tl.int32, tl.uint16),
        (tl.int32, tl.int64),
        (tl.uint32, tl.uint64),
        (tl.int64, tl.int32),
        (tl.uint64, tl.int32),
        (tl.int32, tl.float32),
        (tl.uint32, tl.float32),
        (tl.float32, tl.int32),
        (tl.float32, tl.uint32),
        (tl.int32, tl.int1),
        (tl.float32, tl.int1),
        (tl.int1, tl.int32),
        (tl.float32, tl.float16),
        (tl.float16, tl.float32),
    )
]
CONFIGS += [
    dict(name=str(src) + "_bitcast_" + str(dst), src=src, dst=dst, bitcast=True) for src, dst in (
        (tl.float32, tl.int32),
        (tl.int32, tl.float32),
        (tl.float16, tl.uint16),
    )
]

# Bind each source annotation once for offline compilation and runtime launches.
for case in CONFIGS:
    if case.get("bitcast"):
        case["kernel"] = bitcast_to_integer if case["dst"].is_int() else bitcast_to_float
    elif case["src"] == tl.int1:
        case["kernel"] = cast_bool_to_integer
    else:
        case["kernel"] = _integer_cast_kernel(case["src"]) if case["dst"].is_int() else _float_cast_kernel(case["src"])


def _constants(case):
    dst = case["dst"]
    return dict(DST=dst, N=128 if dst.is_int() else 2048 // dst.primitive_bitwidth)


def _signature(case):
    output = tl.int32 if case["dst"].is_int() else case["dst"]
    signature = {"Out": tl.pointer_type(output)}
    if case.get("bitcast"):
        signature["X"] = tl.pointer_type(case["src"])
    else:
        signature["value"] = case["src"]
    return signature


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    dtype = tl.int32 if case["dst"].is_int() else case["dst"]
    torch_dtype = {tl.float32: torch.float32, tl.float16: torch.float16, tl.int32: torch.int32}[dtype]
    return torch.empty(_constants(case)["N"], dtype=torch_dtype, device="npu")


def _kernel_input(value, case):
    src = case["src"]
    if not case.get("bitcast"):
        if src == tl.int1:
            return bool(value)
        return integer_value(value, src) if src.is_int() else float_value(value, src)

    import torch
    import torch_npu  # noqa: F401

    storage = {16: tl.int16, 32: tl.int32}[src.primitive_bitwidth]
    torch_dtype = {tl.int16: torch.int16, tl.int32: torch.int32}[storage]
    bits = torch.tensor([integer_value(value, storage)], dtype=torch_dtype, device="npu")
    # Preserve exact source bits, including NaN payloads.
    return tr.reinterpret(bits, src)


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


def float_value(value, dtype):
    """CPU reference for the supported scalar floating-point formats."""
    try:
        return struct.unpack("e" if dtype == tl.float16 else "f", struct.pack("e" if dtype == tl.float16 else "f",
                                                                              value))[0]
    except OverflowError:
        return math.copysign(float("inf"), value)


def _values(case):
    src, dst = case["src"], case["dst"]
    if case.get("bitcast"):
        # Exact bits for signed zeros, a subnormal, +/-1, infinities and one NaN payload.
        if src.primitive_bitwidth == 16:
            return [0x0000, 0x8000, 0x0001, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e01]
        return [0x00000000, 0x80000000, 0x00000001, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00001]
    rng = random.Random(0)
    if src == tl.int1:
        return [False, True]
    if src.is_int():
        bits = src.primitive_bitwidth
        limit = 1 << (bits - 1)
        values = [0, 1, -1, -limit, limit - 1, 257, 16842753, 2143289343, -16842753, -2143289343]
        values += [rng.randrange(-limit, limit) for _ in range(16)]
        return values
    values = [-3.75, -1.5, -0.0, 0.0, 1.5, 3.75] + [rng.uniform(-20, 20) for _ in range(16)]
    if dst.is_int() and dst != tl.int1 and not case.get("bitcast"):
        if not dst.is_int_signed():
            values = [abs(x) for x in values]
        return values
    values += [1.00390625, 1.01171875, 65504.0, 1e-8]
    if dst != tl.int1:
        values += [float("inf"), -float("inf"), float("nan")]
    return values


def _reference(value, case):
    src, dst = case["src"], case["dst"]
    if case.get("bitcast"):
        # Floating outputs are inspected through their integer storage view.
        return integer_value(value, dst if dst.is_int() else src)
    value = integer_value(value, src) if src.is_int() else float_value(value, src)
    if dst.is_int():
        return integer_value(value != 0 if dst == tl.int1 else int(value), dst)
    return float_value(value, dst)


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    kernel = case["kernel"]
    if not case.get("bitcast"):
        value_param = next(param for param in kernel.params if param.name == "value")
        assert tl.str_to_ty(value_param.annotation_type, None) == case["src"]
    module, ctx = compile_offline(kernel, _signature(case), _constants(case))
    text = str(lower_offline(module, ctx, output_dir=dump))
    if case.get("bitcast"):
        assert "arith.bitcast" in text
        assert "arith.truncf" not in text and "arith.extf" not in text
    assert text.count("memref.load ") == (1 if case.get("bitcast") else 0)
    assert "hivm.hir.store" in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1
    print("IR PASS: cast", case["name"])


@matrix(CONFIGS)
def test_compile(case):
    constants = _constants(case)
    out = _new_output(case)
    source = _kernel_input(_values(case)[0], case)
    check_kernel_binary(case["kernel"].warmup(out, source, **constants, grid=(1, )))


@matrix(CONFIGS)
def test_run(case):
    import torch

    constants = _constants(case)
    out = _new_output(case)
    for value in _values(case):
        source = _kernel_input(value, case)
        case["kernel"][(1, )](out, source, **constants)
        torch.npu.synchronize()
        result = _reference(value, case)
        actual = out.cpu()
        if case.get("bitcast") and not case["dst"].is_int():
            actual = actual.view(torch.int32)
            expected = torch.full_like(actual, result)
        elif case["dst"].is_int():
            expected = torch.tensor(integer_words(result), dtype=torch.int32).repeat_interleave(64)
        else:
            expected = torch.full_like(actual, result)
        # Numerical NaNs compare by category; all bitcasts compare exact bits.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        if not case.get("bitcast") and not case["dst"].is_int() and result == 0:
            assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    print("CAST OK:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    print("ALL PASS: cast")


if __name__ == "__main__":
    main()
