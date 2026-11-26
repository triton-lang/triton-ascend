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
"""TLA scalar infinity: kernel IR, target compilation, and exact NPU writeback."""
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
def infinity_scalar(Out, DTYPE: tl.constexpr, N: tl.constexpr, IN_VF: tl.constexpr, NEGATIVE: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), DTYPE)
    with tla.scope.vector():
        if not IN_VF:
            value = tla.infinity(DTYPE)
            if NEGATIVE:
                value = -value
            tl.static_assert(value.dtype == DTYPE)
            tl.static_assert(len(value.shape) == 0)
        with tla.scope.simd():
            if IN_VF:
                value = tla.infinity(DTYPE)
                if NEGATIVE:
                    value = -value
                tl.static_assert(value.dtype == DTYPE)
                tl.static_assert(len(value.shape) == 0)
            reg = tla.simd.vdup(value, DTYPE)
            if DTYPE == tl.float32:
                tla.simd.vsts(local[0], reg, dist="NORM_B32")
            else:
                tla.simd.vsts(local[0], reg, dist="NORM_B16")
        tla.store_ub_to_gm(local, out)


CONFIGS = [
    dict(name=f"{tag}_{'negative' if negative else 'positive'}_{'inside' if in_vf else 'outside'}", tag=tag,
         dtype=dtype, negative=negative, in_vf=in_vf, bits=bits,
         expected_bits=negative_bits if negative else positive_bits, ir_type=ir_type)
    for tag, dtype, bits, positive_bits, negative_bits, ir_type in (
        ("fp16", tl.float16, 16, 0x7C00, 0xFC00, "f16"),
        ("bf16", tl.bfloat16, 16, 0x7F80, 0xFF80, "bf16"),
        ("fp32", tl.float32, 32, 0x7F800000, 0xFF800000, "f32"),
    )
    for negative in (False, True)
    for in_vf in (False, True)
]


def constexprs(case):
    return dict(DTYPE=case["dtype"], N=2048 // case["bits"], IN_VF=case["in_vf"], NEGATIVE=case["negative"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(infinity_scalar, {"Out": "*" + case["tag"]}, constexprs(case), output_dir=dump)
    text = str(lower_offline(module, ctx, output_dir=dump))
    assert f"arith.constant 0x{case['expected_bits']:X} : {case['ir_type']}" in text
    assert "arith.select" not in text
    assert "arith.subf" not in text  # Unary minus must fold to a typed -Inf.
    assert "tensor<" not in text and "tt." not in text
    assert "arith.extf" not in text and "arith.truncf" not in text
    assert text.count("memref.alloc()") == 1  # Only explicit output storage.
    assert "ave.hir.broadcast" in text and "hivm.hir.store" in text
    assert f"ave.hir.masked_store <NORM_B{case['bits']}>" in text
    print("IR PASS: infinity", case["name"])


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[case["tag"]]
    return torch.zeros(2048 // case["bits"], dtype=dtype, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    out = _new_output(case)
    compiled = infinity_scalar.warmup(out, **constexprs(case), grid=(1, ))
    check_kernel_binary(compiled)


@matrix(CONFIGS)
def test_run(case):
    import torch

    out = _new_output(case)
    infinity_scalar[(1, )](out, **constexprs(case))
    torch.npu.synchronize()
    actual = out.cpu()
    # Inspect every element's IEEE bits, including the sign and zero mantissa.
    bits = case["expected_bits"]
    signed_bits = bits - (1 << case["bits"]) if case["negative"] else bits
    integer_dtype = torch.int16 if case["bits"] == 16 else torch.int32
    actual_bits = actual.view(integer_dtype)
    assert torch.equal(actual_bits, torch.full_like(actual_bits, signed_bits))
    print("INFINITY OK:", case["name"])


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
