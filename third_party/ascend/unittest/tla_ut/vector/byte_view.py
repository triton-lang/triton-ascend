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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Overlapping typed UB views preserve storage identity and access ordering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, check_kernel_binary, get_stages, matrix

CONFIGS = [dict(name=f"offset_{offset}", offset=offset) for offset in (0, 128)]
SIGNATURE = dict(Initial="*i8", X="*fp32", Y="*bf16", A="*fp32", B="*bf16", Backing="*i8")


@tla.jit
def overlapping_byte_views(Initial, X, Y, A, B, Backing, OFFSET: tl.constexpr):
    with tla.scope.vector():
        storage = tla.alloc((1024, ), tl.int8, space="ub")
        first = tla.byte_view(storage, (128, ), tl.float32, byte_offset=OFFSET)
        second = tla.byte_view(storage, (128, ), tl.bfloat16, byte_offset=OFFSET)
        tla.load_gm_to_ub(tla.view(Initial, (1024, )), storage)
        tla.load_gm_to_ub(tla.view(X, (128, )), first)
        tla.store_ub_to_gm(first, tla.view(A, (128, )))
        # Reuse the first half of the same bytes only after the previous read.
        tla.load_gm_to_ub(tla.view(Y, (128, )), second)
        tla.store_ub_to_gm(second, tla.view(B, (128, )))
        tla.store_ub_to_gm(storage, tla.view(Backing, (1024, )))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(overlapping_byte_views, SIGNATURE, dict(OFFSET=case["offset"]), output_dir=output_dir)
    ast = str(module)
    assert ast.count("memref.view ") == 2
    assert ast.count("memref.alloc()") == 1
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.load ") == text.count("hivm.hir.store ") == 3
    assert "__tla_native_" not in text and "!tt.ptr" not in text


def _launch(case, seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randint(-128, 128, (1024, ), generator=generator, dtype=torch.int8)
    x = torch.randn((128, ), generator=generator, dtype=torch.float32)
    y = torch.randn((128, ), generator=generator).to(torch.bfloat16)
    a = torch.full((128, ), float("nan"), dtype=torch.float32, device="npu")
    b = torch.full((128, ), float("nan"), dtype=torch.bfloat16, device="npu")
    backing = torch.empty((1024, ), dtype=torch.int8, device="npu")
    args = (initial.npu(), x.npu(), y.npu(), a, b, backing)
    if compile_only:
        return overlapping_byte_views.warmup(*args, OFFSET=case["offset"], grid=(1, )), None, None
    kernel = overlapping_byte_views[(1, )](*args, OFFSET=case["offset"])
    torch.npu.synchronize()
    expected = initial.clone()
    start = case["offset"]
    expected[start:start + 512] = x.view(torch.int8)
    expected[start:start + 256] = y.view(torch.int8)
    return kernel, (a.cpu(), b.cpu(), backing.cpu()), (x, y, expected)


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _ = _launch(case, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    for seed in (7, 19):
        _, actual, expected = _launch(case, seed)
        for result, reference in zip(actual, expected):
            assert torch.equal(result, reference), "typed alias read or backing bytes differ"
        print("ACCURACY PASS: byte_view", case["name"], "seed", seed, "exact values and guards")


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
