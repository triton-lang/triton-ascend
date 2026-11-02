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
"""Native UB scalar reads after vector writes, including strided aliases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, check_kernel_binary, get_stages, matrix

CONFIGS = [dict(name=mode, mode=mode) for mode in ("direct", "linear", "strided", "matrix")]
SIGNATURE = dict(X="*i32", O="*i32", row="i32", col="i32")


@tla.jit
def scalar_read_after_vector_write(X, O, row, col, MODE: tl.constexpr):
    with tla.scope.vector():
        src = tla.alloc((256, ), tl.int32, space="ub")
        out = tla.alloc((64, ), tl.int32, space="ub")
        tla.load_gm_to_ub(tla.view(X, (256, )), src)
        if MODE != "direct":
            with tla.scope.simd():
                delta = tla.simd.vfull(17, tl.int32, 64)
                for chunk in tl.static_range(4):
                    value = tla.simd.vadd(tla.simd.vlds(src[chunk * 64]), delta)
                    tla.simd.vsts(src[chunk * 64], value)
        if MODE == "strided":
            alias = tla.subview(src, (9, ), (64, ), (2, ))
            value = tla.scalar_load(alias, row)
        elif MODE == "matrix":
            alias = tla.reshape(src, (16, 16))
            value = tla.scalar_load(alias[row, col])
        else:
            value = tla.scalar_load(src, row)
        with tla.scope.simd():
            tla.simd.vsts(out[0], tla.simd.vdup(value, tl.int32))
        tla.store_ub_to_gm(out, tla.view(O, (64, )))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(scalar_read_after_vector_write, SIGNATURE, dict(MODE=case["mode"]),
                                  output_dir=output_dir)
    assert str(module).count("memref.load ") == 1
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert text.count("memref.load ") == 1
    assert text.count("hivm.hir.load ") == text.count("hivm.hir.store ") == 1
    assert "__tla_native_" not in text and "!tt.ptr" not in text


def _launch(case, seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(-10000, 10000, (256, ), generator=generator, dtype=torch.int32)
    row, col = (seed * 3) % 16, (seed * 5) % 16
    inputs = x.npu()
    out = torch.full((64, ), -123456789, dtype=torch.int32, device="npu")
    args = (inputs, out, row, col)
    if compile_only:
        return scalar_read_after_vector_write.warmup(*args, MODE=case["mode"], grid=(1, )), None, None
    kernel = scalar_read_after_vector_write[(1, )](*args, MODE=case["mode"])
    torch.npu.synchronize()
    index = row if case["mode"] in ("direct",
                                    "linear") else 9 + row * 2 if case["mode"] == "strided" else row * 16 + col
    expected = torch.full((64, ), x[index].item() + (0 if case["mode"] == "direct" else 17), dtype=torch.int32)
    return kernel, out.cpu(), expected


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _ = _launch(case, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    for seed in (7, 19):
        _, actual, expected = _launch(case, seed)
        assert torch.equal(actual, expected), "scalar read lost an offset, stride or producer dependency"
        print("ACCURACY PASS: scalar_load", case["name"], "seed", seed, "exact match")


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
