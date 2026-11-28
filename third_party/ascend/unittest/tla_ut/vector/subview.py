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
"""Subview/reshape composition: IR, compilation and exact device alias checks.

The compact case mirrors FA's (8, 2064) allocation and (8, 1040) prefix.
All cases write through the alias and read both the alias and the complete
allocation, so preserved offsets, row pitches and untouched guards are checked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary, matrix

CONFIGS = [
    dict(name=f"{dtype}_{mode}", dtype=dtype, mode=mode)
    for dtype in ("fp16", "bf16")
    for mode in ("prefix", "static_offset", "dynamic_offset", "nested_offset", "pitched")
]
DTYPES = {"fp16": (tl.float16, torch.float16), "bf16": (tl.bfloat16, torch.bfloat16)}


def constants(case):
    return dict(DTYPE=DTYPES[case["dtype"]][0], MODE=case["mode"])


def signature(case):
    return {
        "X": "*" + case["dtype"], "Y": "*" + case["dtype"], "O": "*" + case["dtype"], "Backing": "*" + case["dtype"],
        "offset": "i32"
    }


@tla.jit
def subview_alias(X, Y, O, Backing, offset, DTYPE: tl.constexpr, MODE: tl.constexpr):
    with tla.scope.vector():
        root = tla.alloc((8, 2064), DTYPE, space="ub")
        tla.load_gm_to_ub(tla.view(X, (8, 2064)), root)
        if MODE == "pitched":
            tile = tla.subview(root, offsets=(0, 0), sizes=(8, 1040))
        else:
            flat = tla.reshape(root, (16512, ))
            if MODE == "nested_offset":
                window = tla.subview(flat, offsets=(offset, ), sizes=(8448, ))
                prefix = tla.subview(window, offsets=(128, ), sizes=(8320, ))
            elif MODE == "dynamic_offset":
                prefix = tla.subview(flat, offsets=(offset, ), sizes=(8320, ))
            elif MODE == "static_offset":
                prefix = tla.subview(flat, offsets=(128, ), sizes=(8320, ))
            else:
                prefix = tla.subview(flat, offsets=(0, ), sizes=(8320, ))
            tile = tla.reshape(prefix, (8, 1040))
        tla.load_gm_to_ub(tla.view(Y, (8, 1040)), tile)
        tla.store_ub_to_gm(tile, tla.view(O, (8, 1040)))
        tla.store_ub_to_gm(root, tla.view(Backing, (8, 2064)))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(subview_alias, signature(case), constants(case), output_dir=output_dir)
    ast = str(module)
    assert "memref.subview" in ast
    assert ast.count("memref.alloc()") == 1, "views must share the original allocation"
    if case["mode"] != "pitched":
        assert "memref.collapse_shape" in ast and "memref.expand_shape" in ast
        assert "strided<[1040, 1], offset: ?>" in ast
    else:
        assert "memref.expand_shape" not in ast
        assert "strided<[2064, 1], offset: ?>" in ast
    if case["mode"] == "nested_offset":
        assert ast.count("memref.subview") == 2
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.load ") == 2
    assert text.count("hivm.hir.store ") == 2
    assert "!tt.ptr" not in text and "tensor<" not in text
    print("IR PASS: subview", case["name"])


def _launch(case, seed=7, compile_only=False):
    dtype = DTYPES[case["dtype"]][1]
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(-127, 128, (8, 2064), generator=generator).to(dtype)
    y = torch.randint(-127, 128, (8, 1040), generator=generator).to(dtype)
    inputs = (x.npu(), y.npu())
    out = torch.full((8, 1040), float("nan"), dtype=dtype, device="npu")
    backing = torch.full_like(inputs[0], float("nan"))
    offset = 256 if seed == 19 else 128
    args = (*inputs, out, backing, offset)
    if compile_only:
        return subview_alias.warmup(*args, grid=(1, ), **constants(case)), None, None, None
    kernel = subview_alias[(1, )](*args, **constants(case))
    torch.npu.synchronize()
    expected = x.clone()
    mode = case["mode"]
    if mode == "pitched":
        expected[:, :1040] = y
    else:
        start = {"prefix": 0, "static_offset": 128, "dynamic_offset": offset, "nested_offset": offset + 128}[mode]
        expected.flatten()[start:start + 8320] = y.flatten()
    return kernel, out.cpu(), backing.cpu(), (y, expected)


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _, _ = _launch(case, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    for seed in (7, 19):
        _, actual, backing, expected = _launch(case, seed)
        assert torch.equal(actual, expected[0]), "alias load/store changed values"
        assert torch.equal(backing, expected[1]), "alias addressed the wrong elements or overwrote guards"
        print("ACCURACY PASS: subview", case["name"], "seed", seed, "exact alias and guard match")


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
