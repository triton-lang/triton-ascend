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
"""Runtime row pitch/count with nested source offsets and pitched UB guards."""
import sys
from functools import partial
from pathlib import Path

# Do not let dma/copy.py shadow Python's standard library in standalone runs.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, check_kernel_binary, get_stages, matrix

CONFIGS = [
    dict(name=f"{dtype}_{mode}", dtype=dtype, mode=mode, width=512 if mode == "wide" else 64)
    for dtype in ("fp16", "bf16")
    for mode in ("static", "dynamic", "wide", "zero_base")
]
DTYPES = {"fp16": (tl.float16, torch.float16), "bf16": (tl.bfloat16, torch.bfloat16)}


def constants(case):
    return dict(DTYPE=DTYPES[case["dtype"]][0], WIDTH=case["width"], DYNAMIC=case["mode"] != "static",
                ZERO_BASE=case["mode"] == "zero_base")


def signature(case):
    return dict(X="*" + case["dtype"], Initial="*" + case["dtype"], Backing="*" + case["dtype"], pointer_offset="i32",
                entry_offset="i32", sub_offset="i32", offset="i32", stride="i32", rows="i32")


@partial(tla.jit, do_not_specialize=["rows"])
def copy_strided_rows(X, Initial, Backing, pointer_offset, entry_offset, sub_offset, offset, stride, rows,
                      DTYPE: tl.constexpr, WIDTH: tl.constexpr, DYNAMIC: tl.constexpr, ZERO_BASE: tl.constexpr):
    with tla.scope.vector():
        storage = tla.alloc((4, WIDTH + 32), DTYPE, space="ub")
        tla.load_gm_to_ub(tla.view(Initial, (4, WIDTH + 32)), storage)
        tile = tla.subview(storage, (1, 16), (2, WIDTH))
        # Each address layer contributes: pointer arithmetic, pointer-entry
        # offset, subview offset, then the load's own relative offset.
        if ZERO_BASE:
            base = tla.view(X, (4096, ))
        else:
            entry = tla.view(X + pointer_offset, (6144, ), offsets=(entry_offset, ))
            base = tla.subview(entry, (sub_offset, ), (4096, ))
        if DYNAMIC:
            tla.load_gm_to_ub_strided(base, tile, offset, stride, rows=rows)
        else:
            tla.load_gm_to_ub_strided(base, tile, 128, 192)
        tla.store_ub_to_gm(storage, tla.view(Backing, (4, WIDTH + 32)))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(copy_strided_rows, signature(case), constants(case), output_dir=output_dir)
    ast = str(module)
    assert "memref.extract_strided_metadata" in ast
    assert ast.count("memref.alloc()") == 1
    assert "memref.subview" in ast and "memref.reinterpret_cast" in ast
    if case["mode"] != "static":
        # The GM entry has static capacity; runtime shape is a subview.
        assert f"memref<2x{case['width']}x" in ast
        assert f"memref<?x{case['width']}x" in ast
        assert "strided<[?, 1], offset: ?>" in ast
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.load ") == 2
    assert text.count("hivm.hir.store ") == 1
    assert "__tla_native_" not in text and "!tt.ptr" not in text


def _launch(case, seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    dtype = DTYPES[case["dtype"]][1]
    width = case["width"]
    source_size = 4096 if case["mode"] == "zero_base" else 8192
    x = torch.randint(-127, 128, (source_size, ), generator=generator).to(dtype)
    initial = torch.randint(-127, 128, (4, width + 32), generator=generator).to(dtype)
    result = torch.full(initial.shape, float("nan"), dtype=dtype, device="npu")
    pointer_offset, entry_offset, sub_offset = 32, 128, 64 if seed == 7 else 128
    count = 1 if seed == 7 else 2
    offset, stride = 128, width * (2 if seed == 7 else 3)
    if case["mode"] == "zero_base" and count == 1:
        # Only the requested first row is valid; a second read would be OOB.
        offset = source_size - width
    args = (x.npu(), initial.npu(), result, pointer_offset, entry_offset, sub_offset, offset, stride, count)
    if compile_only:
        return copy_strided_rows.warmup(*args, grid=(1, ), **constants(case)), None, None
    kernel = copy_strided_rows[(1, )](*args, **constants(case))
    # In particular rows=1 must not become a constexpr: both seeds exercise
    # the same runtime-count DMA path, including the final valid source row.
    assert kernel.src.signature["rows"] in ("i32", "i64")
    torch.npu.synchronize()
    if case["mode"] == "static":
        count, stride = 2, 192
    expected = initial.clone()
    start = offset if case["mode"] == "zero_base" else pointer_offset + entry_offset + sub_offset + offset
    for row in range(count):
        expected[1 + row, 16:16 + width] = x[start + row * stride:start + row * stride + width]
    return kernel, result.cpu(), expected


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _ = _launch(case, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    for seed in (7, 19):
        _, actual, expected = _launch(case, seed)
        assert torch.equal(actual, expected), "strided transfer lost offsets, pitch, count or untouched guards"
        print("ACCURACY PASS: strided_load", case["name"], "seed", seed, "exact rows and guards")


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
