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
"""Dynamic-dimension views: GM pointer entry, GM-buffer subview, UB subview.

Every mode copies the same window ``X[:rows, off:off+D]`` out of a pitched
``(H, DPAD)`` GM tensor, so accuracy is mode-independent while each mode
exercises a different dynamic-dimension path of the view/subview split:

- ``entry_dyn``: ``tla.view`` on the GM kernel pointer with runtime shape,
  stride and offset; ``tt.make_tensor_ptr`` must carry the runtime operands.
- ``gm_subview``: ``tla.subview`` applied to a GM view (the buffer-space
  bridge path) with runtime sizes and offset.
- ``ub_dyn``: pitched ``tla.subview`` out of a UB allocation with runtime
  sizes and offset.

Dynamic entries map to ``?`` in the memref types and to runtime operands of
``memref.subview`` / ``tt.make_tensor_ptr``. Run-stage launches vary ``rows``
and ``off`` with values that share one specialization, so the runtime values
must be honored rather than constant-folded.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_lowered_ir, check_kernel_binary, matrix

H, D, DPAD = 8, 128, 256

CONFIGS = [
    dict(name=f"{dtype}_{mode}", dtype=dtype, mode=mode)
    for dtype in ("fp16", "bf16")
    for mode in ("entry_dyn", "gm_subview", "ub_dyn")
]
DTYPES = {"fp16": (tl.float16, torch.float16), "bf16": (tl.bfloat16, torch.bfloat16)}


def constants(case):
    return dict(DTYPE=DTYPES[case["dtype"]][0], MODE=case["mode"], H=H, D=D, DPAD=DPAD)


def signature(case):
    return {"X": "*" + case["dtype"], "O": "*" + case["dtype"], "rows": "i32", "off": "i32", "pitch": "i32"}


@tla.jit
def dynamic_window(X, O, rows, off, pitch, DTYPE: tl.constexpr, MODE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   DPAD: tl.constexpr):
    with tla.scope.vector():
        if MODE == "ub_dyn":
            # Whole pitched tensor into UB, then a dynamic window out of it.
            ub = tla.alloc((H, DPAD), DTYPE, space="ub")
            tla.load_gm_to_ub(tla.view(X, (H, DPAD), strides=(pitch, 1)), ub)
            tile = tla.subview(ub, offsets=(0, off), sizes=(rows, D))
        else:
            if MODE == "gm_subview":
                # Slice the GM view itself: subview on a GM-space buffer.
                base = tla.view(X, (H, DPAD), strides=(DPAD, 1))
                src = tla.subview(base, offsets=(0, off), sizes=(rows, D))
            else:  # entry_dyn: runtime shape, stride and offset on the entry.
                src = tla.view(X - D, (rows, D), strides=(pitch, 1), offsets=(0, off + D))
            ub = tla.alloc((H, D), DTYPE, space="ub")
            tile = tla.subview(ub, offsets=(0, 0), sizes=(rows, D))
            tla.load_gm_to_ub(src, tile)
        tla.store_ub_to_gm(tile, tla.view(O, (rows, D), strides=(D, 1)))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(dynamic_window, signature(case), constants(case), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    # Every mode routes exactly two GM pointer entries (source side + store side).
    assert text.count("tt.make_tensor_ptr") == 2
    # Every bridge is marked, along with each entry with a runtime size or stride.
    assert text.count("tla.gm_view") == (3 if case["mode"] == "gm_subview" else 4)
    assert text.count("memref.alloc()") == 1, "views must share one allocation"
    if case["mode"] == "entry_dyn":
        assert text.count("memref.subview") == 1
        assert "strided<[?, 1], offset: ?>" in text, "runtime pitch must stay dynamic on the entry"
    elif case["mode"] == "gm_subview":
        assert text.count("memref.subview") == 2, "GM slice + UB tile carve"
        assert "strided<[256, 1], offset: ?>" in text, "pitched GM subview result"
        assert "strided<[128, 1], offset: ?>" in text, "UB tile carve result"
    else:  # ub_dyn
        assert text.count("memref.subview") == 1
        assert "strided<[?, 1], offset: ?>" in text, "runtime pitch must stay dynamic on the entry"
        assert "strided<[256, 1], offset: ?>" in text, "pitched UB subview result"
    # Dynamic extents map to '?' in the memref types instead of specializing.
    assert "memref<?x128x" in text
    lowered = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(lowered, loads=1, stores=1)
    assert "memref.reinterpret_cast" in lowered
    assert "tla.gm_view" not in lowered
    if case["mode"] != "gm_subview":
        assert "strided<[?, 1]" in lowered
    print("IR PASS: view_dynamic", case["name"])


def _launch(case, rows, off, pitch=DPAD, *, compile_only=False):
    dtype = DTYPES[case["dtype"]][1]
    generator = torch.Generator().manual_seed(23)
    x = torch.randint(-127, 128, (H, pitch), generator=generator).to(dtype)
    out = torch.full((rows, D), float("nan"), dtype=dtype, device="npu")
    args = (x.npu(), out, rows, off, pitch)
    if compile_only:
        return dynamic_window.warmup(*args, grid=(1, ), **constants(case)), None
    kernel = dynamic_window[(1, )](*args, **constants(case))
    torch.npu.synchronize()
    return kernel, out.cpu(), x[:rows, off:off + D].cpu()


@matrix(CONFIGS)
def test_compile(case):
    kernel, _ = _launch(case, H, 0, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    # (rows, off) pairs share one int specialization (none is 1 or %16==0
    # except the offsets, which are all %16==0), so a single compiled kernel
    # must honor the runtime values.
    for rows, off in ((H, 0), (H, D), (H // 2, D)):
        pitch = DPAD + D if off and case["mode"] != "gm_subview" else DPAD
        _, actual, expected = _launch(case, rows, off, pitch)
        assert torch.equal(actual, expected), \
            f"dynamic window addressed the wrong elements: rows={rows} off={off}"
        print("ACCURACY PASS: view_dynamic", case["name"], f"rows={rows} off={off} exact window match")


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
