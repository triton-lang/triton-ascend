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
"""tla.fill 3-in-1 test: scalar UB fill through hivm.hir.vbrc.

Scope of the first version: UB buffers with a contiguous row-major layout and
offset 0 -- directly from ``tla.alloc`` or from a legal ``tla.reshape``.  GM,
L1, L0A/L0B/L0C/BT/FB, pitched layouts, subview offsets and i1 are rejected
at trace time and covered by ``fill_frontend_contract.py``.

IR expects exactly one scalar ``hivm.hir.vbrc`` with no ``broadcast_dims``,
one ``memref.alloc`` and no ``linalg.fill``.  Run compares the UB value copied
back to GM against a CPU reference.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from triton.compiler.errors import CompilationError

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary, matrix

N = 256
DTYPES = {
    "f32": dict(dtype=tl.float32, torch_dtype=torch.float32, signature="*fp32", values=(0.0, 1.0, -1.0)),
    "bf16": dict(dtype=tl.bfloat16, torch_dtype=torch.bfloat16, signature="*bf16", values=(0.0, 1.0, -1.0)),
    "i32": dict(dtype=tl.int32, torch_dtype=torch.int32, signature="*i32", values=(0, 1, -1)),
}


def _configs():
    cases = []
    for tag, spec in DTYPES.items():
        for index, value in enumerate(spec["values"]):
            cases.append(dict(name=f"{tag}_{index}", tag=tag, value=value, shape=(N, ), reshape=False))
        cases.append(dict(name=f"{tag}_2d", tag=tag, value=spec["values"][1], shape=(16, 16), reshape=False))
        cases.append(dict(name=f"{tag}_reshape", tag=tag, value=spec["values"][1], shape=(N, ), reshape=True))
    return cases


CONFIGS = _configs()


@tla.jit
def fill_kernel(Y, SHAPE: tl.constexpr, DTYPE: tl.constexpr, VALUE: tl.constexpr, RESHAPE: tl.constexpr):
    with tla.scope.vector():
        buf = tla.alloc(SHAPE, DTYPE, space="ub")
        if RESHAPE:
            dst = tla.reshape(buf, (SHAPE[0] // 4, 4))
            tla.fill(dst, VALUE)
            tla.store_ub_to_gm(dst, tla.view(Y, (SHAPE[0] // 4, 4)))
        else:
            tla.fill(buf, VALUE)
            tla.store_ub_to_gm(buf, tla.view(Y, SHAPE))


def signature(case):
    return {"Y": DTYPES[case["tag"]]["signature"]}


def constants(case):
    return dict(SHAPE=case["shape"], DTYPE=DTYPES[case["tag"]]["dtype"], VALUE=case["value"], RESHAPE=case["reshape"])


def _vbrc_element_types(text):
    """(ins element type, outs element type) of the single scalar vbrc."""
    line = next(line for line in text.splitlines() if "hivm.hir.vbrc " in line)
    match = re.search(r"ins\(.*?:\s*([^)]+)\)\s*outs\(.*?: memref<[^,>]*x([^,>]+)", line)
    assert match, line
    return match.group(1).strip(), match.group(2).strip()


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(fill_kernel, signature(case), constants(case), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.vbrc ") == 1, "one scalar broadcast per fill"
    assert text.count("memref.alloc()") == 1, "fill must not allocate"
    assert text.count("hivm.hir.store ") == 1, "fill result must be stored back"
    assert "broadcast_dims" not in text, "scalar vbrc must not set broadcast_dims"
    assert "linalg.fill" not in text, "fill must lower through hivm.hir.vbrc"
    ins_type, out_type = _vbrc_element_types(text)
    assert ins_type == out_type, f"scalar type {ins_type} != destination element type {out_type}"
    assert text.count("memref.expand_shape") == (1 if case["reshape"] else 0)
    assert "#hivm.address_space<ub>" in text
    print("IR PASS: fill", case["name"], f"ins {ins_type}")


def _launch(case, compile_only=False):
    torch_dtype = DTYPES[case["tag"]]["torch_dtype"]
    init = 12345 if case["tag"] == "i32" else float("nan")
    y = torch.full(case["shape"], init, dtype=torch_dtype)
    device_y = y.npu()
    args = (device_y, )
    if compile_only:
        return fill_kernel.warmup(*args, grid=(1, ), **constants(case))
    kernel = fill_kernel[(1, )](*args, **constants(case))
    torch.npu.synchronize()
    # Return the device tensor: ``y`` is a separate host allocation and is
    # never written by the kernel.
    return kernel, device_y


@matrix(CONFIGS)
def test_compile(case):
    check_kernel_binary(_launch(case, compile_only=True))


@matrix(CONFIGS)
def test_run(case):
    _, actual = _launch(case)
    actual = actual.cpu()
    expected = torch.full(case["shape"], case["value"], dtype=actual.dtype)
    assert torch.equal(actual, expected), "fill did not cover the whole buffer"
    print("ACCURACY PASS: fill", case["name"], "value", case["value"])


@tla.jit
def fill_subview(Y, SIZE: tl.constexpr):
    """Minimal reproduction of the unimplemented offset case."""
    with tla.scope.vector():
        buf = tla.alloc((SIZE, ), tl.float32, space="ub")
        sub = tla.subview(buf, offsets=(16, ), sizes=(SIZE - 16, ))
        tla.fill(sub, 1.0)
        tla.store_ub_to_gm(sub, tla.view(Y, (SIZE - 16, )))


def test_subview_offset_is_rejected():
    # tla.subview records offset=None even for a static subview, so this must
    # stay rejected until the offset is carried in the buffer type and the
    # vbrc lowering is verified against it.  Do not drop the offset to force
    # this through.
    with pytest.raises(CompilationError, match="offset 0"):
        compile_offline(fill_subview, {"Y": "*fp32"}, {"SIZE": N})
    print("IR PASS: fill static subview offset rejected at trace time")


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
