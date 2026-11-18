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
"""Verify that both GM address-space spellings preserve the pointer ABI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.language.extra import tla
from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline, matrix


@tla.jit
def gm_view_copy(source, output, offset, SPACE: tl.constexpr):
    with tla.scope.vector():
        ub = tla.alloc((64, ), tl.float32)
        src = tla.view(source, (64, ), offsets=(offset, ), space=SPACE)
        tla.load_gm_to_ub(src, ub)
        tla.store_ub_to_gm(ub, tla.view(output, (64, ), space=SPACE))


_CASES = [dict(name="string_gm", space="gm"), dict(name="enum_gm", space=tla.AddressSpace.GM)]


@matrix(_CASES)
def test_ir(case):
    sig = {"source": "*fp32", "output": "*fp32", "offset": "i32"}
    module, context = compile_offline(gm_view_copy, sig, {"SPACE": case["space"]})
    text = str(lower_offline(module, context))
    assert "hivm.hir.load" in text and "hivm.hir.store" in text
    assert "hivm.hir.pointer_cast" not in text


@matrix(_CASES)
def test_reject_non_byte_dtype(case):
    sig = {"source": "*i1", "output": "*fp32", "offset": "i32"}
    with pytest.raises(CompilationError, match="byte-addressable"):
        compile_offline(gm_view_copy, sig, {"SPACE": case["space"]})


def _buffers():
    import torch
    import torch_npu  # noqa: F401

    return torch.arange(128, dtype=torch.float32, device="npu"), torch.empty(64, dtype=torch.float32, device="npu")


@matrix(_CASES)
def test_compile(case):
    source, output = _buffers()
    check_kernel_binary(gm_view_copy.warmup(source, output, 64, SPACE=case["space"], grid=(1, )))


@matrix(_CASES)
def test_run(case):
    import torch

    source, output = _buffers()
    for offset in (0, 64):
        gm_view_copy[(1, )](source, output, offset, SPACE=case["space"])
        torch.npu.synchronize()
        expected = torch.arange(offset, offset + 64, dtype=torch.float32)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in _CASES:
        if do_ir:
            test_ir(case)
            test_reject_non_byte_dtype(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
