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
"""Check TLA program IDs at the adapter boundary and on a three-axis grid."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import triton.language as tl
from triton.language.extra import tla
from triton.language.semantic import TritonSemantic
from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline


@tla.jit
def print_program_id(AXIS: tl.constexpr):
    pid = tla.program_id(AXIS)
    tl.static_assert(pid.dtype == tl.int32)
    tla.device_print("pid", pid)


@tla.jit
def write_program_ids(out):
    x, y, z = tla.program_id(0), tla.program_id(1), tla.program_id(2)
    linear_id = x + 2 * y + 4 * z
    ub = tla.alloc((64, ), tl.int32)
    with tla.scope.vector():
        with tla.scope.simd():
            tla.simd.vsts(ub[0], tla.simd.vdup(linear_id, tl.int32), dist="NORM_B32")
        tla.store_ub_to_gm(ub, tla.view(out + 64 * linear_id, (64, )))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_ir(axis):
    module, ctx = compile_offline(print_program_id, {}, {"AXIS": axis})
    lowered = lower_offline(module, ctx)
    ops = []
    lowered.walk(ops.append)
    assert all(op.get_name() != "tt.get_program_id" for op in ops)
    debug, = (op for op in ops if op.get_name() == "hivm.hir.debug")
    block = debug.get_block()
    argument = block.get_argument(block.get_num_arguments() - 3 + axis)
    assert debug.get_operand(0).id() == argument.id()
    assert str(argument.get_type()) == "i32"


def test_invalid_axes():
    for axis in (-1, 3):
        with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
            tla.program_id(axis, _semantic=TritonSemantic(None))
    for axis in (True, 1.0, tl.tensor(None, tl.int32)):
        with pytest.raises(TypeError, match="axis must be a compile-time integer"):
            tla.program_id(axis, _semantic=object())


def _output():
    import torch
    import torch_npu  # noqa: F401

    return torch.full((8, 64), -1, dtype=torch.int32, device="npu")


def test_compile():
    check_kernel_binary(write_program_ids.warmup(_output(), grid=(2, 2, 2)))


def test_run():
    import torch

    out = _output()
    write_program_ids[(2, 2, 2)](out)
    torch.npu.synchronize()
    expected = torch.arange(8, dtype=torch.int32)[:, None].expand_as(out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        for axis in (0, 1, 2):
            test_ir(axis)
        test_invalid_axes()
    if do_compile:
        test_compile()
    if do_run:
        test_run()


if __name__ == "__main__":
    main()
