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
"""TLA 3-in-1 test: host-side ``tla.buffer`` kernel parameters.

The kernels declare their GM inputs/outputs in the signature and the host
passes tensors:

- ``host_copy``: typed ``tla.buffer_type`` parameters, 2-D copy;
- ``host_add``: typed parameters feeding a SIMD vector add;
- ``bare_copy``: bare ``tla.buffer`` parameters whose type comes from the
  tensors, launched with two different shapes;
- ``mixed_copy``: one annotated buffer plus a plain ``*fp32`` pointer.

The IR stage compiles the runtime-style signatures (``tlabuf:...`` entries
produced by ``@tla.jit``) and checks the ABI boundary: annotated buffers are
GM memrefs, plain pointers stay ``!tt.ptr``.  The compile/run stages launch
with real tensors and check the results plus the launch-time validation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.host import buffer_spec
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)


@tla.jit
def host_copy(x: tla.buffer_type((2, 32), tl.float32), y: tla.buffer_type((2, 32), tl.float32)):
    ub = tla.alloc((2, 32), tl.float32)
    tla.copy(x, ub)
    tla.copy(ub, y)


@tla.jit
def host_add(x: tla.buffer_type((64, ), tl.float32), y: tla.buffer_type((64, ), tl.float32), out: tla.buffer_type(
    (64, ), tl.float32)):
    ux = tla.alloc((64, ), tl.float32)
    uy = tla.alloc((64, ), tl.float32)
    uo = tla.alloc((64, ), tl.float32)
    tla.copy(x, ux)
    tla.copy(y, uy)
    with tla.scope.simd():
        tla.simd.vsts(uo[0], tla.simd.vadd(tla.simd.vlds(ux[0]), tla.simd.vlds(uy[0])))
    tla.copy(uo, out)


@tla.jit
def host_copy_bf16(x: tla.buffer_type((2, 32), tl.bfloat16), y: tla.buffer_type((2, 32), tl.bfloat16)):
    ub = tla.alloc((2, 32), tl.bfloat16)
    tla.copy(x, ub)
    tla.copy(ub, y)


@tla.jit
def bare_copy(x: tla.buffer, y: tla.buffer, M: tl.constexpr, N: tl.constexpr):
    ub = tla.alloc((M, N), tl.float32)
    tla.copy(x, ub)
    tla.copy(ub, y)


@tla.jit
def mixed_copy(x: tla.buffer_type((2, 32), tl.float32), y):
    ub = tla.alloc((2, 32), tl.float32)
    tla.copy(x, ub)
    tla.copy(ub, tla.view(y, (2, 32)))


def signature(shape, dtype=tl.float32):
    return buffer_spec(tla.buffer_type(shape, dtype))


def entry(text):
    return next(line for line in text.splitlines() if "tt.func public" in line)


def test_ir(output_dir=None):
    module, ctx = compile_offline(host_copy, {"x": signature((2, 32)), "y": signature((2, 32))}, {},
                                  output_dir=output_dir)
    text = str(module)
    assert "%x: memref<2x32xf32, #hivm.address_space<gm>>" in text
    assert "%y: memref<2x32xf32, #hivm.address_space<gm>>" in text
    assert "tt.ptr" not in text
    check_lowered_ir(str(lower_offline(module, ctx, output_dir=output_dir)), loads=1, stores=1, buffers=None)

    # The runtime signature entry and the buffer_type object compile to the same entry.
    objects, _ = compile_offline(host_copy,
                                 {"x": tla.buffer_type((2, 32), tl.float32), "y": tla.buffer_type(
                                     (2, 32), tl.float32)}, {})
    assert entry(str(objects)) == entry(text)

    module, ctx = compile_offline(host_add, {name: signature((64, ))
                                             for name in ("x", "y", "out")}, {}, output_dir=output_dir)
    text = str(module)
    assert text.count("memref<64xf32, #hivm.address_space<gm>>") == 6
    assert "vector<64xf32>" in text and "ave.hir.vadd" in text
    check_lowered_ir(str(lower_offline(module, ctx, output_dir=output_dir)), loads=2, stores=1, buffers=None)

    module, ctx = compile_offline(host_copy_bf16,
                                  {"x": signature((2, 32), tl.bfloat16), "y": signature((2, 32), tl.bfloat16)}, {})
    text = str(module)
    assert text.count("memref<2x32xbf16, #hivm.address_space<gm>>") == 4
    lower_offline(module, ctx, output_dir=output_dir)

    module, ctx = compile_offline(bare_copy, {"x": signature((4, 16)), "y": signature((4, 16))}, {"M": 4, "N": 16},
                                  output_dir=output_dir)
    assert "%x: memref<4x16xf32, #hivm.address_space<gm>>" in str(module)
    lower_offline(module, ctx, output_dir=output_dir)

    module, ctx = compile_offline(mixed_copy, {"x": signature((2, 32)), "y": "*fp32"}, {}, output_dir=output_dir)
    text = str(module)
    assert "%x: memref<2x32xf32, #hivm.address_space<gm>>" in text
    assert "%y: !tt.ptr<f32>" in text
    check_lowered_ir(str(lower_offline(module, ctx, output_dir=output_dir)), loads=1, stores=1, buffers=None)
    print("IR PASS: typed, bare and mixed host buffers reach their kernel signatures")


def _launch_copy():
    x = torch.randn((2, 32), dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    host_copy[(1, )](x, y)
    torch.npu.synchronize()
    return y.cpu(), x.cpu()


def _launch_add():
    x = torch.randn((64, ), dtype=torch.float32, device="npu")
    y = torch.randn((64, ), dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    host_add[(1, )](x, y, out)
    torch.npu.synchronize()
    return out.cpu(), (x + y).cpu()


def _launch_bf16():
    x = torch.randn((2, 32), dtype=torch.bfloat16, device="npu")
    y = torch.empty_like(x)
    host_copy_bf16[(1, )](x, y)
    torch.npu.synchronize()
    return y.cpu(), x.cpu()


def _launch_bare():
    results = []
    for shape in ((2, 32), (4, 16)):
        x = torch.randn(shape, dtype=torch.float32, device="npu")
        y = torch.empty_like(x)
        bare_copy[(1, )](x, y, M=shape[0], N=shape[1])
        torch.npu.synchronize()
        results.append((y.cpu(), x.cpu()))
    return results


def _launch_mixed():
    x = torch.randn((2, 32), dtype=torch.float32, device="npu")
    y = torch.empty((2, 32), dtype=torch.float32, device="npu")
    mixed_copy[(1, )](x, y)
    torch.npu.synchronize()
    return y.cpu(), x.cpu()


def _check():
    got, ref = _launch_copy()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    got, ref = _launch_add()
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)
    got, ref = _launch_bf16()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    for got, ref in _launch_bare():
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    got, ref = _launch_mixed()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_compile():
    _check()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    # Launch-time adaptation rejects values that do not match the annotation.
    y = torch.empty((2, 32), dtype=torch.float32, device="npu")
    with pytest.raises(TypeError, match="expects fp32"):
        host_copy[(1, )](torch.zeros((2, 32), dtype=torch.int32, device="npu"), y)
    with pytest.raises(ValueError, match="expects shape"):
        host_copy[(1, )](torch.zeros((2, 16), dtype=torch.float32, device="npu"), y)
    print("COMPILE PASS: host buffers launched", len(artifacts), "artifacts ->", artifacts[0])


def test_run():
    _check()
    print("HOST OK: typed, bare and mixed tla.buffer arguments")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
