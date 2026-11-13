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
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from triton.compiler.errors import CompilationError

from utils.tla_test_utils import check_kernel_binary, check_lowered_ir, compile_offline, lower_offline


@tla.jit
def histogram_kernel(X, Y, SIZE: tl.constexpr, BINS: tl.constexpr, REPEAT: tl.constexpr):
    with tla.scope.vector():
        src = tla.alloc((SIZE, ), tl.int32, space="ub")
        dst = tla.alloc((BINS, ), tl.int32, space="ub")
        tla.load_gm_to_ub(tla.view(X, (SIZE, )), src)
        tla.histogram(src, dst)
        if REPEAT:
            tla.histogram(src, dst)
        tla.store_ub_to_gm(dst, tla.view(Y, (BINS, )))


@pytest.mark.parametrize("bins", [1, 256, 257])
def test_ir(bins, output_dir):
    module, ctx = compile_offline(histogram_kernel, {"X": "*i32", "Y": "*i32"}, dict(SIZE=256, BINS=bins, REPEAT=False),
                                  output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(
        text, loads=1, stores=1, buffers=0,
        extra=('"__builtin_histogram"', "#hivm.vf_mode<SIMT>", "#hivm.pipe<PIPE_V>", f"arith.constant {bins} : i64"))
    assert text.count("memref.alloc()") == 2
    assert text.count("hivm.hir.custom ") == 1
    assert text.count("ave.hir.masked_store ") == 1
    assert text.index("ave.hir.masked_store ") < text.index("hivm.hir.custom ")


@tla.jit
def histogram_invalid(X, KIND: tl.constexpr):
    src = tla.alloc((32, ), tl.int32, space="ub")
    if KIND == "alias":
        tla.histogram(src, src)
    elif KIND == "rank":
        tla.histogram(src, tla.alloc((4, 8), tl.int32, space="ub"))
    elif KIND == "dtype":
        tla.histogram(tla.alloc((32, ), tl.float32, space="ub"), src)
    elif KIND == "space":
        tla.histogram(tla.view(X, (32, )), src)
    elif KIND == "bins":
        tla.histogram(src, tla.alloc((1025, ), tl.int32, space="ub"))
    elif KIND == "offset":
        tla.histogram(tla.subview(src, offsets=(1, ), sizes=(31, )), src)
    else:
        tla.histogram(X, src)


@pytest.mark.parametrize(("kind", "message"), [
    ("alias", "disjoint"),
    ("rank", "1D int32 UB buffer"),
    ("dtype", "1D int32 UB buffer"),
    ("space", "1D int32 UB buffer"),
    ("bins", "supports 1 to 1024 bins"),
    ("offset", "contiguous at offset 0"),
    ("pointer", "must be a tla.buffer"),
])
def test_invalid(kind, message):
    with pytest.raises(CompilationError, match=message):
        compile_offline(histogram_invalid, {"X": "*i32"}, {"KIND": kind})


device_cases = pytest.mark.parametrize(("size", "bins", "repeat"), [
    (17, 1, False),
    (256, 256, False),
    (1031, 257, True),
    (1031, 1024, True),
])


@device_cases
def test_compile(size, bins, repeat):
    x = torch.empty(size, dtype=torch.int32, device="npu")
    y = torch.empty(bins, dtype=torch.int32, device="npu")
    check_kernel_binary(histogram_kernel.warmup(x, y, size, bins, repeat, grid=(1, )))


@device_cases
def test_run(size, bins, repeat):
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(-4, bins + 4, (size, ), dtype=torch.int32, generator=generator)
    x[:8] = torch.tensor([0, 0, 0, bins - 1, bins, -1, -(2**31), 2**31 - 1], dtype=torch.int32)
    y = torch.full((bins, ), -1, dtype=torch.int32, device="npu")
    histogram_kernel[(1, )](x.npu(), y, size, bins, repeat)
    expected = torch.bincount(x[(x >= 0) & (x < bins)].long(), minlength=bins).to(torch.int32)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
