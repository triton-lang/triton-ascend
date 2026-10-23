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
"""
Regression test: an autotune kernel with a config that fails to compile.

Runs on both triton-ascend-3.6 and 3.7.  With parallel compilation (the
default), on 3.7 ``AutoTilingTuner._batch_bench`` used to raise the compile
error of the first bad config while *submitting* the configs
(``FutureKernel.__getattr__`` eagerly resolving the future on the
``packed_metadata`` read inside ``_make_kernel_call``), where it escaped the
per-config try/except and was swallowed by the catch-all handler -- aborting
the whole autotune with ``RuntimeError: No valid triton configs. NoneType:
None``.

After the fix the compile error surfaces at ``fut.result()`` inside the
per-config try/except: the bad config is silently dropped and autotune
completes with the remaining config.  This test asserts that end to end:
no exception, and the output matches the reference computed with the
surviving config.  On 3.6 the per-config handler already drops the failing
config, so the test passes there as well.

config #1 sets ``STRIDE0=0``: the leading axis of ``tl.make_block_ptr``
gets stride 0 and is boundary-checked.  TritonToLinalg's
``getBoundarySizes()`` (third_party/ascend/lib/Utils/Utils.cpp) then
divides the flat block offset by the zero stride, failing the pass
(MLIRCompilationError).  config #2 uses the row-major stride and
compiles fine.
"""
import torch
import torch_npu

import triton
import triton.language as tl
import triton.backends.ascend.runtime  # noqa: F401


@triton.autotune(
    configs=[
        triton.Config({'STRIDE0': 0}, num_warps=4),  # bad: zero-stride checked axis
        triton.Config({'STRIDE0': 64}, num_warps=4),  # good: row-major stride
    ],
    key=[],
)
@triton.jit
def boundary_kernel(in_ptr, out_ptr, M, N, STRIDE0: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    bptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(M, N),
        strides=(STRIDE0, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    val = tl.load(bptr, boundary_check=(0, 1), padding_option="zero")

    out = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out, val, boundary_check=(0, 1))


def test_autotune_drops_failing_config():
    M, N = 8, 64
    BLOCK_M, BLOCK_N = 4, 32

    x = torch.arange(M * N, dtype=torch.float32, device='npu').reshape(M, N)
    out = torch.zeros((M, N), dtype=torch.float32, device='npu')

    # Before the fix this call raises
    # RuntimeError("No valid triton configs. NoneType: None"); see docstring.
    boundary_kernel[(1, )](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()

    # The surviving config (STRIDE0=64) copies the block unchanged.
    ref = torch.zeros_like(out)
    ref[:BLOCK_M, :BLOCK_N] = x[:BLOCK_M, :BLOCK_N]
    torch.testing.assert_close(out, ref)
