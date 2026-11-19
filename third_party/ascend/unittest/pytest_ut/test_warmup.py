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
import tempfile

import pytest
import torch

import triton
import triton.language as tl

# ``triton.autotune`` is resolved when a test builds its kernel, so this
# side-effect import must run first to install the Ascend factory.
import triton.backends.ascend.runtime  # noqa: F401
from triton.compiler.compiler import CompiledKernel


@pytest.fixture
def fresh_triton_cache():
    """Isolate the on-disk cache, mirroring ``python/test/conftest.py``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with triton.knobs.cache.scope(), triton.knobs.runtime.scope():
            triton.knobs.cache.dir = tmpdir
            yield tmpdir


def _make_autotuned_add():

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128}, num_warps=1),
            triton.Config({"BLOCK": 256}, num_warps=1),
        ],
        key=["N"],
    )
    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        tl.store(o + idx, tl.load(a + idx, mask=mask) + tl.load(b + idx, mask=mask), mask=mask)

    return kernel_add


def test_autotune_warmup_with_tensors(fresh_triton_cache) -> None:
    """Autotuned warmup returns one CompiledKernel per config and stays correct."""
    kernel_add = _make_autotuned_add()
    n = 1024
    a = torch.randn(n, dtype=torch.float32, device="npu")
    b = torch.randn(n, dtype=torch.float32, device="npu")
    sentinel = -12345.0
    out = torch.full((n, ), sentinel, dtype=torch.float32, device="npu")

    handles = kernel_add.warmup(a, b, out, N=n, grid=(1, ))
    torch.npu.synchronize()
    assert isinstance(handles, list)
    assert len(handles) == 2
    # The Ascend autotuner compiles configs in parallel by default, so warmup
    # returns FutureKernel handles; resolving them must yield CompiledKernels.
    kernels = [handle.result() if hasattr(handle, "result") else handle for handle in handles]
    assert all(isinstance(k, CompiledKernel) for k in kernels)
    assert torch.all(out == sentinel), "autotuned warmup executed the kernel"

    kernel_add[lambda meta: (triton.cdiv(n, meta["BLOCK"]), )](a, b, out, N=n)
    torch.npu.synchronize()
    assert torch.allclose(out, a + b)


def test_autotune_warmup_dtype_only(fresh_triton_cache) -> None:
    """Autotuned warmup accepts torch.dtype placeholders, matching upstream."""
    kernel_add = _make_autotuned_add()
    handles = kernel_add.warmup(torch.float32, torch.float32, torch.float32, N=1024, grid=(1, ))
    assert isinstance(handles, list)
    assert len(handles) == 2
    kernels = [handle.result() if hasattr(handle, "result") else handle for handle in handles]
    assert all(isinstance(k, CompiledKernel) for k in kernels)


def _make_autotuned_add_auto_configs():

    @triton.autotune(configs=[], key=["N"])
    @triton.jit
    def kernel_add_auto(a, b, o, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        tl.store(o + idx, tl.load(a + idx, mask=mask) + tl.load(b + idx, mask=mask), mask=mask)

    return kernel_add_auto


def test_autotune_warmup_dtype_only_auto_configs(fresh_triton_cache) -> None:
    """dtype-only warmup also drives Ascend's automatic tile generation."""
    kernel_add = _make_autotuned_add_auto_configs()
    handles = kernel_add.warmup(torch.float32, torch.float32, torch.float32, N=1024, grid=(1, ))
    assert isinstance(handles, list)
    assert len(handles) > 0
    kernels = [handle.result() if hasattr(handle, "result") else handle for handle in handles]
    assert all(isinstance(k, CompiledKernel) for k in kernels)
