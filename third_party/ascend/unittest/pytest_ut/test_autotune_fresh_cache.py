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
"""Regression tests for issue #2818.

AutoTilingTuner must release caller-owned references and the active async
compile context on exceptional exits, so a kernel that fails to compile does
not leak device memory or poison every later compilation on the thread.
"""
import gc
import tempfile
import weakref

import pytest
import torch
import torch_npu  # noqa: F401  registers the ``npu`` device

import triton
import triton.language as tl

# Install the Ascend autotuner factory (also auto-loaded via the cann hook, but
# keep the explicit import so the test is robust to import-order changes).
import triton.backends.ascend.runtime  # noqa: F401
from triton.runtime._async_compile import active_mode

MIB = 1024 * 1024


@pytest.fixture
def fresh_triton_cache():
    """Isolate the on-disk cache, mirroring ``python/test/conftest.py``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with triton.knobs.cache.scope(), triton.knobs.runtime.scope():
            triton.knobs.cache.dir = tmpdir
            yield tmpdir


def _grid(meta):
    return (triton.cdiv(1024, meta["BLOCK"]), )


# Two distinct kernels on purpose: with a single kernel the first rejected
# compile populates the in-memory failure cache, so the second path never
# reaches parallel compilation and the context-poisoning symptom is masked.
@triton.jit
def _rejecting_kernel_a(x, y, N: tl.constexpr, FAIL: tl.constexpr, BLOCK: tl.constexpr):
    tl.static_assert(not FAIL, "deliberate compile rejection")
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets, offsets < N, 0) + 1, offsets < N)


@triton.jit
def _rejecting_kernel_b(x, y, N: tl.constexpr, FAIL: tl.constexpr, BLOCK: tl.constexpr):
    tl.static_assert(not FAIL, "deliberate compile rejection")
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets, offsets < N, 0) + 1, offsets < N)


def _make_tuner(kernel):
    return triton.autotune(
        configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
        key=["FAIL"],
    )(kernel)


def _make_inputs():
    # Two 16 MiB storages; the kernel touches only the first 1024 elements, so
    # retention is visible without a large launch.
    x = torch.empty(4 * MIB, device="npu", dtype=torch.float32)[:1024]
    y = torch.empty(4 * MIB, device="npu", dtype=torch.float32)[:1024]
    return x, y


def test_rejected_run_does_not_retain_inputs(fresh_triton_cache):
    """Part A: a failed compile must not leave the tuner holding the inputs."""
    tuner = _make_tuner(_rejecting_kernel_a)
    baseline = torch.npu.memory_allocated()
    x, y = _make_inputs()
    refs = [weakref.ref(x), weakref.ref(y)]
    with pytest.raises(Exception):
        tuner[_grid](x, y, 1024, True)
    del x, y
    gc.collect()
    torch.npu.synchronize()
    assert tuner.nargs is None
    assert all(r() is None for r in refs)
    assert torch.npu.memory_allocated() - baseline == 0


def test_failed_parallel_warmup_clears_active_mode(fresh_triton_cache):
    """Part B: a failed parallel warmup must not leave an exited
    AsyncCompileMode installed, or every later compile on this thread fails
    with 'Another AsyncCompileMode is already active'."""
    tuner = _make_tuner(_rejecting_kernel_b)
    tuner.compile_parallel = True
    x, y = _make_inputs()
    with pytest.raises(Exception):
        tuner.warmup(x, y, 1024, True, grid=_grid)
    assert active_mode.get() is None
    # Subsequent valid compile and launch on the same tuner must work.
    handles = tuner.warmup(x, y, 1024, False, grid=_grid)
    assert isinstance(handles, list) and len(handles) > 0
    tuner[_grid](x, y, 1024, False)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 1, rtol=0, atol=0)
