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
Mechanism-level guard: ``getattr(obj, name, default)`` only falls back to
``default`` on ``AttributeError``.  In parallel compile mode the Ascend
autotuner reads ``getattr(res, "packed_metadata", None)``, where ``res`` is a
``FutureKernel`` (see test_autotune_parallel_masked_error_repro.py):

* 3.6: no ``__getattr__`` -> lookup raises AttributeError -> returns ``None``;
  compile errors stay cached in the future.
* 3.7: ``__getattr__`` resolves the future eagerly -> a failed compile raises
  a non-AttributeError (here a stand-in ``ValueError``) that propagates
  instead of being replaced by the default.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest

import triton.runtime._async_compile as _async_compile


def _make_failed_future_kernel():
    """A FutureKernel whose compile task failed (exception cached on the future)."""

    def compile_fn():
        raise ValueError("boom")  # stands in for MLIRCompilationError

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(compile_fn)
        return _async_compile.FutureKernel(None, future)


def test_future_kernel_getattr_behavior():
    """FutureKernel behavior: 3.6 returns default, 3.7 propagates the compile error."""
    future_kernel = _make_failed_future_kernel()

    if hasattr(type(future_kernel), "__getattr__"):
        # 3.7: eager resolution -> the worker's exception propagates
        with pytest.raises(ValueError, match="boom"):
            getattr(future_kernel, "packed_metadata", None)
    else:
        # 3.6: no __getattr__ -> default None, error stays cached in the future
        assert getattr(future_kernel, "packed_metadata", None) is None
