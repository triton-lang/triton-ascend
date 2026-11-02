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

from triton.runtime.jit import JITFunction


def _module_kernels(module):
    """The ``@triton.jit`` kernels defined at the top level of ``module``."""
    return [obj for obj in vars(module).values() if isinstance(obj, JITFunction)]


@pytest.fixture(params=[0, 1, 2])
def reuse_mode(request, monkeypatch):
    """Force one TRITON_ASCEND_ENABLE_DYNAMIC_REUSE mode and count kernel loads.

    ``request.param`` is the reuse mode. Yields ``(mode, loads)`` where ``loads``
    counts how many distinct CompiledKernels were really loaded (observed through
    ``triton.knobs.runtime.kernel_load_start_hook``).
    """
    from triton import knobs
    from triton.backends.ascend.reuse import bridge, config

    mode = request.param
    monkeypatch.setenv("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", str(mode))
    config.mode.cache_clear()  # mode() is read once at startup; force a re-read.

    # The global ReuseHandler (registry + analysis cache + owner device caches)
    # survives across parametrized runs in this process. A prior mode's already
    # loaded variant would otherwise be handed to this run as an "exact ready"
    # hit, so no load hook would ever fire. Reset it so each mode starts clean.
    monkeypatch.setattr(bridge, "_handler", None)

    loads = []

    def on_load(*args, **_kwargs):
        loads.append(args[4] if len(args) > 4 else None)  # CompiledKernel.hash

    # Observational only; the reuse path explicitly permits load/launch hooks.
    knobs.runtime.kernel_load_start_hook.add(on_load)
    if mode:
        # Gate the [ascend-reuse] stderr diagnostics (analysis + dispatch).
        monkeypatch.setattr(knobs.runtime, "debug", True)

    # Clean slate for the native JIT path: each kernel keeps its per-device
    # cache across parametrized runs in the same process. Inspect the current
    # test module so every test file keeps (and cleans) its own kernels.
    kernels = _module_kernels(request.module)
    for fn in kernels:
        fn.device_caches.clear()
    try:
        yield mode, loads
    finally:
        knobs.runtime.kernel_load_start_hook.remove(on_load)
        for fn in kernels:
            fn.device_caches.clear()
