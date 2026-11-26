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
"""Device-level observation tests: does changing constexpr SCALE recompile ``scale``?

Run on an NPU with a patched triton-ascend build (the JIT reuse handoff must be
present, see ``patch/triton-ascend-3.6.0.patch``)::

    python -m pytest -q -s third_party/ascend/unittest/reuse_ut/test_scale_reuse.py

The test launches the same ``scale``-based kernel with different constexpr
``SCALE`` values and reports, per launch, how many distinct kernels were really
loaded:

* ``TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=0`` (disabled): every new ``SCALE`` is a
  new specialized variant, so it recompiles and reloads -> one load per SCALE.
* ``TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=1``: ``SCALE`` is dynamicized into a
  runtime argument -> all SCALE values share one kernel -> a single load.
* ``TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=2``: same single-load contract plus the
  optional tile reuse / background compilation machinery.

Kernel loads are counted with ``triton.knobs.runtime.kernel_load_start_hook``
(fires exactly once per loaded ``CompiledKernel``). Unlike ``jit_cache_hook``,
this hook does NOT disable the reuse path. With ``-s`` you also see the
``[ascend-reuse]`` diagnostics on stderr (analysis decisions and dispatch
outcomes) when reuse is enabled.

The parametrized modes share one pytest process, so the fixture resets the
global ``ReuseHandler`` per run: its registry and the private owner's device
cache would otherwise leak an already-loaded variant into the next mode (an
"exact ready" hit that loads nothing).

The kernel, launch helper and the ``reuse_mode`` fixture are defined here; the
fixture itself is shared via ``reuse_common.py``.
"""

import torch
import torch_npu

import triton
import triton.language as tl

from reuse_common import reuse_mode


@triton.jit
def scale(x, SCALE: tl.constexpr):
    return x * SCALE


@triton.jit
def scale_kernel(out_ptr, x_ptr, n, SCALE: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * TILE + tl.arange(0, TILE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = scale(x, SCALE)
    tl.store(out_ptr + offsets, y, mask=mask)


def launch_scale(scale_value, x, out, tile=64):
    n = x.numel()
    kernel = scale_kernel[(triton.cdiv(n, tile), )](out, x, n, SCALE=scale_value, TILE=tile)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x * scale_value)
    return kernel


def test_scale_constexpr_changes_and_recompilation(reuse_mode):
    mode, loads = reuse_mode
    n, tile = 1024, 64
    x = torch.randn(n, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)

    scales = [1, 2, 3, 4]
    kernels = []
    for scale_value in scales:
        kernel = launch_scale(scale_value, x, out, tile)
        kernels.append(kernel)
        print(f"[reuse_mode={mode}] SCALE={scale_value} -> kernel=0x{id(kernel):x} "
              f"loads_so_far={len(loads)}", flush=True)

    distinct = len({id(k) for k in kernels})
    print(f"[reuse_mode={mode}] summary: distinct_kernels={distinct} kernel_loads={len(loads)}", flush=True)

    if mode == 0:
        assert len(loads) == len(scales), (
            f"mode 0 must recompile for every SCALE (got {len(loads)} loads, expected {len(scales)})")
        assert distinct == len(scales)
    else:
        assert len(loads) == 1, (
            f"mode {mode} must reuse a single kernel for every SCALE (got {len(loads)} loads, expected 1). "
            "If the runtime handoff patch is missing, reuse never activates; check that "
            "AscendBackend.get_jit_reuse_handler is reachable in the installed triton.")
        assert distinct == 1
