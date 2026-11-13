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
"""Does a constexpr TILE change recompile when tile reuse is enabled?

Run on an NPU with a patched triton-ascend build::

    python -m pytest -q -s third_party/ascend/unittest/reuse_ut/test_tile_reuse.py

The kernel is a plain copy indexed by ``pid * TILE + arange(TILE)`` (the same
shape as ``copy_kernel`` in ``test_constexpr_reuse.py``). ``N`` is a constexpr
that mode 2 dynamicizes into a runtime argument; ``TILE`` is proven as a
*schedule* parameter (``ParameterClass.ScheduleReusable``) and stays constexpr
in the compiled variant.

Per mode:

* ``TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=0`` (disabled): every ``TILE`` value is
  its own specialized variant -> one load per TILE.
* ``TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=1``: ``TILE`` stays a compile-time
  specialization (tile reuse is not proven without mode 2) -> one load per
  TILE again.
* ``TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=2``: tile reuse. The first ``TILE``
  compiles and loads one kernel; a later ``TILE`` is *scheduled* onto it
  (``adapt_tile``/``adapt_grid`` re-derive the runtime TILE override and the
  grid that preserves full coverage) -> a single load, no recompile.

Kernel loads are counted with ``triton.knobs.runtime.kernel_load_start_hook``
as in the sibling tests. With ``-s`` the ``[ascend-reuse]`` diagnostics show
the ``schedule-reused=[TILE: ...]`` dispatch.
"""

import torch
import torch_npu

import triton
import triton.language as tl

from reuse_common import reuse_mode


@triton.jit
def copy_kernel(src, dst, N: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.load(src + index, mask=index < N, other=0)
    tl.store(dst + index, value, mask=index < N)


def launch_copy(n, tile, src, dst):
    """Launch ``copy_kernel`` with one TILE and check the copy result."""
    # The grid contract is cdiv(N, TILE); dispatch recomputes it when it
    # schedules the launch onto a kernel compiled with a different TILE.
    grid = lambda meta: (triton.cdiv(meta["N"], meta["TILE"]), )
    kernel = copy_kernel[grid](src, dst, N=n, TILE=tile)
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src)
    return kernel


def test_tile_change_reuses_loaded_kernel(reuse_mode):
    mode, loads = reuse_mode
    n = 1000
    src = torch.arange(n, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)

    tiles = [64, 32, 128]  # cross the TILE boundary; later tiles must reuse the first kernel
    kernels = []
    for tile in tiles:
        kernel = launch_copy(n, tile, src, dst)
        kernels.append(kernel)
        print(f"[reuse_mode={mode}] TILE={tile} -> kernel=0x{id(kernel):x} "
              f"loads_so_far={len(loads)}", flush=True)

    distinct = len({id(k) for k in kernels})
    print(f"[reuse_mode={mode}] summary: distinct_kernels={distinct} kernel_loads={len(loads)}", flush=True)

    if mode == 2:
        # Tile reuse is proven only in mode 2: TILE is a schedule parameter, so
        # every later TILE is dispatched onto the already-loaded kernel and the
        # grid is re-derived to preserve coverage. No new load fires.
        assert len(loads) == 1, (
            f"mode 2 must schedule every TILE onto one loaded kernel (got {len(loads)} loads, expected 1). "
            "If the runtime handoff patch or the compile worker is missing, tile reuse never activates.")
        assert distinct == 1
    else:
        # Modes 0/1 keep TILE a compile-time specialization: each TILE value
        # produces its own specialized variant and recompiles.
        assert len(loads) == len(tiles), (
            f"mode {mode} must recompile for every TILE (got {len(loads)} loads, expected {len(tiles)})")
        assert distinct == len(tiles)
