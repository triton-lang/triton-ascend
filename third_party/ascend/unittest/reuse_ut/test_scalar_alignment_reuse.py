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
"""Does a runtime scalar divisible by 16 specialize and force a recompile?

Mechanism (``python/src/specialize.cc``, ``handle_long_type``): an i32 scalar
specializes to ``("i32", "D")`` when ``(value & 15) == 0`` and alignment
specialization is enabled, otherwise to ``("i32", "")``. The flag ``align`` in
the binder comes from ``backend.use_alignment_specialization(options)``, which
the Ascend backend enables only for ``compile_mode="simt_only"`` (default
``"simd"`` keeps it off, see ``third_party/ascend/backend/compiler.py``).

So on Ascend (simd) a multiple of 16 and a non-multiple share the key ``""``:
crossing the 16-byte boundary does NOT recompile. The test reports the real
specialization key observed on each launch (``kernel.src.signature``), the
kernel-load count per reuse mode, and what the keys would be if alignment
specialization were turned on (``simt_only``).

Unlike ``SCALE`` in ``test_scale_reuse.py``, the scalar under test is a runtime
argument, not a constexpr, so the constexpr-reuse feature (mode 1/2) does not
dynamicize it: all three modes must observe the same load count.
"""

import torch
import torch_npu

import triton
import triton.language as tl

from triton._C.libtriton import native_specialize_impl
from triton.backends.compiler import BaseBackend

from reuse_common import reuse_mode


@triton.jit
def add_value(out_ptr, x_ptr, n, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x + value, mask=mask)


def launch_add(value, x, out, tile=64):
    """Launch ``add_value`` with one runtime scalar and check the result."""
    n = x.numel()
    kernel = add_value[(triton.cdiv(n, tile), )](out, x, n, value, BLOCK=tile)
    torch.npu.synchronize()
    expected = x + torch.tensor(value, dtype=x.dtype, device=x.device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, check_dtype=True)
    return kernel


def _specialize_key(value, align):
    """The cache-key fragment native_specialize_impl computes for a scalar int."""
    return native_specialize_impl(BaseBackend, value, False, True, align)[1]


def test_scalar_divisible_by_16_specialization(reuse_mode):
    mode, loads = reuse_mode
    values = [16, 17, 32, 33, 48, 64]  # cross the 16-byte boundary repeatedly
    n, tile = 1024, 64
    x = torch.arange(n, dtype=torch.int32, device="npu")
    out = torch.empty_like(x)

    kernels = []
    for value in values:
        kernel = launch_add(value, x, out, tile)
        kernels.append(kernel)
        key = kernel.src.signature["value"][1]  # observed specialization key
        print(
            f"[reuse_mode={mode}] value={value:3d} spec_key={key!r} "
            f"kernel=0x{id(kernel):x} loads_so_far={len(loads)}", flush=True)

    distinct = len({id(k) for k in kernels})
    print(f"[reuse_mode={mode}] summary: distinct_kernels={distinct} kernel_loads={len(loads)}", flush=True)

    keys_aligned = sorted({_specialize_key(v, True) for v in values})
    keys_ascend = sorted({_specialize_key(v, False) for v in values})
    print(f"[reuse_mode={mode}] specialize_impl keys: align=True -> {keys_aligned}; "
          f"align=False -> {keys_ascend}", flush=True)
    assert keys_aligned == ["", "D"], "16|32|48|64 must key 'D' and 17|33 '' with alignment specialization on"
    assert keys_ascend == [""], "Ascend simd disables alignment specialization; all values must key ''"

    # Ascend default simd: alignment specialization is off, so every value gets
    # the same key and a single kernel is loaded - 16-divisibility does NOT
    # recompile. With simt_only the boundary would produce two loads.
    assert len(loads) == 1, (f"values crossing the 16-byte boundary must reuse one kernel under simd "
                             f"(got {len(loads)} loads, expected 1); simt_only would split into "
                             f"{len(keys_aligned)} specializations -> {len(keys_aligned)} loads")
    assert distinct == 1
