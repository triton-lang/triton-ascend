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
"""Native gate-on regression for ``StridedAxisCoalescing``.

This intentionally uses the original T2L eligibility shape instead of forcing
an option in the test: a one-dimensional block pointer has ``stride=S`` and a
base of ``ptr + (pid % S)``; ``pid // S`` selects the T tile.  The default NPU
compile mode is ``simd_simt_template`` (also called ``unstructured_in_simt``),
which supplies the historical template-SIMT half of the T2L gate on a real
910_95/A5 target.

Do not run this test on a generic/B4 machine.  Pretending that such a machine
is 910_95 would exercise a different compiler/toolchain contract and would not
validate this pass's original gate.
"""

import pytest
import torch
import triton
import triton.language as tl

try:
    from triton.backends.ascend.utils import is_compile_on_910_95
except Exception:
    # Keep collection safe in generic CI images where the Ascend device helper
    # is unavailable.  The actual test must never run without the real gate.
    def is_compile_on_910_95(_arch):
        return False


pytestmark = pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="StridedAxisCoalescing native validation requires an Ascend 910_95/A5 toolchain",
)


@triton.jit
def strided_axis_coalescing_copy(src, dst, T: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    # Keep both pieces of the canonical FLA split alive in TTIR.  The pass
    # folds the S head programs into one [BLOCK, S] tile and redirects `tile`
    # to the post-shrink program id.
    pid = tl.program_id(0)
    head = pid % S
    tile = pid // S

    src_block = tl.make_block_ptr(
        base=src + head,
        shape=(T, ),
        strides=(S, ),
        offsets=(tile * BLOCK, ),
        block_shape=(BLOCK, ),
        order=(0, ),
    )
    dst_block = tl.make_block_ptr(
        base=dst + head,
        shape=(T, ),
        strides=(S, ),
        offsets=(tile * BLOCK, ),
        block_shape=(BLOCK, ),
        order=(0, ),
    )
    value = tl.load(src_block)
    tl.store(dst_block, value)


def _launch_spec_from_compiled_metadata(metadata):
    from triton.backends.ascend import driver as ascend_driver
    return ascend_driver.make_launch_spec(metadata, ascend_driver.NPUUtils())


def test_strided_axis_coalescing_gate_on_e2e():
    """Compile and run the canonical S=4 axis fold through the real T2L gate."""
    s = 4
    block = 16
    t = 64
    grid_x = (t // block) * s
    assert grid_x % s == 0

    # Logical layout is [T, S], while each block pointer sees one strided
    # head column.  A successful rewrite has to launch only grid_x / S
    # programs and still copy every logical element exactly once.
    src = torch.arange(t * s, dtype=torch.float32, device="npu").reshape(t, s)
    dst = torch.empty_like(src)

    # Compile first through the normal JIT API, then retain its real metadata
    # for the ABI checks below.  Use the canonical public template-SIMT mode.
    kernel = strided_axis_coalescing_copy.warmup(
        src,
        dst,
        grid=(grid_x, ),
        T=t,
        S=s,
        BLOCK=block,
        compile_mode="simd_simt_template",
    )
    assert kernel is not None
    strided_axis_coalescing_copy[(grid_x, )](
        src,
        dst,
        T=t,
        S=s,
        BLOCK=block,
        compile_mode="simd_simt_template",
    )
    torch.npu.synchronize()

    assert torch.equal(dst.cpu(), src.cpu())

    # These fields are compiler output, not test-supplied options: together
    # they prove the compile_on_910_95 && compile_mode="simd_simt_template"
    # condition actually ran StridedAxisCoalescing and exported the launch contract.
    metadata = kernel.metadata
    assert metadata.compile_mode == "simd_simt_template"
    assert metadata.compile_on_910_95 is True
    assert metadata.coalesce_factor == s
    assert metadata.coalesce_axis == 0
    assert metadata.coalesce_grid_ceil_div is False

    from triton.backends.ascend.launcher import COALESCE_CEIL
    spec = _launch_spec_from_compiled_metadata(metadata)
    assert (spec.coalesce_factor, spec.coalesce_axis) == (s, 0)
    assert not spec.flags & COALESCE_CEIL
