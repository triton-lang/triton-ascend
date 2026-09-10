import json
import os

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95


pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_RUN_SIMD_SIMT_COSTMODEL_GUARDS") != "1",
    reason="SIMD/SIMT costmodel guards are opt-in",
)

simd_simt_910_95_only = pytest.mark.xfail(
    not is_compile_on_910_95(),
    reason="SIMD/SIMT cost model only supports 910_95",
    run=False,
)


@triton.jit
def cumsum_backend_route_kernel(
    input_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    result = tl.cumsum(values, axis=0)
    tl.store(output_ptr + offsets, result)


@simd_simt_910_95_only
def test_cumsum_keeps_supported_routes_legal(tmp_path):
    block = 256
    torch.manual_seed(7)
    values = torch.randn((block,), dtype=torch.float32, device="npu")
    output = torch.empty_like(values)
    report_path = tmp_path / "cumsum_route.json"

    cumsum_backend_route_kernel[(1,)](
        values,
        output,
        BLOCK=block,
        num_warps=4,
        compile_mode="simd_simt",
        auto_simt_scope_mode="auto",
        auto_simt_scope_dump=str(report_path),
        logical_program_count_hint=1,
        physical_vector_core_count_hint=1,
    )

    torch.testing.assert_close(output, torch.cumsum(values, dim=0),
                               rtol=1e-4, atol=1e-4)
    report = json.loads(report_path.read_text())
    assert report["stage_model"]["applied"]

    stages = report["stage_model"]["logical_stages"]
    scan_stages = [
        stage for stage in stages if stage["features"]["has_prefix_scan"]
    ]
    assert len(scan_stages) == 1
    assert scan_stages[0]["model"] == "prefix_scan"
    assert not scan_stages[0]["local_simt_materializable"]

    # Route choice is profile-dependent. Plain cumsum remains legal for both
    # whole-kernel backend routes. A local mixed route requires the separate
    # anchor-free materialization feature and is intentionally unavailable in
    # this standalone scan change.
    routes = report["stage_model"]["routes"]
    assert routes["all_simd"]["legal"]
    assert routes["all_simt_only"]["legal"]
    assert not routes["mixed_simd_simt"]["legal"]
