import json
import os

import pytest
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver
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


def _vector_core_count():
    properties = driver.active.utils.get_device_properties(
        torch.npu.current_device())
    return int(properties["num_vectorcore"])


def _launch_options(report_path, logical_programs):
    return {
        "num_warps": 4,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": _vector_core_count(),
    }


@triton.jit
def anchorless_mixed_stage_scope_kernel(
    x_ptr,
    y_ptr,
    seed_ptr,
    out_ptr,
    scalar_out_ptr,
    BLOCK: tl.constexpr,
    SCALAR_ITERATIONS: tl.constexpr,
    RECURRENCE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x * y + 1.0)

    # This vector recurrence has no primitive SIMT anchor. Its tensor result
    # also matches the backend's SIMT VF return ABI.
    recurrence_offs = pid * RECURRENCE_BLOCK + tl.arange(0, RECURRENCE_BLOCK)
    state = tl.load(seed_ptr + recurrence_offs)
    for _ in range(SCALAR_ITERATIONS):
        state = state * 1.0000001 + 1e-7
    tl.store(scalar_out_ptr + recurrence_offs, state)


@simd_simt_910_95_only
def test_costmodel_anchorless_mixed_stage_scope(tmp_path):
    logical_programs = _vector_core_count()
    block = 8192
    iterations = 4096
    recurrence_block = 32
    torch.manual_seed(3)
    x = torch.randn((logical_programs * block,), dtype=torch.float32,
                    device="npu")
    y = torch.randn_like(x)
    seed = torch.rand((logical_programs * recurrence_block,),
                      dtype=torch.float32, device="npu")
    output = torch.empty_like(x)
    scalar_output = torch.empty_like(seed)
    report_path = tmp_path / "anchorless_mixed_route.json"

    anchorless_mixed_stage_scope_kernel[(logical_programs,)](
        x, y, seed, output, scalar_output, BLOCK=block,
        SCALAR_ITERATIONS=iterations,
        RECURRENCE_BLOCK=recurrence_block,
        **_launch_options(report_path, logical_programs))

    torch.testing.assert_close(output, x * y + 1.0, rtol=1e-5, atol=1e-5)
    reference = seed.clone()
    for _ in range(iterations):
        reference = reference * 1.0000001 + 1e-7
    torch.testing.assert_close(scalar_output, reference, rtol=1e-3,
                               atol=1e-3)

    report = json.loads(report_path.read_text())
    assert report["stage_model"]["applied"]
    assert report["features"]["simt_anchors"]["count"] == 0

    stages = report["stage_model"]["logical_stages"]
    mixed = report["stage_model"]["routes"]["mixed_simd_simt"]
    assert mixed["legal"]
    simt_indices = [
        index for index, stage in enumerate(mixed["stages"])
        if stage["implementation"]["mode"] == "simt"
    ]
    assert simt_indices
    assert any(stages[index]["model"] == "loop_carried_recurrence"
               for index in simt_indices)
    for index in simt_indices:
        assert stages[index]["simt_anchor_indices"] == []
        assert stages[index]["local_simt_materializable"]
        assert mixed["stages"][index]["implementation"]["materialization"] == (
            "local_simt_scope_with_kernel_v1")

    # Selection is profile-dependent. If mixed wins, the modeled stage-owned
    # scope must also be materialized in the compiled module.
    if report["effective_decision_kind"] == "mixed_simd_simt":
        assert report["materialized_stage_owned_scope_count"] > 0
    else:
        assert report["materialized_stage_owned_scope_count"] == 0
