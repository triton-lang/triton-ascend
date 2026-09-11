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
def loop_body_split_stage_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
    LOOP_COUNT: tl.constexpr,
):
    pid = tl.program_id(0)
    region = pid * (BLOCK * (LOOP_COUNT + 1))
    offs = region + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, value * 2.0 + 1.0)

    # The induction variable is used only for addressing, so the loop has no
    # algorithmic loop-carried dependency and its body can be staged alone.
    base = region + BLOCK
    for i in range(LOOP_COUNT):
        loop_offs = base + i * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x_ptr + loop_offs)
        tl.store(out_ptr + loop_offs, value * 3.0 + 0.5)


@simd_simt_910_95_only
def test_costmodel_loop_body_stage_division(tmp_path):
    logical_programs = _vector_core_count()
    block = 2048
    loop_count = 8
    torch.manual_seed(4)
    x = torch.randn((logical_programs * block * (loop_count + 1),),
                    dtype=torch.float32, device="npu")
    output = torch.empty_like(x)
    report_path = tmp_path / "loop_body_stage_division.json"

    loop_body_split_stage_kernel[(logical_programs,)](
        x, output, BLOCK=block, LOOP_COUNT=loop_count,
        **_launch_options(report_path, logical_programs))

    x_blocks = x.view(logical_programs, loop_count + 1, block)
    out_blocks = output.view(logical_programs, loop_count + 1, block)
    torch.testing.assert_close(out_blocks[:, 0, :],
                               x_blocks[:, 0, :] * 2.0 + 1.0,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_blocks[:, 1:, :],
                               x_blocks[:, 1:, :] * 3.0 + 0.5,
                               rtol=1e-6, atol=1e-6)

    report = json.loads(report_path.read_text())
    assert report["stage_model"]["applied"]
    stages = report["stage_model"]["logical_stages"]
    shells = [stage for stage in stages
              if stage["model"] == "independent_pipelined_loop"]
    assert len(shells) == 1
    shell = shells[0]
    assert shell["iteration_count"] == loop_count
    assert shell["workload"]["load_bytes_per_iteration"] == 0
    assert shell["workload"]["store_bytes_per_iteration"] == 0
    assert shell["features"]["loop_backedge_count"] >= 1

    bodies = [
        stage for stage in stages
        if stage["iteration_count"] == loop_count
        and stage["workload"]["store_bytes_per_iteration"] > 0
    ]
    assert bodies
    assert all(stage["model"] != "independent_pipelined_loop"
               for stage in bodies)

    total_store_bytes = sum(
        stage["iteration_count"]
        * stage["workload"]["store_bytes_per_iteration"]
        for stage in stages)
    assert total_store_bytes == (loop_count + 1) * block * 4
