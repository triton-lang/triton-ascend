# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Real batching acceptance on original kernels; positive tests forbid fallback."""
import pytest
import torch
import triton
import triton.language as tl
from block_batching_test_utils import run_case


@pytest.mark.parametrize("kind,axis", [("rmsnorm", 0), ("situ_fwd", 0), ("situ_bwd", 0)])
@pytest.mark.parametrize("factor", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_original_kernel_batching(kind, axis, factor, dtype):
    result, _ = run_case(kind, 10, 129, dtype, factor, axis, require_applied=factor > 1)
    assert result["correctness"]
    assert result["execution"]["runtime_status"] == ("APPLIED" if factor > 1 else "BASELINE_SELECTED")


def test_request_changes_jit_cache_identity():
    _, first = run_case("situ_fwd", 10, 129, torch.float32, 3, 0, require_applied=True)
    _, second = run_case("situ_fwd", 10, 129, torch.float32, 6, 0, require_applied=True)
    _, again = run_case("situ_fwd", 10, 129, torch.float32, 3, 0, require_applied=True)
    assert first.hash != second.hash
    assert first is again


@triton.jit
def _original_grid_kernel(out):
    x, y, z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gx, gy, gz = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    offset = x + gx * (y + gy * z)
    tl.store(out + offset, gx + 100 * gy + 10000 * gz)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("graph", [True, False])
def test_dynamic_grid_and_callable_are_preserved(axis, graph):
    calls = 0
    for extent in (1, 2, 3, 4, 7):
        grid = [2, 2, 2]
        grid[axis] = extent
        out = torch.full((grid[0] * grid[1] * grid[2], ), -1, device="npu", dtype=torch.int32)

        def get_grid(meta):
            nonlocal calls
            calls += 1
            return tuple(grid)

        before = calls
        compiled = _original_grid_kernel[get_grid](out, block_batching_factor=3, block_batching_axis=axis,
                                                   enable_graph_optimize=graph)
        torch.npu.synchronize()
        assert calls == before + 1
        torch.testing.assert_close(out, torch.full_like(out, grid[0] + 100 * grid[1] + 10000 * grid[2]))
        assert compiled.last_launch["runtime_status"] == "APPLIED", compiled.block_batching


def test_runtime_write_alias_selects_complete_baseline():
    from block_batching_kernels import _fused_situgluv2_fwd_triton_kernel
    a = torch.randn((10, 129), device="npu")
    b = torch.randn_like(a)
    beta = .5

    def launch(out):
        return _fused_situgluv2_fwd_triton_kernel[(10, 2)](a, b, out, 129, 129, 1, 129, 1, 129, 1, beta, 2., beta, 2.,
                                                           BLOCK=128, block_batching_factor=3, block_batching_axis=0)

    separate = torch.empty_like(a)
    kernel = launch(separate)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.block_batching
    expected = beta * torch.tanh(a / beta) * torch.sigmoid(a) * beta * torch.tanh(b / beta)
    launch(a)
    torch.npu.synchronize()
    assert kernel.last_launch["reason"] == "WRITE_ALIAS"
    assert kernel.last_launch["runtime_status"] == "BASELINE_SELECTED"
    torch.testing.assert_close(a, expected, rtol=2e-4, atol=2e-5)


def test_capture_replay_selects_baseline():
    from block_batching_test_utils import prepare_case
    case = prepare_case("situ_fwd", 10, 129, torch.float32, 3, 0)
    kernel = case["launch"]()
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.block_batching
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        case["launch"]()
    assert kernel.last_launch["reason"] == "CAPTURE_BASELINE"
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(case["outputs"][0], case["expected"][0], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_empty_grid_never_launches_candidate(axis):
    out = torch.full((8, ), -1, device="npu", dtype=torch.int32)
    kernel = _original_grid_kernel[(2, 2, 2)](out, block_batching_factor=3, block_batching_axis=axis)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.block_batching
    out.fill_(-1)
    empty = [2, 2, 2]
    empty[axis] = 0
    _original_grid_kernel[tuple(empty)](out, block_batching_factor=3, block_batching_axis=axis)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "EMPTY_GRID"
    torch.testing.assert_close(out, torch.full_like(out, -1))


def test_column_tail_with_standard_backend_templates():
    result, kernel = run_case("situ_bwd", 10, 129, torch.float32, 3, 1)
    assert result["correctness"]
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.block_batching
    assert kernel.block_batching["validation_policy"] == "ta_only_v1"
    assert kernel.block_batching["internal_simt"] == "not_inspected"
