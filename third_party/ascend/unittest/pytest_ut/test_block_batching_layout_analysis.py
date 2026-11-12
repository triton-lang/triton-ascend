# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse, pointwise
from block_batching_memory_cases import memory_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def analyze(tmp_path, text, factor=3, axis=0):
    module = parse(tmp_path, text)
    before = str(module)
    result = ascend.passes.ttir.analyze_block_batching(module, factor, axis)
    assert str(module) == before
    assert module.verify()
    assert result["compile_status"] == "ANALYZED", result
    layout = json.loads(result["layout_json"])
    assert layout["planner_version"] == 2
    assert not any(row["rewritten"] for row in layout["accesses"])
    return layout["accesses"]


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("factor", [2, 3, 6])
def test_linear_direction(tmp_path, axis, factor):
    rows = analyze(tmp_path, pointwise(axis), factor, axis)
    assert len(rows) == 2
    for row in rows:
        assert row["action"] == "MemoryView"
        assert row["selected_grid_axis"] == axis
        assert row["local_axes"] == [0]
        assert row["tile_extent"] == 4
        assert row["origin"] == 0


def test_nonzero_logical_origin(tmp_path):
    text = pointwise(0).replace("start = 0 : i32, end = 4 : i32", "start = 4 : i32, end = 8 : i32")
    assert all(row["origin"] == 4 for row in analyze(tmp_path, text))


def test_per_use_projection_of_reused_range(tmp_path):
    text = memory_ir("nd_reused_range").replace(
        "%row_offset = arith.muli %rows, %kv : tensor<4x4xi32>", "%stride = arith.constant dense<8> : tensor<4x4xi32>\n"
        "    %row_offset = arith.muli %rows, %stride : tensor<4x4xi32>")
    rows = analyze(tmp_path, text)
    assert all(row["local_axes"] == [0] and row["action"] == "MemoryView" for row in rows)
    # Both axes come from the SAME make_range. Transposing each use changes
    # its projected direction; matching range SSA identity would be wrong.
    text = text.replace("%xa = tt.addptr", "%raw_xa = tt.addptr").replace("%ya = tt.addptr", "%raw_ya = tt.addptr")
    text = text.replace(
        "    %zero =",
        """    %xa = tt.trans %raw_xa {order = array<i32: 1, 0>} : tensor<4x4x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %ya = tt.trans %raw_ya {order = array<i32: 1, 0>} : tensor<4x4x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %tm = tt.trans %mask {order = array<i32: 1, 0>} : tensor<4x4xi1> -> tensor<4x4xi1>
    %zero =""")
    text = text.replace("tt.load %xa, %mask", "tt.load %xa, %tm").replace("%result, %mask", "%result, %tm")
    rows = analyze(tmp_path, text)
    assert all(row["local_axes"] == [1] and row["action"] == "PointwiseRegion" for row in rows)


def test_dynamic_stride_is_proven_symbolically(tmp_path):
    rows = analyze(tmp_path, memory_ir("nd_reused_range"))
    assert all(row["action"] == "MemoryView" and row["reason"] == "PROVEN_SYMBOLIC_STRIDE_CHAIN" for row in rows)


def test_rectangular_row_and_shared_column_keep(tmp_path):
    text = pointwise(0).replace("%mask = arith.cmpi slt, %idx, %nv", "%mask = arith.cmpi slt, %range, %nv")
    rows = analyze(tmp_path, text)
    assert all(row["action"] == "Keep" and row["reason"] == "ALREADY_STRUCTURED" for row in rows)
    text = text.replace("tt.addptr %xp, %idx", "tt.addptr %xp, %range")
    rows = analyze(tmp_path, text)
    assert rows[0]["reason"] == "SHARED_ACCESS"
    assert rows[1]["reason"] == "ALREADY_STRUCTURED"


def test_holes_cannot_become_a_prefix(tmp_path):
    text = pointwise(0).replace(
        "%xp =", """%local_mask = arith.cmpi slt, %range, %nv : tensor<4xi32>
    %holes = arith.andi %mask, %local_mask : tensor<4xi1>
    %xp =""").replace("tt.load %xa, %mask", "tt.load %xa, %holes").replace("%result, %mask", "%result, %holes")
    rows = analyze(tmp_path, text)
    assert all(row["reason"] == "MASK_NOT_INTERVAL_OR_RECTANGLE" for row in rows)
    assert all(row["action"] == "Keep" for row in rows)


def test_address_stride_mismatch(tmp_path):
    text = pointwise(0).replace("%xp =", """%twice = arith.addi %range, %idx : tensor<4xi32>
    %xp =""").replace("tt.addptr %xp, %idx", "tt.addptr %xp, %twice")
    rows = analyze(tmp_path, text)
    assert rows[0]["reason"] == "STRIDE_CHAIN_UNPROVEN"
    assert rows[1]["action"] == "MemoryView"


def test_original_address_division_never_authorizes_overlapping_writes(tmp_path):
    text = pointwise(0).replace("%base = arith.muli %p, %b", "%base = arith.divsi %p, %b")
    module = parse(tmp_path, text)
    before = str(module)
    result = ascend.passes.ttir.analyze_block_batching(module, 3, 0)
    assert result["compile_status"] == "ANALYZED"
    assert str(module) == before
    result = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert result['compile_status'] == 'REWRITTEN'
    from triton.backends.ascend.block_batching_guard import check_guard
    guard = dict(json.loads(result['guard_json']), axis=0)
    assert check_guard(guard, [1 << 32, 1 << 45, 64], (8, 1, 1), 3) == (False, 'OVERLAPPING_BLOCK_WRITES')


def test_general_address_division_stays_rejected(tmp_path):
    text = pointwise(0).replace("%base = arith.muli %p, %b", "%base = arith.divsi %p, %n")
    module = parse(tmp_path, text)
    before = str(module)
    result = ascend.passes.ttir.analyze_block_batching(module, 3, 0)
    assert result['compile_status'] == 'REJECTED'
    assert result['reason'] == 'UNSUPPORTED_ADDRESS_EXPRESSION'
    assert str(module) == before


def test_factor_one_skips_planner(tmp_path):
    module = parse(tmp_path, pointwise(0))
    result = ascend.passes.ttir.analyze_block_batching(module, 1, 0)
    assert result["compile_status"] == "DISABLED"
    assert result["layout_json"] == ""
