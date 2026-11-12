# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Native normalization plus the unchanged runtime memory proof."""
import json

import pytest
from triton._C.libtriton import ascend

from test_block_batching_reductions import parse
from block_batching_reduction_cases import reduction_ir
from test_block_batching_guard_v2 import module as guard

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def fields_ir(columns=(0, 1, 2), masked=False):
    stores = []
    for i, column in enumerate(columns):
        stores.append(f"""
    %c{i} = arith.constant {column} : i32
    %ptr{i} = tt.addptr %base, %c{i} : !tt.ptr<f32>, i32
    tt.store {"%base" if column == 0 else f"%ptr{i}"}, %value{', %live' if masked else ''} : !tt.ptr<f32>
""")
    return """module {
  tt.func public @fields(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %stride: i32) {
    %pid = tt.get_program_id x : i32
    %p64 = arith.extsi %pid : i32 to i64
    %s64 = arith.extsi %stride : i32 to i64
    %row = arith.muli %p64, %s64 : i64
    %base = tt.addptr %out, %row : !tt.ptr<f32>, i64
    %in = tt.addptr %x, %p64 : !tt.ptr<f32>, i64
    %value = tt.load %in : !tt.ptr<f32>
    %live = arith.constant true
""" + "".join(stores) + """
    tt.return
  }
}
"""


def rewritten_guard(tmp_path, text, factor=3):
    ir = parse(tmp_path, text)
    result = ascend.passes.ttir.rewrite_block_batching(ir, factor, 0)
    assert result["compile_status"] == "REWRITTEN", result
    assert ir.verify()
    spec = json.loads(result["guard_json"])
    spec["axis"] = 0
    return ir, spec


@pytest.mark.parametrize("kind", ["sum", "max", "maxnum"])
@pytest.mark.parametrize("factor", [2, 3])
def test_identity_reduction_reshape_is_normalized(tmp_path, kind, factor):
    text = reduction_ir(kind).replace(
        '%reduced = "tt.reduce"(%v)', '%flat = tt.reshape %v allow_reorder : tensor<4xf32> -> tensor<4xf32>\n'
        '    %reduced = "tt.reduce"(%flat)')
    assert "%flat" in text
    ir, _ = rewritten_guard(tmp_path, text, factor)
    assert "allow_reorder" not in str(ir)
    assert "axis = 1 : i32" in str(ir)
    assert f"tensor<{factor}x4xf32>" in str(ir)


@pytest.mark.parametrize("extra_use,layout_hint", [(True, False), (False, True)])
def test_identity_normalization_does_not_widen_reorder_permission(tmp_path, extra_use, layout_hint):
    text = reduction_ir("sum").replace(
        '%reduced = "tt.reduce"(%v)',
        '%flat = tt.reshape %v allow_reorder' + (' efficient_layout' if layout_hint else '') +
        ' : tensor<4xf32> -> tensor<4xf32>\n    %reduced = "tt.reduce"(%flat)')
    if extra_use:
        text = text.replace("tt.return", "%extra = arith.addf %flat, %flat : tensor<4xf32>\n    tt.return")
    ir = parse(tmp_path, text)
    before = str(ir)
    result = ascend.passes.ttir.rewrite_block_batching(ir, 3, 0)
    assert result["reason"] == "CROSS_LANE_REORDER"
    assert str(ir) == before


@pytest.mark.parametrize("columns", [(0, 1, 2), (2, 0, 1), (0, 1)])
def test_joint_effect_retains_stores_and_access_ids(tmp_path, columns):
    ir, spec = rewritten_guard(tmp_path, fields_ir(columns))
    assert str(ir).count("tt.store") == len(columns)
    assert len(spec["accesses"]) == len(columns) + 1
    joint = spec["accesses"][columns.index(0) + 1]
    assert joint["shape"] == [len(columns)]
    assert sum(a["mask"] == ["false"] for a in spec["accesses"]) == len(columns) - 1
    args = [0x100000, 0x200000, len(columns)]
    assert guard.check_guard(spec, args, (7, 1, 1), 3)[0]
    args[2] -= 1
    assert guard.check_guard(spec, args, (7, 1, 1), 3) == (False, "OVERLAPPING_BLOCK_WRITES")
    args[2] += 1
    args[0] = args[1]
    assert guard.check_guard(spec, args, (7, 1, 1), 3) == (False, "WRITE_ALIAS")


@pytest.mark.parametrize("columns,masked", [((0, 2), False), ((0, 1, 1), False), ((0, 1, 2), True)])
def test_unproven_groups_keep_original_effects(tmp_path, columns, masked):
    _, spec = rewritten_guard(tmp_path, fields_ir(columns, masked))
    assert all(a["shape"] == [] for a in spec["accesses"])
    assert all(a["mask"] != ["false"] for a in spec["accesses"])
    assert guard.check_guard(spec, [0x100000, 0x200000, 3], (7, 1, 1), 3) == (False, "WRITE_ALIAS")


def test_disabled_request_preserves_identity_reshape(tmp_path):
    text = reduction_ir("sum").replace(
        '%reduced = "tt.reduce"(%v)', '%flat = tt.reshape %v allow_reorder : tensor<4xf32> -> tensor<4xf32>\n'
        '    %reduced = "tt.reduce"(%flat)')
    ir = parse(tmp_path, text)
    before = str(ir)
    report = ascend.passes.ttir.rewrite_block_batching(ir, 1, 0)
    assert report["compile_status"] == "DISABLED"
    assert str(ir) == before


def test_loop_store_alias_remains_guarded_after_identity_normalization(tmp_path):
    from block_batching_control_flow_cases import control_ir
    text = control_ir("for_accumulate").replace(
        "scf.yield %updated : f32", "%v = tt.splat %updated : f32 -> tensor<4xf32>\n"
        "%flat = tt.reshape %v allow_reorder : tensor<4xf32> -> tensor<4xf32>\n"
        '%r = "tt.reduce"(%flat) <{axis = 0 : i32}> ({\n'
        '^bb0(%a: f32, %b: f32):\n'
        '%pair_sum = arith.addf %a, %b : f32\n'
        'tt.reduce.return %pair_sum : f32\n'
        '}) : (tensor<4xf32>) -> f32\n'
        "tt.store %out, %r : !tt.ptr<f32>\nscf.yield %updated : f32")
    ir = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(ir, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert ir.verify()
    spec = json.loads(report['guard_json'])
    assert guard.check_guard(spec, [1 << 32, 1 << 40, 7, 4], (7, 1, 1), 3) == (False, 'OVERLAPPING_BLOCK_WRITES')
