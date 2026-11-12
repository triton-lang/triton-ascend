# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from test_block_batching_memory import parse
from block_batching_control_flow_cases import CONTROL_KINDS, control_ir
from triton._C.libtriton import ascend

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("kind", CONTROL_KINDS)
@pytest.mark.parametrize("factor", [3, 6])
def test_recursive_uniform_regions_preserve_lane_results(tmp_path, kind, factor):
    module = parse(tmp_path, control_ir(kind))
    # Exercise the original lane model independently of the automatic loop
    # schedule. Sequential composition has its own real-kernel regressions.
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, 'semantic_only')
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()
    text = str(module)
    if kind not in ("if_effects", ):
        assert f"tensor<{factor}xf32>" in text
    if kind in ("for_accumulate", "for_fixed_point", "if_for", "for_if", "bounded_read_loop"):
        assert "scf.for" in text
        assert "iter_args" in text
    if kind == "bounded_read_loop":
        guard = json.loads(report["guard_json"])
        read = next(access for access in guard["accesses"] if not access["write"])
        assert len(read["loops"]) == 1
        assert read["loops"][0]["upper"] == ["arg", 3, 32]
        assert '"iv"' in json.dumps(read["offset"])


@pytest.mark.parametrize("kind,old,new,reason", [
    ("if_join", "arith.cmpi sgt, %k, %zero", "arith.cmpi sgt, %p, %zero", "VARYING_CONTROL_CONDITION"),
    ("for_accumulate", "to %k step", "to %p step", "LOOP_WITHOUT_MEMORY_ACCESS"),
    ("for_accumulate", "step %one iter_args", "step %k iter_args", "UNPROVEN_POSITIVE_LOOP_STEP"),
    ("for_accumulate", "%one = arith.constant 1", "%one = arith.constant -1", "UNPROVEN_POSITIVE_LOOP_STEP"),
])
def test_unknown_or_unsupported_regions_keep_complete_baseline(tmp_path, kind, old, new, reason):
    module = parse(tmp_path, control_ir(kind).replace(old, new))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED", report
    assert report["reason"] == reason
    assert str(module) == before


def test_loop_and_terminal_stores_still_require_noninterference(tmp_path):
    from test_block_batching_guard_v2 import module as guard
    source = control_ir('for_accumulate').replace(
        'scf.yield %updated : f32', 'tt.store %out, %updated : !tt.ptr<f32>\n      scf.yield %updated : f32')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert module.verify()
    spec = json.loads(report['guard_json'])
    # The existing write-domain proof rejects repeated writes before the
    # subsequent cross-access alias check is reached.
    assert guard.check_guard(spec, [1 << 32, 1 << 40, 7, 4], (7, 1, 1), 3) == (False, 'OVERLAPPING_BLOCK_WRITES')


@pytest.mark.parametrize("kind", CONTROL_KINDS)
def test_preflight_uses_the_same_uniform_region_models(tmp_path, kind):
    module = parse(tmp_path, control_ir(kind))
    before = str(module)
    report = ascend.passes.ttir.analyze_block_batching(module, 3, 0)
    assert report["compile_status"] == "ANALYZED", report
    assert str(module) == before


def test_nested_varying_condition_reports_its_original_region(tmp_path):
    text = control_ir("for_if").replace("arith.cmpi slt, %iv, %one", "arith.cmpi slt, %p, %one")
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.analyze_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "VARYING_CONTROL_CONDITION"
    assert report["operation"] == "scf.if"
    assert report["context"] == "ForBody"
    assert report["location"]
    assert str(module) == before
