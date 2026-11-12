# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Keep CV-specific memory obligations when using the unified effect plan."""
import copy
import json

import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching_guard import check_guard
from triton.backends.ascend.block_batching_guard_v3 import bind_plan, validate_plan_binding
from test_block_batching_cv_memory_lowering import matrix_guard
from test_block_batching_cv_spatial import spatial_module
from test_block_batching_rewrite import parse
from test_block_batching_unified import unified, terminal_guard

pytestmark = pytest.mark.backend("native")


def masked_batch_guard():
    spec = matrix_guard()
    lhs = spec["accesses"][0]
    rhs = dict(copy.deepcopy(lhs), root=3, mask=None)
    output = dict(copy.deepcopy(lhs), root=4, mask=None, write=True)
    spec["accesses"].extend([rhs, output])
    spec["cv_mask_contract"] = dict(
        version=1,
        inputs=[dict(access=0, operand=0, axes=[0, 1]),
                dict(access=1, operand=1, axes=[0, 1])],
        outputs=[2],
    )
    return unified(spec, factor=2)


@pytest.mark.parametrize("m,k,reason", [
    (32, 32, "PROVEN_AT_RUNTIME"),
    (29, 32, "PROVEN_AT_RUNTIME"),
    (24, 32, "CV_BATCH_PADDING_NOT_PROVEN"),
    (32, 29, "CV_BATCH_REDUCTION_MASK_NOT_PROVEN"),
])
def test_unified_guard_keeps_masked_batch_padding_and_reduction_proofs(m, k, reason):
    spec = masked_batch_guard()
    args = [1 << 32, m, k, 1 << 40, 1 << 45]
    assert check_guard(spec, args, (8, 1, 1), 2) == (reason == "PROVEN_AT_RUNTIME", reason)


def test_cv_tail_proof_does_not_waive_unified_alias_or_effect_checks():
    spec = masked_batch_guard()
    args = [1 << 32, 29, 32, 1 << 40, 1 << 32]
    assert check_guard(spec, args, (8, 1, 1), 2) == (False, "WRITE_ALIAS")
    args[-1] = 1 << 45
    spec["obligations"][2]["proof"] = "lane_correspondence"
    assert check_guard(spec, args, (8, 1, 1), 2) == (False, "INVALID_EFFECT_PROOF")


@pytest.mark.parametrize("field", ["require_same_fractal_bounds", "cv_mask_contract", "obligations"])
def test_cv_obligations_are_bound_to_the_plan_fingerprint(field):
    spec = masked_batch_guard()
    bind_plan(spec, "normalized CV source")
    assert validate_plan_binding(spec)
    spec.pop(field)
    assert not validate_plan_binding(spec)


def test_cv_cannot_use_the_vv_terminal_read_relaxation():
    spec = terminal_guard()
    spec["require_same_fractal_bounds"] = True
    assert check_guard(spec, [1 << 32, 1 << 45, 4, 4, 4], (2, 1, 1), 3) == (False, "CV_DEVICE_ADDRESS_NOT_PROVEN")


@pytest.mark.parametrize("mode", ["m", "n"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_spatial_rewrite_exports_a_complete_unified_guard(tmp_path, mode, axis):
    mod = parse(tmp_path, spatial_module(mode, axis))
    source = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, axis, "semantic_only", True)
    assert report["compile_status"] == "REWRITTEN", report
    spec = json.loads(report["guard_json"])
    assert spec["version"] == 3 and spec["plan_kind"] == "unified"
    assert len(spec["obligations"]) == len(spec["accesses"]) == 3
    assert all(not access["device_address"] for access in spec["accesses"])
    bind_plan(spec, source)
    assert validate_plan_binding(spec)
    grid = [1, 1, 1]
    grid[axis] = 4
    assert check_guard(spec, [1 << 32, 1 << 40, 1 << 45, 13], grid, 2)[0]


def test_cv_opaque_address_is_rejected_transactionally(tmp_path):
    source = spatial_module().replace("%limit: i32)", "%limit: i32, %indices: !tt.ptr<i32>)").replace(
        "%aoff = arith.addi %aoff1, %aoff2 : tensor<4x4xi32>", """
    %loaded_offset = tt.load %indices : !tt.ptr<i32>
    %offsets = tt.splat %loaded_offset : i32 -> tensor<4x4xi32>
    %aoff_base = arith.addi %aoff1, %aoff2 : tensor<4x4xi32>
    %aoff = arith.addi %aoff_base, %offsets : tensor<4x4xi32>""")
    mod = parse(tmp_path, source)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, "semantic_only", True)
    assert report["reason"] == "CV_DEVICE_ADDRESS_NOT_PROVEN", report
    assert str(mod) == before
