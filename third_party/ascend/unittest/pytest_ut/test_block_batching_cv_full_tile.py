"""Masked batch MMA and the legacy unmasked full-tile proof contract."""
import copy
import json
import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching_guard_v2 import check_guard_v2
from test_block_batching_cv import dot_module
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend("native")


def masked_dot():
    return dot_module().replace("tt.load %xa :", "tt.load %xa, %mask, %zero :").replace(
        "tt.store %ya, %result :", "tt.store %ya, %result, %mask :").replace("slt, %idx, %nv", "slt, %range, %nv")


def test_batch_mode_rejects_untraceable_dot_operands(tmp_path):
    module = parse(tmp_path, masked_dot())
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 4, 0, "semantic_only", True, True)
    assert report["compile_status"] == "REJECTED", report
    assert report["reason"] == "CV_MASKED_DOT_COORDINATES_NOT_PROVEN"
    assert str(module) == before


def test_default_masked_lowering_is_unchanged(tmp_path):
    module = parse(tmp_path, masked_dot())
    report = ascend.passes.ttir.rewrite_block_batching(module, 4, 0, "semantic_only", True)
    assert report["compile_status"] == "REWRITTEN"
    assert report["cv_lowering"] == "static_masked_lanes_v1"
    assert str(module).count("tt.dot ") == 4


def test_source_dot_cannot_supply_internal_mkn_contract(tmp_path):
    source = transposed_load_dot().replace("tt.dot %av, %bv, %zero :",
                                           "tt.dot %av, %bv, %zero {tt.cv_batch_full_mkn} :")
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 4, 0, "semantic_only", True, True)
    assert report["compile_status"] == "REJECTED", report
    assert str(module) == before


def rewritten_metadata(tmp_path):
    from test_block_batching_guard import cv_metadata
    module = parse(tmp_path, transposed_load_dot())
    report = ascend.passes.ttir.rewrite_block_batching(module, 4, 0, "semantic_only", True, True)
    assert report["compile_status"] == "REWRITTEN", report
    assert "tt.cv_batch_full_mkn" in str(module)
    report["guard"] = json.loads(report.pop("guard_json"))
    metadata = cv_metadata()
    metadata["block_batching"] = report
    metadata["enable_hivm_batch_matmul"] = True
    report["cv_memory_lowering"] = "full_reduction_guard_v3"
    report["guard"]["require_same_fractal_bounds"] = True
    return metadata


def test_native_full_tile_report_passes_final_compilation_contract(tmp_path):
    from triton.backends.ascend.block_batching_report import validate_compilation_result, CV_VALIDATION_POLICY
    metadata = rewritten_metadata(tmp_path)
    validate_compilation_result(metadata)
    assert metadata["block_batching"]["compile_status"] == "COMPILED_GUARDED"
    assert metadata["block_batching"]["validation_policy"] == CV_VALIDATION_POLICY


@pytest.mark.parametrize("field", ["enable_hivm_batch_matmul", "require_full_masks"])
@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_full_tile_final_contract_requires_both_strict_boolean_proofs(tmp_path, field, value):
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    metadata = rewritten_metadata(tmp_path)
    metadata["block_batching"]["cv_lowering"] = "rank3_full_tile_batch_v2"
    metadata["block_batching"]["guard"]["require_full_masks"] = True
    target = metadata if field == "enable_hivm_batch_matmul" else metadata["block_batching"]["guard"]
    if value is None:
        target.pop(field)
    else:
        target[field] = value
    with pytest.raises(BatchingRejected, match="CV_FULL_TILE_BATCH_CONTRACT_MISSING"):
        validate_compilation_result(metadata)


def test_full_tile_final_contract_rejects_unknown_lowering(tmp_path):
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    metadata = rewritten_metadata(tmp_path)
    metadata["block_batching"]["cv_lowering"] = "rank3_full_tile_batch_unknown"
    with pytest.raises(BatchingRejected, match="CV_LOWERING_CONTRACT_MISSING"):
        validate_compilation_result(metadata)


@pytest.mark.parametrize("value", [True, False, 1, None])
def test_masked_batch_contract_cannot_request_all_true_launch_proof(tmp_path, value):
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    metadata = rewritten_metadata(tmp_path)
    metadata["block_batching"]["guard"]["require_full_masks"] = value
    with pytest.raises(BatchingRejected, match="CV_MASKED_BATCH_CONTRACT_MISSING"):
        validate_compilation_result(metadata)


@pytest.mark.parametrize("value", [None, "default_nd2nz", True])
def test_masked_batch_contract_requires_safe_memory_lowering(tmp_path, value):
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    metadata = rewritten_metadata(tmp_path)
    metadata["block_batching"]["cv_memory_lowering"] = value
    with pytest.raises(BatchingRejected, match="CV_MASKED_BATCH_MEMORY_CONTRACT_MISSING"):
        validate_compilation_result(metadata)


def test_ragged_batch_mask_is_rejected_transactionally(tmp_path):
    module = parse(tmp_path, masked_dot().replace("slt, %range, %nv", "slt, %idx, %nv"))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True, True)
    assert report["reason"] == "CV_BATCH_MASK_RECTANGLE_NOT_PROVEN", report
    assert str(module) == before


def memory_guard():
    index = ["addi", 32, ["muli", 32, ["pid", 0], ["const", 32, 32]], ["coord", 0, 32]]
    return dict(
        version=2, axis=0, require_full_masks=True, accesses=[
            dict(root=0, shape=[32], bytes=4, write=False, offset=["muli", 32, index, ["const", 4, 32]],
                 mask=["slt", index, ["arg", 1, 32], 32], pointer_checks=[], loops=[])
        ])


@pytest.mark.parametrize("limit,accepted", [(64, True), (65, True), (63, False), (33, False), (0, False)])
def test_mask_truth_uses_full_original_grid(limit, accepted):
    guard = memory_guard()
    ok, reason = check_guard_v2(guard, [4096, limit], [2, 1, 1], 2)
    assert ok is accepted
    if not accepted:
        assert reason == "CV_FULL_TILE_MASK_NOT_PROVEN"


@pytest.mark.parametrize("mask", [["unknown"], ["false"], ["unsupported"], ["and", ["true"], ["unknown"]]])
def test_unknown_or_inactive_mask_cannot_authorize_unmasked_access(mask):
    guard = memory_guard()
    guard["accesses"][0]["mask"] = mask
    assert check_guard_v2(guard, [4096, 64], [2, 1, 1], 2) == (False, "CV_FULL_TILE_MASK_NOT_PROVEN")


def test_ordinary_masked_domain_can_still_handle_tail():
    guard = memory_guard()
    del guard["require_full_masks"]
    assert check_guard_v2(guard, [4096, 33], [2, 1, 1], 2)[0]


def test_unsigned_mask_with_negative_index_fails_closed():
    guard = memory_guard()
    guard["accesses"][0]["mask"] = ["ult", ["const", -1, 32], ["arg", 1, 32], 32]
    assert not check_guard_v2(guard, [4096, 64], [2, 1, 1], 2)[0]


def test_mask_proof_never_waives_index_overflow():
    guard = memory_guard()
    assert not check_guard_v2(guard, [4096, 2147483647], [134217728, 1, 1], 2)[0]


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_full_mask_contract_requires_boolean(invalid):
    guard = memory_guard()
    guard["require_full_masks"] = invalid
    assert check_guard_v2(guard, [4096, 64], [2, 1, 1], 2) == (False, "INVALID_GUARD_SCHEMA")


def transposed_load_dot():
    return """module {
  tt.func public @arbitrary_transposed_dot(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %p = tt.get_program_id x : i32
    %c32 = arith.constant 32 : i32
    %base = arith.muli %p, %c32 : i32
    %r4 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r8 = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %m = tt.expand_dims %r4 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %k = tt.expand_dims %r8 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %mm = tt.broadcast %m : tensor<4x1xi32> -> tensor<4x8xi32>
    %kk = tt.broadcast %k : tensor<1x8xi32> -> tensor<4x8xi32>
    %s4 = arith.constant dense<4> : tensor<4x8xi32>
    %ks = arith.muli %kk, %s4 : tensor<4x8xi32>
    %offset = arith.addi %mm, %ks : tensor<4x8xi32>
    %bs = tt.splat %base : i32 -> tensor<4x8xi32>
    %at = arith.addi %bs, %offset : tensor<4x8xi32>
    %ap = tt.splat %a : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %ptr = tt.addptr %ap, %at : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %av = tt.load %ptr : tensor<4x8x!tt.ptr<f32>>
    %bv = tt.trans %av {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %dot = tt.dot %av, %bv, %zero : tensor<4x8xf32> * tensor<8x4xf32> -> tensor<4x4xf32>
    %row = tt.broadcast %m : tensor<4x1xi32> -> tensor<4x4xi32>
    %n = tt.expand_dims %r4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %col = tt.broadcast %n : tensor<1x4xi32> -> tensor<4x4xi32>
    %cs4 = arith.constant dense<4> : tensor<4x4xi32>
    %rs = arith.muli %row, %cs4 : tensor<4x4xi32>
    %co = arith.addi %rs, %col : tensor<4x4xi32>
    %cbase = tt.splat %base : i32 -> tensor<4x4xi32>
    %ct = arith.addi %cbase, %co : tensor<4x4xi32>
    %cp = tt.splat %c : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %cptr = tt.addptr %cp, %ct : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %cptr, %dot : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_column_major_load_is_materialized_before_batch_transpose(tmp_path, factor):
    module = parse(tmp_path, transposed_load_dot())
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, "semantic_only", True, True)
    assert report["compile_status"] == "REWRITTEN", report
    text = str(module)
    load, = [line for line in text.splitlines() if "tt.load " in line]
    assert f"tensor<{factor}x8x4x!tt.ptr<f32>>" in load
    dot, = [line for line in text.splitlines() if "tt.dot " in line]
    assert f"tensor<{factor}x4x8xf32> * tensor<{factor}x8x4xf32>" in dot
    transposes = [line for line in text.splitlines() if "tt.trans " in line]
    assert len(transposes) == 2 and all("!tt.ptr" not in line for line in transposes)
    guard = json.loads(report["guard_json"])
    assert guard["accesses"][0]["shape"] == [4, 8]  # proof stays in original coordinates
    assert module.verify()


def test_transposed_load_default_lowering_remains_unchanged(tmp_path):
    module = parse(tmp_path, transposed_load_dot())
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True)
    assert report["compile_status"] == "REWRITTEN", report
    load, = [line for line in str(module).splitlines() if "tt.load " in line]
    assert "tensor<2x4x8x!tt.ptr<f32>>" in load


@pytest.mark.parametrize("factor", [2, 4])
def test_transposed_load_permutes_original_mask_and_zero_together(tmp_path, factor):
    text = transposed_load_dot().replace("%c: !tt.ptr<f32>)", "%c: !tt.ptr<f32>, %limit: i32)").replace(
        "%av = tt.load %ptr :", """%bound = tt.splat %limit : i32 -> tensor<4x8xi32>
    %mask = arith.cmpi slt, %kk, %bound : tensor<4x8xi32>
    %pad = arith.constant dense<0.0> : tensor<4x8xf32>
    %av = tt.load %ptr, %mask, %pad :""")
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, "semantic_only", True, True)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["cv_lowering"] == "rank3_masked_batch_v1"
    load, = [line for line in str(module).splitlines() if "tt.load " in line]
    assert f"tensor<{factor}x8x4x!tt.ptr<f32>>" in load
    assert load.split(" : ")[0].count(",") == 2
    assert "require_full_masks" not in json.loads(report["guard_json"])
    assert module.verify()


def test_diagonal_mask_does_not_enter_rectangular_batch_path(tmp_path):
    text = transposed_load_dot().replace(
        "%av = tt.load %ptr :", """%mask = arith.cmpi slt, %mm, %kk : tensor<4x8xi32>
    %pad = arith.constant dense<0.0> : tensor<4x8xf32>
    %av = tt.load %ptr, %mask, %pad :""")
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True, True)
    assert report["reason"] == "CV_BATCH_MASK_RECTANGLE_NOT_PROVEN", report
    assert str(module) == before
