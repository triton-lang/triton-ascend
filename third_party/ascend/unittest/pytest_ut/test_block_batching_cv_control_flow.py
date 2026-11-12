# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CV uniform-region proof, exact lane cloning and pointer recurrence guards."""
import json

import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching_guard import check_guard
from test_block_batching_cv import dot_module
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


DOT = "%d = tt.dot %a, %bmat, %acc, inputPrecision = ieee : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>"


def loop_dot():
    return dot_module().replace(
        DOT, """
    %zero_i = arith.constant 0 : i32
    %one_i = arith.constant 1 : i32
    %d = scf.for %iv = %zero_i to %n step %one_i
        iter_args(%carry = %acc) -> (tensor<4x4xf32>) : i32 {
      %nested_p = tt.get_program_id x : i32
      %nested_g = tt.get_num_programs x : i32
      %s = arith.addi %nested_p, %nested_g : i32
      %sf = arith.sitofp %s : i32 to f32
      %sv = tt.splat %sf : f32 -> tensor<4x4xf32>
      %c = arith.addf %carry, %sv : tensor<4x4xf32>
      %next = tt.dot %a, %bmat, %c : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
      scf.yield %next : tensor<4x4xf32>
    }
    """)


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("batch_matmul", [False, True])
def test_loop_accumulators_and_nested_grid_queries_stay_per_lane(tmp_path, factor, batch_matmul):
    module = parse(tmp_path, loop_dot())
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, "semantic_only", True, batch_matmul)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["cv_mode"] == "independent_batch_v1"
    assert report["cv_lowering"] == "static_uniform_control_lanes_v1"
    text = str(module)
    assert text.count("scf.for") == factor
    assert text.count("tt.dot ") == factor
    assert text.count("tt.get_program_id") == 1
    assert "tt.get_num_programs" not in text
    assert f"tensor<{factor}x4x4xf32>" not in text
    assert module.verify()
    proof = json.loads(report["guard_json"])["cv_control_flow_contract"]
    assert proof == dict(version=1, lane_semantics="original_program_clone", uniform_bounds=True, dot_rank=2,
                         for_count=1, if_count=0)


@pytest.mark.parametrize("change,reason", [
    (("to %n step", "to %p step"), "CV_VARYING_LOOP_BOUND"),
    (("step %one_i", "step %n"), "UNPROVEN_POSITIVE_LOOP_STEP"),
])
def test_loop_rejection_is_transactional(tmp_path, change, reason):
    module = parse(tmp_path, loop_dot().replace(*change))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True)
    assert report["reason"] == reason, report
    assert str(module) == before


def test_program_id_only_inside_uniform_if_is_remapped(tmp_path):
    text = dot_module().replace(
        "    %p =", """
    %z = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %n, %z : i32
    scf.if %cond {
    %p =""").replace("    tt.return", "    scf.yield\n    }\n    tt.return")
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True)
    assert report["compile_status"] == "REWRITTEN", report
    assert str(module).count("scf.if") == 2
    assert str(module).count("tt.get_program_id") == 1
    assert "tt.get_num_programs" not in str(module)
    assert module.verify()


def test_varying_if_inside_loop_is_rejected(tmp_path):
    text = loop_dot().replace(
        "      %next =", """
      %cond = arith.cmpi sgt, %nested_p, %zero_i : i32
      scf.if %cond { scf.yield }
      %next =""")
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True)
    assert report["reason"] == "VARYING_CONTROL_CONDITION", report
    assert str(module) == before


def pointer_loop():
    # A 4x4 dot in each iteration, with an independently advanced input pointer.
    # This deliberately uses no grouped-grid arithmetic, so the test isolates
    # recurrence provenance rather than an unsupported division expression.
    text = dot_module()
    start = text.index("    %v = tt.load")
    end = text.index('    %result = "tt.reduce"')
    body = text[start:end].replace("%xa", "%ptr").replace("%acc, inputPrecision", "%carry, inputPrecision")
    body = body.replace("    %acc = arith.constant dense<0.0> : tensor<4x4xf32>\n", "")
    body = body.replace("%d = tt.dot", "%next = tt.dot")
    return text[:start] + """
    %acc = arith.constant dense<0.0> : tensor<4x4xf32>
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %advance = arith.constant dense<16> : tensor<4xi32>
    %d:2 = scf.for %iv = %z to %two step %one
        iter_args(%ptr = %xa, %carry = %acc) -> (tensor<4x!tt.ptr<f32>>, tensor<4x4xf32>) : i32 {
    %ivv = tt.splat %iv : i32 -> tensor<4xi32>
    %loopmask = arith.cmpi slt, %ivv, %nv : tensor<4xi32>
    """ + body.replace("tt.load %ptr :", "tt.load %ptr, %loopmask, %zero :") + """
    %nextptr = tt.addptr %ptr, %advance : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    scf.yield %nextptr, %next : tensor<4x!tt.ptr<f32>>, tensor<4x4xf32>
    }
    """ + text[end:].replace('"tt.reduce"(%d)', '"tt.reduce"(%d#1)')


def test_unit_step_pointer_recurrence_has_exact_byte_domain(tmp_path):
    module = parse(tmp_path, pointer_loop())
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True)
    assert report["compile_status"] == "REWRITTEN", report
    guard = json.loads(report["guard_json"])
    read = guard["accesses"][0]
    assert read["device_address"] is False
    assert read["root"] == 0 and len(read["loops"]) == 1
    assert '"iv"' in json.dumps(read["offset"])
    assert check_guard(guard, [1 << 30, 1 << 40, 32], (4, 1, 1), 2)[0]
    # The new summary must not turn cross-program input/output alias into a pass.
    assert not check_guard(guard, [1 << 30, 1 << 30, 32], (4, 1, 1), 2)[0]
    assert module.verify()


def test_loop_dependent_pointer_increment_is_not_summarized(tmp_path):
    text = pointer_loop().replace("tt.addptr %ptr, %advance", "tt.addptr %ptr, %ivv")
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, "semantic_only", True)
    assert report["compile_status"] == "REJECTED", report
    assert str(module) == before


def test_uniform_control_auto_blockify_requires_owned_contract_and_launch_checks():
    from triton.backends.ascend import compiler
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    from test_block_batching_guard import cv_metadata
    from types import SimpleNamespace

    metadata = cv_metadata()
    metadata["block_batching"]["cv_lowering"] = "static_uniform_control_lanes_v1"
    with pytest.raises(BatchingRejected, match="CV_UNIFORM_CONTROL_CONTRACT_MISSING"):
        validate_compilation_result(metadata)
    metadata["block_batching"]["guard"] = {}
    metadata["block_batching"]["guard"]["cv_control_flow_contract"] = dict(version=1,
                                                                           lane_semantics="original_program_clone",
                                                                           uniform_bounds=True, dot_rank=2, for_count=1,
                                                                           if_count=0)
    metadata["auto_blockify_enabled"] = False
    validate_compilation_result(metadata)
    metadata["auto_blockify_enabled"] = True
    metadata['has_auto_blockify_blacklist_op'] = False
    validate_compilation_result(metadata)
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata["auto_blockify_enabled"] is True
    metadata['has_auto_blockify_blacklist_op'] = True
    with pytest.raises(BatchingRejected, match="CV_AUTO_BLOCKIFY_NOT_PROVEN"):
        validate_compilation_result(metadata)
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata["auto_blockify_enabled"] is False
    metadata['has_auto_blockify_blacklist_op'] = False
    metadata['mix_mode'] = 'aic'
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata["auto_blockify_enabled"] is False


@pytest.mark.parametrize('data_dependent', [False, True])
def test_control_lower_bound_mask_keeps_stronger_memory_guard(tmp_path, data_dependent):
    text = loop_dot().replace(
        '    %v = tt.load %xa :', '''
    %zi = arith.constant dense<0> : tensor<4xi32>
    %lower = arith.cmpi sge, %range, %zi : tensor<4xi32>
    %both = arith.andi %lower, %mask : tensor<4xi1>
    %v = tt.load %xa, %both, %zero :''')
    if data_dependent:
        text = text.replace(
            '    %lower = arith.cmpi sge, %range, %zi : tensor<4xi32>', '''
        %data = tt.load %xa : tensor<4x!tt.ptr<f32>>
        %lower = arith.cmpf ogt, %data, %zero : tensor<4xf32>''')
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True)
    if data_dependent:
        assert report['reason'] == 'CV_UNSTRUCTURED_MEMORY_MASK', report
        assert str(module) == before
    else:
        assert report['compile_status'] == 'REWRITTEN', report
        assert str(module).count('arith.cmpi sge') == 2
        guard = json.loads(report['guard_json'])
        assert check_guard(guard, [1 << 30, 1 << 40, 32], (4, 1, 1), 2)[0]
        assert not check_guard(guard, [1 << 30, 1 << 30, 32], (4, 1, 1), 2)[0]
        assert module.verify()


@pytest.mark.parametrize('other', ['0.0', '1.0', '0x7FC00000'])
def test_normalized_block_pointer_zero_padding_is_recognized(tmp_path, other):
    text = loop_dot().replace(
        '%zero = arith.constant dense<0.0> : tensor<4xf32>',
        f'%scalar_zero = arith.constant {other} : f32\n    %zero = tt.splat %scalar_zero : f32 -> tensor<4xf32>')
    text = text.replace('tt.load %xa :', 'tt.load %xa, %mask, %zero :')
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True)
    if other == '0.0':
        assert report['compile_status'] == 'REWRITTEN', report
        assert str(module).count('tt.load') == 2
        assert module.verify()
    else:
        assert report['reason'] == 'CV_MASKED_LOAD_REQUIRES_ZERO_OTHER', report
        assert str(module) == before
