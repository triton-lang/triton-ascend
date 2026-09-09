# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Direct contracts for the shared per-program-axis dependence analysis."""

from pathlib import Path

import pytest

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


def _facts(text, tmp_path, name):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / f"{name}.mlir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    function = module.get_function(name)
    return {result["axis"]: result for result in ascend_ir.analyze_program_axis_dependence(function)}


def _store_module(name, stride):
    return f"""
module {{
  tt.func public @{name}(%out: !tt.ptr<i32>) {{
    %pid = tt.get_program_id x : i32
    %stride = arith.constant {stride} : i32
    %base = arith.muli %pid, %stride : i32
    %base_splat = tt.splat %base : i32 -> tensor<16xi32>
    %range = tt.make_range {{end = 16 : i32, start = 0 : i32}} : tensor<16xi32>
    %offsets = arith.addi %base_splat, %range : tensor<16xi32>
    %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %out_ptrs = tt.addptr %out_splat, %offsets : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %zero = arith.constant dense<0> : tensor<16xi32>
    tt.store %out_ptrs, %zero : tensor<16x!tt.ptr<i32>>
    tt.return
  }}
}}
"""


def _masked_store_module(name, stride, valid_lanes):
    return f"""
module {{
  tt.func public @{name}(%out: !tt.ptr<i32>) {{
    %pid = tt.get_program_id x : i32
    %stride = arith.constant {stride} : i32
    %base = arith.muli %pid, %stride : i32
    %range = tt.make_range {{end = 256 : i32, start = 0 : i32}} : tensor<256xi32>
    %out_base = tt.addptr %out, %base : !tt.ptr<i32>, i32
    %out_splat = tt.splat %out_base : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %limit = arith.constant dense<{valid_lanes}> : tensor<256xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<256xi32>
    %zero = arith.constant dense<0> : tensor<256xi32>
    tt.store %out_ptrs, %zero, %mask : tensor<256x!tt.ptr<i32>>
    tt.return
  }}
}}
"""


def _program_grid_module(*, version=1, axis=0, factor=2):
    return f'''
module attributes {{hacc.program_grid_transforms = {{
  version = {version} : i64,
  transforms = [{{
    order = 0 : i32,
    kind = "ceil_div",
    axis = {axis} : i32,
    factor = {factor} : i64,
    logical_extent = 8 : i64,
    persistent_coverage = false,
    grid_stride_abi_verified = false
  }}]
}}}} {{
  tt.func public @grid_contract_fixture() {{
    tt.return
  }}
}}
'''


def test_program_axis_analysis_reports_closed_disjoint_store_slice(tmp_path):
    facts = _facts(_store_module("disjoint", 16), tmp_path, "disjoint")
    x = facts[0]

    assert x["program_id_count"] == 1
    assert "arith.muli" in x["dependence_closure"]
    assert "tt.addptr" in x["dependence_closure"]
    assert "tt.store" in x["dependence_closure"]
    assert x["reduction_axes"] == []
    assert x["escapes"] is False
    assert x["has_side_effects"] is True
    assert x["has_unsupported_side_effects"] is False
    assert x["reads_num_programs"] is False
    assert x["stores"] == [{
        "pointer_depends_on_axis": True,
        "address_independence": "proven_disjoint",
    }]
    assert x["is_independent_axis_transform_candidate"] is True
    assert facts[1]["program_id_count"] == facts[2]["program_id_count"] == 0


def test_program_axis_analysis_rejects_overlapping_store_intervals(tmp_path):
    facts = _facts(_store_module("overlapping", 1), tmp_path, "overlapping")
    x = facts[0]

    assert x["stores"][0]["pointer_depends_on_axis"] is True
    assert x["stores"][0]["address_independence"] == "unknown"
    assert x["is_independent_axis_transform_candidate"] is False


def test_program_axis_analysis_uses_exact_final_store_mask_to_bound_lanes(tmp_path):
    facts = _facts(
        _masked_store_module("masked_disjoint", stride=192, valid_lanes=192),
        tmp_path,
        "masked_disjoint",
    )
    x = facts[0]

    assert x["stores"] == [{
        "pointer_depends_on_axis": True,
        "address_independence": "proven_disjoint",
    }]
    assert x["is_independent_axis_transform_candidate"] is True


def test_program_axis_analysis_rejects_mask_bound_equal_to_program_stride(tmp_path):
    facts = _facts(
        _masked_store_module("masked_overlap", stride=192, valid_lanes=193),
        tmp_path,
        "masked_overlap",
    )
    x = facts[0]

    assert x["stores"][0]["address_independence"] == "unknown"
    assert x["is_independent_axis_transform_candidate"] is False


def test_program_axis_analysis_tracks_reductions_escape_effects_and_per_axis_num_programs(tmp_path, ):
    facts = _facts(
        """
module {
  tt.func private @side_effect(%value: i32) {
    tt.return
  }
  tt.func public @unsafe(%out: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %programs_y = tt.get_num_programs y : i32
    %pid_splat = tt.splat %pid : i32 -> tensor<16xi32>
    %reduced = "tt.reduce"(%pid_splat) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.reduce.return %sum : i32
    }) : (tensor<16xi32>) -> i32
    tt.call @side_effect(%pid) : (i32) -> ()
    %zero = arith.constant 0 : i32
    %exit = arith.cmpi eq, %pid, %zero : i32
    cf.cond_br %exit, ^bb_return, ^bb_work
  ^bb_return:
    tt.return
  ^bb_work:
    tt.return
  }
}
""",
        tmp_path,
        "unsafe",
    )
    x = facts[0]
    y = facts[1]

    assert x["reduction_axes"] == [0]
    assert "tt.reduce" in x["dependence_closure"]
    assert "tt.call" in x["dependence_closure"]
    assert "cf.cond_br" in x["dependence_closure"]
    assert x["escapes"] is True
    assert x["has_unsupported_side_effects"] is True
    assert x["is_independent_axis_transform_candidate"] is False
    assert x["reads_num_programs"] is False
    assert y["program_id_count"] == 0
    assert y["reads_num_programs"] is True


def test_cxx_program_grid_schema_accepts_the_frozen_contract(tmp_path):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / "valid-grid-contract.mlir"
    path.write_text(_program_grid_module())
    module = ir.parse_mlir_module(str(path), context)

    assert ascend_ir.get_program_grid_transforms(module) == {
        "version":
        1,
        "transforms": [{
            "order": 0,
            "kind": "ceil_div",
            "axis": 0,
            "factor": 2,
            "logical_extent": 8,
            "persistent_coverage": False,
            "grid_stride_abi_verified": False,
        }],
    }


@pytest.mark.parametrize(
    ("version", "axis", "factor"),
    [(2, 0, 2), (1, 3, 2), (1, 0, 1)],
)
def test_cxx_program_grid_schema_rejects_invalid_version_axis_or_factor(
    tmp_path,
    version,
    axis,
    factor,
):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / f"invalid-grid-contract-{version}-{axis}-{factor}.mlir"
    path.write_text(_program_grid_module(version=version, axis=axis, factor=factor))
    module = ir.parse_mlir_module(str(path), context)

    with pytest.raises(RuntimeError, match="invalid hacc.program_grid_transforms contract"):
        ascend_ir.get_program_grid_transforms(module)


def test_cxx_grid_specialization_is_versioned_and_cleared_from_module_and_function(tmp_path):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / "grid-specialization.mlir"
    path.write_text("""
module {
  tt.func public @grid_specialization_fixture() {
    tt.return
  }
}
""")
    module = ir.parse_mlir_module(str(path), context)
    function = module.get_function("grid_specialization_fixture")

    ascend_ir.set_program_grid_specialization(module, 1, 8, 65, 1, 512)
    expected = {"version": 1, "grid": [8, 65, 1], "rule_mask": 512}
    assert ascend_ir.get_program_grid_specialization(module) == expected
    assert ascend_ir.get_program_grid_specialization(function) == expected
    assert "hacc.grid_specialization" in str(module)

    ascend_ir.clear_program_grid_specialization(module)
    assert ascend_ir.get_program_grid_specialization(module) is None
    assert ascend_ir.get_program_grid_specialization(function) is None


def test_cxx_runtime_scalar_specialization_is_versioned_and_cleared_from_module_and_function(tmp_path, ):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / "runtime-scalar-specialization.mlir"
    path.write_text("""
module {
  tt.func public @runtime_scalar_specialization_fixture(%stride: i32) {
    tt.return
  }
}
""")
    module = ir.parse_mlir_module(str(path), context)
    function = module.get_function("runtime_scalar_specialization_fixture")

    ascend_ir.set_program_mapping_scalar_specialization(module, 1, [(2, 64), (10, -4)])
    expected = {
        "version": 1,
        "arguments": [
            {"index": 2, "value": 64},
            {"index": 10, "value": -4},
        ],
    }
    assert ascend_ir.get_program_mapping_scalar_specialization(module) == expected
    assert ascend_ir.get_program_mapping_scalar_specialization(function) == expected
    assert "hacc.program_mapping_scalar_specialization" in str(module)

    ascend_ir.clear_program_mapping_scalar_specialization(module)
    assert ascend_ir.get_program_mapping_scalar_specialization(module) is None
    assert ascend_ir.get_program_mapping_scalar_specialization(function) is None


def test_cxx_named_runtime_scalar_specialization_round_trips_parameter_names(tmp_path):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / "named-runtime-scalar-specialization.mlir"
    path.write_text("""
module {
  tt.func public @named_runtime_scalar_specialization_fixture(%stride: i32) {
    tt.return
  }
}
""")
    module = ir.parse_mlir_module(str(path), context)
    function = module.get_function("named_runtime_scalar_specialization_fixture")

    ascend_ir.set_program_mapping_scalar_specialization(module, 2, [(4, "stride", 64)])
    expected = {
        "version": 2,
        "arguments": [
            {"index": 4, "name": "stride", "value": 64},
        ],
    }
    assert ascend_ir.get_program_mapping_scalar_specialization(module) == expected
    assert ascend_ir.get_program_mapping_scalar_specialization(function) == expected


@pytest.mark.parametrize(
    ("version", "grid0", "rule_mask"),
    [(2, 8, 512), (1, 0, 512), (1, 8, 1), (1, 8, 1 << 32)],
)
def test_cxx_grid_specialization_rejects_invalid_static_inputs(
    tmp_path,
    version,
    grid0,
    rule_mask,
):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = Path(tmp_path) / "invalid-grid-specialization.mlir"
    path.write_text("""
module {
  tt.func public @grid_specialization_fixture() {
    tt.return
  }
}
""")
    module = ir.parse_mlir_module(str(path), context)

    with pytest.raises((RuntimeError, ValueError), match="grid_specialization|rule_mask"):
        ascend_ir.set_program_grid_specialization(module, version, grid0, 1, 1, rule_mask)
