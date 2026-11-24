"""IR-level coverage and ownership for calibrated ordinary memory stages."""
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

_LOAD = r'''module {
  tt.func public @row_indexed_load(%X: !tt.ptr<i8>, %Index: !tt.ptr<i32>, %Output: !tt.ptr<i8>) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %program_source_span = arith.constant 2097152 : i32
    %program_source_offset = arith.muli %pid, %program_source_span : i32
    %program_X = tt.addptr %X, %program_source_offset : !tt.ptr<i8>, i32
    %program_index_span = arith.constant 4 : i32
    %program_index_offset = arith.muli %pid, %program_index_span : i32
    %program_Index = tt.addptr %Index, %program_index_offset : !tt.ptr<i32>, i32
    %program_output_span = arith.constant 64 : i32
    %program_output_offset = arith.muli %pid, %program_output_span : i32
    %program_Output = tt.addptr %Output, %program_output_offset : !tt.ptr<i8>, i32
    %linear = arith.constant dense<16> : tensor<4x1xi32>
    %pointer = arith.constant dense<32768> : tensor<4x1xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %index_ptr_base = tt.splat %program_Index : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %index_ptrs = tt.addptr %index_ptr_base, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %index_bound = arith.constant 4 : i32
    %index_bound_splat = tt.splat %index_bound : i32 -> tensor<4xi32>
    %index_mask = arith.cmpi slt, %r, %index_bound_splat : tensor<4xi32>
    %index_zero = arith.constant dense<0> : tensor<4xi32>
    %rows_1 = tt.load %index_ptrs, %index_mask, %index_zero : tensor<4x!tt.ptr<i32>>
    %pointer_2 = tt.expand_dims %rows_1 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %pointer_3 = arith.muli %pointer_2, %pointer : tensor<4x1xi32>
    %pointer_4 = tt.splat %program_X : !tt.ptr<i8> -> tensor<4x1x!tt.ptr<i8>>
    %pointer_5 = tt.addptr %pointer_4, %pointer_3 : tensor<4x1x!tt.ptr<i8>>, tensor<4x1xi32>
    %pointer_6 = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %pointer_7 = tt.broadcast %pointer_5 : tensor<4x1x!tt.ptr<i8>> -> tensor<4x16x!tt.ptr<i8>>
    %pointer_8 = tt.broadcast %pointer_6 : tensor<1x16xi32> -> tensor<4x16xi32>
    %pointer_9 = tt.addptr %pointer_7, %pointer_8 : tensor<4x16x!tt.ptr<i8>>, tensor<4x16xi32>
    %linear_10 = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %linear_11 = arith.muli %linear_10, %linear : tensor<4x1xi32>
    %linear_12 = tt.broadcast %linear_11 : tensor<4x1xi32> -> tensor<4x16xi32>
    %linear_13 = arith.addi %linear_12, %pointer_8 : tensor<4x16xi32>
    %proved_H = arith.constant 4 : i32
    %proved_K = arith.constant 16 : i32
    %mask = tt.splat %proved_H : i32 -> tensor<4x1xi32>
    %mask_14 = arith.cmpi slt, %linear_10, %mask : tensor<4x1xi32>
    %mask_15 = tt.splat %proved_K : i32 -> tensor<1x16xi32>
    %mask_16 = arith.cmpi slt, %pointer_6, %mask_15 : tensor<1x16xi32>
    %mask_17 = tt.broadcast %mask_14 : tensor<4x1xi1> -> tensor<4x16xi1>
    %mask_18 = tt.broadcast %mask_16 : tensor<1x16xi1> -> tensor<4x16xi1>
    %mask_19 = arith.andi %mask_17, %mask_18 : tensor<4x16xi1>
    %zero = arith.constant dense<0> : tensor<4x16xi8>
    %loaded = tt.load %pointer_9, %mask_19, %zero : tensor<4x16x!tt.ptr<i8>>
    %outptr = tt.splat %program_Output : !tt.ptr<i8> -> tensor<4x16x!tt.ptr<i8>>
    %outputp = tt.addptr %outptr, %linear_13 : tensor<4x16x!tt.ptr<i8>>, tensor<4x16xi32>
    tt.store %outputp, %loaded : tensor<4x16x!tt.ptr<i8>>
    tt.return
  }
}
'''
_STORE = r'''module {
  tt.func public @row_indexed_store(%X: !tt.ptr<i8>, %Index: !tt.ptr<i32>, %Values: !tt.ptr<i8>) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %rows_per_program = arith.constant 8 : i32
    %values_per_program = arith.constant 256 : i32
    %pid_rows = arith.muli %pid, %rows_per_program : i32
    %pid_values = arith.muli %pid, %values_per_program : i32
    %Index_pid = tt.addptr %Index, %pid_rows : !tt.ptr<i32>, i32
    %Values_pid = tt.addptr %Values, %pid_values : !tt.ptr<i8>, i32
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %index_base = tt.splat %Index_pid : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %index_ptr = tt.addptr %index_base, %r : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %index_rows = tt.load %index_ptr : tensor<8x!tt.ptr<i32>>
    %index_rows_2 = tt.expand_dims %index_rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %index_rows_b = tt.broadcast %index_rows_2 : tensor<8x1xi32> -> tensor<8x32xi32>
    %c = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rb = tt.broadcast %r2 : tensor<8x1xi32> -> tensor<8x32xi32>
    %cb = tt.broadcast %c2 : tensor<1x32xi32> -> tensor<8x32xi32>
    %dense_pitch = arith.constant dense<32> : tensor<8x32xi32>
    %dense_row = arith.muli %rb, %dense_pitch : tensor<8x32xi32>
    %dense_offset = arith.addi %dense_row, %cb : tensor<8x32xi32>
    %values_base = tt.splat %Values_pid : !tt.ptr<i8> -> tensor<8x32x!tt.ptr<i8>>
    %values_ptr = tt.addptr %values_base, %dense_offset : tensor<8x32x!tt.ptr<i8>>, tensor<8x32xi32>
    %values = tt.load %values_ptr : tensor<8x32x!tt.ptr<i8>>
    %target_pitch = arith.constant dense<32768> : tensor<8x32xi32>
    %target_row = arith.muli %index_rows_b, %target_pitch : tensor<8x32xi32>
    %target_offset = arith.addi %target_row, %cb : tensor<8x32xi32>
    %target_base = tt.splat %X : !tt.ptr<i8> -> tensor<8x32x!tt.ptr<i8>>
    %target_ptr = tt.addptr %target_base, %target_offset : tensor<8x32x!tt.ptr<i8>>, tensor<8x32xi32>
    %row_literal = arith.constant 8 : i32
    %row_bound = tt.splat %row_literal : i32 -> tensor<8x1xi32>
    %row_mask = arith.cmpi slt, %r2, %row_bound : tensor<8x1xi32>
    %col_literal = arith.constant 32 : i32
    %col_bound = tt.splat %col_literal : i32 -> tensor<1x32xi32>
    %col_mask = arith.cmpi slt, %c2, %col_bound : tensor<1x32xi32>
    %rm = tt.broadcast %row_mask : tensor<8x1xi1> -> tensor<8x32xi1>
    %cm = tt.broadcast %col_mask : tensor<1x32xi1> -> tensor<8x32xi1>
    %mask = arith.andi %rm, %cm : tensor<8x32xi1>
    tt.store %target_ptr, %values, %mask : tensor<8x32x!tt.ptr<i8>>

    tt.return
  }
}
'''

_TYPES = ("u8", "f16", "bf16", "f32", "i32", "i64")
_LOAD_PRICES = {
    "u8": (139.28159097761974, 137.54592793357193, 219.0001541748643),
    "f16": (2539.2055958800643, 2640.2998334696103, 2755.082428279016),
    "bf16": (2536.706377773883, 2640.2998334696103, 2755.082428279016),
    "f32": (2525.4598962960677, 2692.8057960670176, 2835.084821840657),
    "i32": (2525.4598962960677, 2692.8057960670176, 2835.084821840657),
    "i64": (2591.0017326002285, 2742.237241353064, 2619.7489780656438),
}
_STORE_PRICES = {
    "u8": (4921.062849462032, 5484.313704073429, 4993.4983253479),
    "f16": (60.00002031214535, 3557.6866939663887, 3332.686610519886),
    "bf16": (60.00002031214535, 3557.6866939663887, 3332.686610519886),
    "f32": (1175.9214103221893, 216.93753660656512, 211.25009516254067),
    "i32": (1175.9214103221893, 216.93753660656512, 199.24994558095932),
    "i64": (3417.296800762415, 4134.561866521835, 4543.9377427101135),
}


def memory_module(operation, payload):
    dtype = "i8" if payload == "u8" else payload
    source = (_LOAD if operation == "Load" else _STORE)
    source = source.replace("!tt.ptr<i8>", "!tt.ptr<" + dtype + ">").replace("xi8>", "x" + dtype + ">")
    if dtype in ("f16", "bf16", "f32"):
        source = source.replace("dense<0> : tensor<4x16x" + dtype + ">", "dense<0.0> : tensor<4x16x" + dtype + ">")
    return source


def analyze(tmp_path, source, programs=224, capability=True, factor2=True, baseline=False):
    profile = Path(__file__).resolve().parents[2] / "costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json"
    module_path = tmp_path / "memory.mlir"
    module_path.write_text(source)
    native = os.getenv("TRITON_ASCEND_COSTMODEL_NATIVE_DRIVER")
    if native:
        selector = "2" if not capability else "1" if factor2 else "0"
        result = subprocess.run(
            [native, str(module_path),
             str(profile), "1" if baseline else "0",
             str(programs), selector], capture_output=True, text=True, timeout=20, check=True)
        return json.loads(result.stdout)
    lib = pytest.importorskip("triton._C.libtriton")
    ir, ascend = lib.ir, lib.ascend
    context = ir.context()
    ir.load_dialects(context)
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(module_path), context)
    report = tmp_path / "report.jsonl"
    factors = [1, 2, 4] if factor2 else [1, 4]
    caps = {
        "source_logical_program_grid_hint": [programs, 1, 1],
        "source_logical_program_count_hint": programs,
        "logical_program_count_hint": programs,
        "physical_vector_core_count_hint": 56,
        "layout_coalescing_factor": 1,
        "layout_merge_applied": False,
        "auto_blockify_v1_materializable": True,
        "whole_kernel_superblock_factors": factors,
    }
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(pm, "report", str(profile), "Ascend950PR_9579", 4, True, True,
                                                      True, 4, 4, programs, "",
                                                      json.dumps(caps) if capability else "", str(report))
    pm.run(module, "calibrated_memory")
    return json.loads(report.read_text().splitlines()[-1])


def _routes(report):
    return report["stage_model"]["all_simt_factor_routes"]


def assert_same_report(actual, baseline):
    differences = []

    def semantic_edge(edge):
        # An intra-stage-pair edge represents potentially several SSA values.
        # Its first producer's location/size are diagnostic representatives,
        # selected by DenseSet iteration, and are not used for route pricing.
        representative = edge.get("kind") == "intra_iteration_data" and edge.get(
            "carried_value_name") == "inter_stage_ssa"
        ignored = ("source_location", "carried_value_bytes") if representative else ()
        return {key: value for key, value in edge.items() if key not in ignored}

    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                compare(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            if path == "/stage_model/stage_dependencies":
                # Keep the full semantic edge multiset, including all loop-
                # carried metadata, independent of diagnostic emission order.
                a = [semantic_edge(edge) for edge in a]
                b = [semantic_edge(edge) for edge in b]
                a = sorted(a, key=lambda edge: json.dumps(edge, sort_keys=True))
                b = sorted(b, key=lambda edge: json.dumps(edge, sort_keys=True))
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + "/" + str(index))
        elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        elif a != b:
            differences.append((path, a, b))

    compare(actual, baseline, "")
    assert not differences, differences[:12]


@pytest.mark.parametrize("payload", _TYPES)
@pytest.mark.parametrize("operation", ("Load", "Store"))
def test_calibrated_target_keeps_auxiliary_operations(tmp_path, payload, operation):
    report = analyze(tmp_path, memory_module(operation, payload))
    for factor, price in zip((1, 2, 4), (_LOAD_PRICES if operation == "Load" else _STORE_PRICES)[payload]):
        route = _routes(report)[str(factor)]
        if operation == "Load":
            assert route["qualified_normal_full224_SIMT_target_Load_applied"]
            graph = json.loads(route["qualified_normal_full224_SIMT_target_Load_json"])
            assert graph["target_group_system_cycles"] == pytest.approx(price)
            assert graph["scheduled_target_system_cycles"] == pytest.approx(price * (4 // factor))
            assert graph["target_operation_count"] == 1
            assert graph["group_rounds"] == 4 // factor
            assert graph["all_other_operations_original_ownership"]
            assert not graph["kernel_setup_added_here"]
            mapping = graph["Op_pointer_derived_mapping"]
            targets = [x for x in mapping if x["owner"] == "target_Load_only"]
            assert len({x["original_operation_ordinal"] for x in mapping}) == len(mapping)
        else:
            assert route["qualified_normal_full224_SIMTStore_scoped_bundle_applied"]
            graph = json.loads(route["qualified_normal_full224_SIMTStore_scoped_graph_json"])
            assert graph["target_group_system_cycles"] == pytest.approx(price)
            assert graph["target_system_cycles"] == pytest.approx(price * (4 // factor))
            assert graph["runtime_repetitions"] == 4 // factor
            assert graph["kernel_setup_owners"] == 1
            assert graph["target_setup_owners"] == 0
            assert graph["covered_ops"] == 1
            assert graph["new_body_system_cycles"] == pytest.approx(graph["auxiliary_solver_body"] +
                                                                    graph["target_system_cycles"])
            mapping = graph["Op_pointer_derived_mapping"]
            targets = [x for x in mapping if x["owner"] == "target_Store_only"]
            assert len({x["ordinal"] for x in mapping}) == len(mapping)
        assert len(targets) == 1
        assert targets[0]["operation"] == "tt." + operation.lower()
        assert len(mapping) > len(targets)
        assert not graph["external_boundary_added_here"]


@pytest.mark.parametrize("operation", ("Load", "Store"))
@pytest.mark.parametrize("outside", ("pitch", "grid", "capability", "factor2"))
def test_uncalibrated_domains_keep_fallback(tmp_path, operation, outside):
    source = memory_module(operation, "u8")
    if outside == "pitch":
        source = source.replace("dense<32768>", "dense<65536>")
    report = analyze(tmp_path, source, programs=225 if outside == "grid" else 224, capability=outside != "capability",
                     factor2=outside != "factor2")
    routes = _routes(report)
    if outside == "factor2":
        assert "2" not in routes
        return
    key = ("qualified_normal_full224_SIMT_target_Load_applied"
           if operation == "Load" else "qualified_normal_full224_SIMTStore_scoped_bundle_applied")
    assert not any(r.get(key, False) for r in routes.values())
    if os.getenv("TRITON_ASCEND_COSTMODEL_NATIVE_DRIVER"):
        assert_same_report(
            report,
            analyze(tmp_path, source, programs=225 if outside == "grid" else 224, capability=outside != "capability",
                    baseline=True))


def test_store_rejects_unknown_nullary_pointer_producer(tmp_path):
    source = memory_module("Store", "u8")
    old = "%values_ptr = tt.addptr %values_base, %dense_offset : tensor<8x32x!tt.ptr<i8>>, tensor<8x32xi32>"
    assert old in source
    source = source.replace(old,
                            '%values_ptr = "builtin.unrealized_conversion_cast"() : () -> tensor<8x32x!tt.ptr<i8>>')
    report = analyze(tmp_path, source)
    assert not any(
        r.get("qualified_normal_full224_SIMTStore_scoped_bundle_applied", False) for r in _routes(report).values())
    if os.getenv("TRITON_ASCEND_COSTMODEL_NATIVE_DRIVER"):
        assert_same_report(report, analyze(tmp_path, source, baseline=True))


def single_program_module(source):
    """Use the calibrated one-program entry without SIMT PID scaffolding."""
    source = source.replace("%program_Index :", "%Index :").replace("%program_X :", "%X :")
    source = source.replace("%program_Output :", "%Output :")
    source = source.replace("%Index_pid :", "%Index :").replace("%Values_pid :", "%Values :")
    scaffolding = ("%pid", "%program_", "%rows_per_program", "%values_per_program", "%Index_pid", "%Values_pid")
    return "\n".join(line for line in source.splitlines() if not line.strip().startswith(scaffolding)) + "\n"


def simd_store_module(payload):
    source = memory_module("Store", payload)
    source = source.replace("arith.constant 256 : i32", "arith.constant 4096 : i32")
    source = source.replace("arith.constant 8 : i32", "arith.constant 16 : i32")
    source = source.replace("arith.constant 32 : i32", "arith.constant 256 : i32")
    source = source.replace("8x32", "16x256").replace("8x1", "16x1").replace("1x32", "1x256")
    source = source.replace("8xi32", "16xi32")
    source = source.replace("32xi32", "256xi32").replace("tensor<8x!", "tensor<16x!")
    source = source.replace("end = 8", "end = 16").replace("end = 32", "end = 256")
    source = source.replace("dense<32>", "dense<256>").replace("dense<32768>", "dense<256>")
    return single_program_module(source)


@pytest.mark.parametrize("payload", _TYPES)
@pytest.mark.parametrize("operation", ("Load", "Store"))
def test_simd_target_preserves_non_memory_work(tmp_path, payload, operation):
    source = single_program_module(memory_module("Load",
                                                 payload)) if operation == "Load" else simd_store_module(payload)
    route = analyze(tmp_path, source, programs=1)["stage_model"]["routes"]["all_simd"]
    if operation == "Load":
        assert route["qualified_normal_segments_load_bundle_applied"]
        assert route["qualified_normal_segments_load_target_system_cycles"] > 0
    else:
        assert route["qualified_normal_segments_bundle_applied"]
        assert route["qualified_normal_segments_target_system_cycles"] > 0
