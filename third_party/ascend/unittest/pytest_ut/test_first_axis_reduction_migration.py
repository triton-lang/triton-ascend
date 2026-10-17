"""Exercise first-axis calibration through the native TTIR cost-model pass."""
import json
from pathlib import Path

import pytest
from triton._C.libtriton import ascend, ir


@pytest.mark.parametrize("warps", [1, 4])
@pytest.mark.parametrize("load_stride", [1, 2])
@pytest.mark.parametrize("axis,expected", [(0, "first"), (1, "tail")])
def test_axis_calibration_and_legacy_fallback(tmp_path, axis, expected, load_stride, warps):
    profiles = Path(__file__).resolve().parents[2] / "costmodel" / "profiles"
    source = profiles / "simd_simt" / "ascend_950pr_simd_simt_v1.json"
    selection = json.loads(source.read_text())
    selection["microbenchmark_profile"] = str((source.parent / selection["microbenchmark_profile"]).resolve())
    reduction = json.loads((source.parent / selection["reduction_profile"]).read_text())
    dependency = tmp_path / "reduction.json"
    dependency.write_text(json.dumps(reduction))
    selection["reduction_profile"] = str(dependency)
    profile = tmp_path / "selection.json"
    profile.write_text(json.dumps(selection))
    result_type = "tensor<64xf32>" if axis == 0 else "tensor<8xf32>"
    text = """module {
      tt.func public @sum(%ptr: !tt.ptr<f32>) -> RESULT {
        %rows = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
        %cols = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
        %rr = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
        %cc = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %r = tt.broadcast %rr : tensor<8x1xi32> -> tensor<8x64xi32>
        %c = tt.broadcast %cc : tensor<1x64xi32> -> tensor<8x64xi32>
        %stride = arith.constant dense<64> : tensor<8x64xi32>
        %base = arith.muli %r, %stride : tensor<8x64xi32>
        %offset = arith.addi %base, %c : tensor<8x64xi32>
        %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<8x64x!tt.ptr<f32>>
        %step = arith.constant dense<STEP> : tensor<8x64xi32>
        %strided = arith.muli %offset, %step : tensor<8x64xi32>
        %addresses = tt.addptr %ptrs, %strided : tensor<8x64x!tt.ptr<f32>>, tensor<8x64xi32>
        %x = tt.load %addresses : tensor<8x64x!tt.ptr<f32>>
        %out = "tt.reduce"(%x) <{axis = AXIS : i32}> ({
        ^bb0(%a: f32, %b: f32):
          %v = arith.addf %a, %b : f32
          tt.reduce.return %v : f32
        }) : (tensor<8x64xf32>) -> RESULT
        tt.return %out : RESULT
      }
    }""".replace("RESULT", result_type).replace("AXIS", str(axis)).replace("STEP", str(load_stride))
    module_path = tmp_path / "input.mlir"
    module_path.write_text(text)

    def analyze():
        context = ir.context()
        ir.load_dialects(context)
        ascend.load_dialects(context)
        module = ir.parse_mlir_module(str(module_path), context)
        manager = ir.pass_manager(context)
        ascend.passes.ttir.add_select_simd_simt_costmodel(manager, "report", str(profile), "Ascend950PR_9579", warps,
                                                          True)
        manager.run(module, "first_axis_migration_test")
        return json.loads(ascend.ir.get_string_attr(module, "ascend.simt_costmodel.report_json"))

    report = analyze()
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2))

    def flags(value, name):
        if isinstance(value, dict):
            return [value[name]] if name in value else [item for child in value.values() for item in flags(child, name)]
        if isinstance(value, list):
            return [item for child in value for item in flags(child, name)]
        return []

    assert any(flags(report, "calibrated_" + expected + "_reduction")), report
    if axis == 0 and warps == 4:
        simd = [
            candidate for stage in report["stage_model"]["logical_stages"] for candidate in stage["implementations"]
            if candidate["calibrated_first_reduction"] and candidate["implementation"]["mode"] == "simd"
        ]
        assert simd
        # Source sum/f32 segment at R=8,G=64: 67.8430493701 + 0.72716896804*8.
        assert simd[0]["resource_system_cycles"]["reduction_per_iteration"] == pytest.approx(73.66040111442)
    # Older tail-only profiles remain usable and cannot activate first-axis costs.
    reduction["families"].pop("first_axis")
    dependency.write_text(json.dumps(reduction))
    fallback = analyze()
    assert not any(flags(fallback, "calibrated_first_reduction"))
    if axis == 1:
        assert flags(fallback, "calibrated_tail_reduction") == flags(report, "calibrated_tail_reduction")
