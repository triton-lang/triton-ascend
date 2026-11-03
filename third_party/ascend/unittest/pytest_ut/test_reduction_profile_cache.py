"""Profile dependency invalidation through the native pass, without an NPU run."""

import json
import os
from pathlib import Path

import pytest
from triton._C.libtriton import ascend, ir


def test_reduction_profile_changes_in_one_process(tmp_path):
    profiles = Path(__file__).resolve().parents[2] / "costmodel" / "profiles"
    source = profiles / "simd_simt" / "ascend_950pr_simd_simt_v1.json"
    selection = json.loads(source.read_text())
    reduction = json.loads((source.parent / selection["reduction_profile"]).read_text())
    for key in ("microbenchmark_profile", "indirect_memory_profile", "prefixscan_profile"):
        if key in selection:
            selection[key] = str((source.parent / selection[key]).resolve())
    dependency = tmp_path / "reduction.json"
    dependency.write_text(json.dumps(reduction))
    selection["reduction_profile"] = str(dependency)
    profile = tmp_path / "selection.json"
    profile.write_text(json.dumps(selection))
    module_path = tmp_path / "input.mlir"
    module_path.write_text("""module {
      tt.func public @sum(%x: tensor<64xf32>) -> f32 {
        %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
        ^bb0(%a: f32, %b: f32):
          %v = arith.addf %a, %b : f32
          tt.reduce.return %v : f32
        }) : (tensor<64xf32>) -> f32
        tt.return %r : f32
      }
    }""")

    def analyze():
        context = ir.context()
        ir.load_dialects(context)
        ascend.load_dialects(context)
        module = ir.parse_mlir_module(str(module_path), context)
        manager = ir.pass_manager(context)
        ascend.passes.ttir.add_select_simd_simt_costmodel(manager, "report", str(profile), "Ascend950PR_9579", 4, True)
        manager.run(module, "reduction_profile_cache_test")
        report = json.loads(ascend.ir.get_string_attr(module, "ascend.simt_costmodel.report_json"))
        return report["profile_content_sha256"]

    first = analyze()
    assert analyze() == first
    original_stamp = dependency.stat()
    reduction["families"]["tail_axis"]["simd"]["r1_standard_sum_f32"][1] += 10
    dependency.write_text(json.dumps(reduction))
    os.utime(dependency, ns=(original_stamp.st_atime_ns, original_stamp.st_mtime_ns + 1_000_000_000))
    changed = analyze()
    assert changed != first
    assert analyze() == changed
    dependency.unlink()
    with pytest.raises(RuntimeError):
        analyze()
    dependency.write_text(json.dumps(reduction))
    assert analyze() == changed
