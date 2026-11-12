"""Long number-min/max reductions must retain the parallel reduction path."""
from pathlib import Path
import subprocess
import pytest
import triton
from triton._C.libtriton import ir, ascend

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("operation", ["maxnumf", "minnumf"])
@pytest.mark.parametrize("lanes", [1, 3])
def test_long_number_reduction_remains_linalg_reduce(tmp_path, operation, lanes):
    source = f'''module attributes {{hacc.target = #hacc.target<"Ascend950PR_9579">}} {{
      tt.func public @number_reduce(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {{
        %p = tt.get_program_id x : i32
        %b = arith.constant 4096 : i32
        %base = arith.muli %p, %b : i32
        %r = tt.make_range {{start = 0 : i32, end = 4096 : i32}} : tensor<4096xi32>
        %bv = tt.splat %base : i32 -> tensor<4096xi32>
        %idx = arith.addi %bv, %r : tensor<4096xi32>
        %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
        %ptrs = tt.addptr %xp, %idx : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
        %input = tt.load %ptrs : tensor<4096x!tt.ptr<f32>>
        %result = "tt.reduce"(%input) <{{axis = 0 : i32}}> ({{
        ^bb0(%lhs: f32, %rhs: f32):
          %value = arith.{operation} %lhs, %rhs : f32
          tt.reduce.return %value : f32
        }}) : (tensor<4096xf32>) -> f32
        %out = tt.addptr %y, %p : !tt.ptr<f32>, i32
        tt.store %out, %result : !tt.ptr<f32>
        tt.return
      }}
    }}'''
    path = tmp_path / "number-reduce.ttir"
    path.write_text(source)
    if lanes > 1:
        context = ir.context()
        ir.load_dialects(context)
        ascend.ir.load_dialects(context)
        ascend.load_dialects(context)
        module = ir.parse_mlir_module(str(path), context)
        module.context = context
        report = ascend.passes.ttir.rewrite_block_batching(module, lanes, 0)
        assert report["compile_status"] == "REWRITTEN", report
        path.write_text(str(module))
    tool = Path(triton.__file__).parent / "_C/triton-opt"
    # Use the production prefix, which prepares scalar combiners before the
    # ordinary ReduceConverter. Calling only triton-to-linalg bypasses it.
    pipeline = ("any(triton-control-flow-opt,"
                "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
                "discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},"
                "triton-to-annotation,triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true},"
                "triton-to-hivm,triton-to-hfusion,triton-to-llvm,"
                "bubble-up-operation{compile-on-910-95=true enable-aggressive-mode=true},"
                "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
                "triton-to-linalg{compile-mode=simd_simt_template compile-on-910-95=true named-ops=true})")
    lowered = subprocess.check_output([str(tool), str(path), "--pass-pipeline=" + pipeline, "--verify-each"], text=True)
    assert "linalg.reduce" in lowered
    assert "scf.for" not in lowered
