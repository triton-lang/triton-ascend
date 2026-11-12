# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton.backends.ascend.block_batching import IMPLEMENTATION_VERSION
from triton._C.libtriton import ir, ascend
from block_batching_memory_cases import KINDS, memory_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, text):
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend.ir.load_dialects(ctx)
    ascend.load_dialects(ctx)
    path = tmp_path / "memory.ttir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), ctx)
    module.context = ctx
    assert module.verify()
    return module


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("factor", [3, 6])
def test_byte_provenance_and_explicit_scalar_index(tmp_path, kind, factor):
    module = parse(tmp_path, memory_ir(kind))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["implementation_version"] == IMPLEMENTATION_VERSION
    assert module.verify()
    guard = json.loads(report["guard_json"])
    assert guard["version"] == 3
    if kind == "nd_reused_range":
        assert guard["accesses"][0]["shape"] == [4, 4]
        serialized = json.dumps(guard["accesses"][0]["offset"])
        assert '["coord", 0,' in serialized and '["coord", 1,' in serialized
    if kind in ("ptr_bitcast", "integer64"):
        assert {access["bytes"] for access in guard["accesses"]} == ({8} if kind == "integer64" else {4})
    if kind == "ptr_roundtrip":
        assert guard["accesses"][0]["root"] == 0
        assert guard["accesses"][0]["pointer_checks"]


@pytest.mark.parametrize("kind", ["index_cast", "index_castui"])
def test_missing_index_layout_selects_baseline_transactionally(tmp_path, kind):
    module = parse(tmp_path, memory_ir(kind, layout=False))
    original = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "INDEX_LAYOUT_NOT_PROVEN"
    assert str(module) == original


def test_varying_index_cannot_become_invalid_triton_index_tensor(tmp_path):
    text = memory_ir("index_cast").replace("arith.index_cast %k :", "arith.index_cast %p :", 1)
    module = parse(tmp_path, text)
    original = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "VARYING_INDEX_NOT_IMPLEMENTED"
    assert str(module) == original


def test_integer_address_without_pointer_provenance_is_rejected(tmp_path):
    text = memory_ir("ptr_roundtrip").replace("%address = arith.addi %bits, %bytes : i64",
                                              "%address = arith.addi %bytes, %bytes : i64")
    module = parse(tmp_path, text)
    original = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "UNKNOWN_POINTER_ROOT"
    assert str(module) == original


def test_pointer_roundtrip_uses_addptr_through_ta_adapter(tmp_path):
    from pathlib import Path
    import subprocess
    import triton
    module = parse(tmp_path, memory_ir("ptr_roundtrip"))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    path = tmp_path / "inttoptr-batched.ttir"
    path.write_text(str(module))
    tool = Path(triton.__file__).parent / "_C/triton-opt"
    pipeline = ("any(triton-control-flow-opt,"
                "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
                "discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},"
                "triton-to-annotation,"
                "triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true})")
    text = subprocess.check_output([str(tool), str(path), "--pass-pipeline=" + pipeline, "--verify-each"], text=True)
    assert "tt.int_to_ptr" not in text
    assert "tt.addptr" in text or "indirect" in text


def test_mixed_width_pointer_view_retains_byte_provenance_in_ir(tmp_path):
    # This form remains a stock converter limitation at device compilation.
    module = parse(tmp_path, memory_ir("ptr_bitcast", mixed_width=True))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert {access["bytes"] for access in json.loads(report["guard_json"])["accesses"]} == {8}
    assert module.verify()


def test_non_element_byte_offset_uses_complete_baseline(tmp_path):
    text = memory_ir("ptr_roundtrip").replace("%bytes = arith.muli %wide, %four : i64",
                                              "%bytes = arith.addi %wide, %four : i64")
    module = parse(tmp_path, text)
    original = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "POINTER_ROUNDTRIP_NOT_LOWERABLE"
    assert str(module) == original


def test_masked_scalar_like_load_keeps_a_guarded_memory_operation(tmp_path):
    from pathlib import Path
    import subprocess
    import triton
    module = parse(tmp_path, memory_ir("masked_scalar_like"))
    path = tmp_path / "masked-scalar-like.ttir"
    path.write_text(str(module))
    tool = Path(triton.__file__).parent / "_C/triton-opt"
    text = subprocess.check_output([
        str(tool),
        str(path), "--pass-pipeline=any(triton-control-flow-opt,"
        "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
        "discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},"
        "triton-to-annotation,"
        "triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true})", "--verify-each"
    ], text=True)
    # The scalar read belongs to a loop whose dynamic trip count is derived
    # from the mask, so an all-false mask executes no memory operation.
    assert text.index("scf.for") < text.index("tt.load") < text.index("scf.yield")
    assert "arith.index_castui" in text
    assert "arith.minsi" in text


def test_tensor_inttoptr_offset_analysis_does_not_extract_a_scalar(tmp_path):
    from pathlib import Path
    import subprocess
    import triton
    text = """module {
  tt.func public @tensor_roundtrip(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
    %bits = tt.ptr_to_int %x : !tt.ptr<f32> -> i64
    %base = tt.splat %bits : i64 -> tensor<3xi64>
    %range = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %wide = arith.extsi %range : tensor<3xi32> to tensor<3xi64>
    %four = arith.constant dense<4> : tensor<3xi64>
    %delta = arith.muli %wide, %four : tensor<3xi64>
    %addresses = arith.addi %base, %delta : tensor<3xi64>
    %ptrs = tt.int_to_ptr %addresses : tensor<3xi64> -> tensor<3x!tt.ptr<f32>>
    %value = tt.load %ptrs : tensor<3x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
    %out = tt.addptr %yp, %range : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
    tt.store %out, %value : tensor<3x!tt.ptr<f32>>
    tt.return
  }
}"""
    module = parse(tmp_path, text)
    path = tmp_path / "tensor-address-prefix.ttir"
    path.write_text(str(module))
    tool = Path(triton.__file__).parent / "_C/triton-opt"
    text = subprocess.check_output([
        str(tool),
        str(path), "--pass-pipeline=any(triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true})",
        "--verify-each"
    ], text=True)
    assert "tt.int_to_ptr" in text
    assert "tensor.extract" in text or "tt.indirect_load" in text
