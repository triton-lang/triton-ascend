# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
from triton._C.libtriton import ir, ascend
from block_batching_reduction_cases import REDUCTION_KINDS, reduction_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, text):
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend.ir.load_dialects(ctx)
    ascend.load_dialects(ctx)
    path = tmp_path / "reduction.ttir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), ctx)
    module.context = ctx
    assert module.verify()
    return module


@pytest.mark.parametrize("kind",
                         [kind for kind in REDUCTION_KINDS if kind not in ("maxnum_capture", "argmax", "argmax_index")])
@pytest.mark.parametrize("factor", [3, 6])
def test_segmented_reduce_and_scan_mapping(tmp_path, kind, factor):
    module = parse(tmp_path, reduction_ir(kind))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()
    text = str(module)
    assert "axis = 1 : i32" in text
    assert f"tensor<{factor}x4xf32>" in text
    if kind in ("sum", "max", "maxnum", "maxnum_capture", "argmax", "argmax_index"):
        assert f"tensor<{factor}xf32>" in text
    if kind == "reverse_scan":
        assert "reverse = true" in text


@pytest.mark.parametrize("uniform_inputs", [False, True])
def test_varying_scalar_cannot_be_captured_in_combiner(tmp_path, uniform_inputs):
    text = reduction_ir("sum")
    text = text.replace('%reduced = "tt.reduce"',
                        '%captured = arith.sitofp %p : i32 to f32\n    %reduced = "tt.reduce"')
    text = text.replace("tt.reduce.return %value : f32",
                        "%biased = arith.addf %value, %captured : f32\n        tt.reduce.return %biased : f32")
    if uniform_inputs:
        text = text.replace('"tt.reduce"(%v)', '"tt.reduce"(%onef)')
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "UNSUPPORTED_REGION_CAPTURE"
    assert str(module) == before


def test_ieee_elementwise_policy_does_not_admit_new_combiners(tmp_path):
    module = parse(
        tmp_path,
        reduction_ir("sum").replace("arith.addf %lhs_value, %rhs_value", "arith.divf %lhs_value, %rhs_value"))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "UNSUPPORTED_REDUCTION_COMBINER"
    assert str(module) == before


@pytest.mark.parametrize("factor", [3, 6])
def test_terminator_only_varying_capture_is_rejected(tmp_path, factor):
    text = reduction_ir("maxnum_capture").replace("tt.reduce.return %value, %n : f32, i32",
                                                  "tt.reduce.return %value, %p : f32, i32")
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REJECTED", report
    assert report["reason"] == "UNSUPPORTED_REDUCTION_RETURN_CAPTURE"
    assert str(module) == before


@pytest.mark.parametrize("kind,reason", [
    ("argmax", "UNSUPPORTED_NAN_REDUCTION_COMBINER"),
    ("argmax_index", "UNSUPPORTED_NAN_REDUCTION_COMBINER"),
    ("maxnum_capture", "UNSUPPORTED_REDUCTION_RETURN_CAPTURE"),
])
def test_removed_seeded_capabilities_reject_candidate_without_mutation(tmp_path, kind, reason):
    module = parse(tmp_path, reduction_ir(kind))
    before = str(module)
    disabled = ascend.passes.ttir.rewrite_block_batching(module, 1, 0)
    assert disabled["compile_status"] == "DISABLED"
    assert str(module) == before
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED", report
    assert report["reason"] == reason
    assert str(module) == before
