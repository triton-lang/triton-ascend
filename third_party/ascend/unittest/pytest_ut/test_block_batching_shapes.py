# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
from triton._C.libtriton import ascend, ir
from block_batching_shape_cases import SHAPE_OPS, shape_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    path = tmp_path / "shapes.ttir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    assert module.verify()
    return module


@pytest.mark.parametrize("op", SHAPE_OPS)
@pytest.mark.parametrize("factor", [3, 6])
def test_shape_models_preserve_the_lane_axis(tmp_path, op, factor):
    module = parse(tmp_path, shape_ir(op))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["lane_width"] == factor
    assert module.verify()
    text = str(module)
    if op == "tt.trans":
        assert "order = array<i32: 0, 2, 1>" in text
    if op in ("tt.broadcast", "tt.expand_dims"):
        assert f"tensor<{factor}x2x4xf32>" in text
    if op == "tt.split":
        assert f"tensor<{factor}x2xf32>" in text and "tt.split" in text
    before = str(module)
    assert ascend.passes.ttir.rewrite_block_batching(module, factor, 0)["reason"] == "ALREADY_MAPPED"
    assert str(module) == before


@pytest.mark.parametrize("attribute,reason", [("allow_reorder", "CROSS_LANE_REORDER"),
                                              ("efficient_layout", "UNSUPPORTED_LAYOUT_HINT")])
def test_unproven_reshape_reordering_is_transactional(tmp_path, attribute, reason):
    text = shape_ir("tt.reshape").replace("tt.reshape %v :", f"tt.reshape %v {attribute} :")
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == reason
    assert report["operation"] == "tt.reshape"
    assert str(module) == before


@pytest.mark.parametrize("factor", [3, 6])
def test_non_splat_literal_is_preserved_at_batching_boundary(tmp_path, factor):
    # Device execution of this literal form is a separately recorded stock
    # backend limitation; the TTIR transform must still retain every value.
    module = parse(tmp_path, shape_ir("tt.reshape", literal_weights=True))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REWRITTEN", report
    text = str(module)
    assert "tensor<2x2xf32>" in text
    for value in ("1.000000e+00", "2.000000e+00", "3.000000e+00", "4.000000e+00"):
        assert value in text
    assert module.verify()
