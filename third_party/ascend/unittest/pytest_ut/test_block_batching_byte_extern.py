# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import re
import pytest
from triton.backends.ascend.block_batching import IMPLEMENTATION_VERSION
from triton._C.libtriton import ir, ascend
from block_batching_byte_extern_cases import case_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    path = tmp_path / "byte-extern.ttir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    assert module.verify()
    return module


@pytest.mark.parametrize("kind", ["bool", "byte", "pow", "tanh"])
@pytest.mark.parametrize("factor", [3, 6])
def test_new_subsets_keep_lane_mapping_and_memory_contract(tmp_path, kind, factor):
    module = parse(tmp_path, case_ir(kind))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["implementation_version"] == IMPLEMENTATION_VERSION
    assert module.verify()
    text = str(module)
    assert f"tensor<{factor}x8xf32>" in text
    if kind in ("bool", "byte"):
        assert 1 in {access["bytes"] for access in json.loads(report["guard_json"])["accesses"]}
        if kind == "bool": assert "was_bool_to_int8 = true" in text
    else:
        assert "tt.extern_elementwise" in text
        # tanh lowers to total floating arithmetic; pow retains its
        # conservative input policy because its implementation casts values.
        assert ("hacc.block_batching_safe_input" in text) == (kind == "pow")
        assert ("__hmf_pow_fp32" if kind == "pow" else "__hmf_tanhf") in text


@pytest.mark.parametrize(
    "kwargs", [dict(symbol="__hmf_unknown"),
               dict(symbol="__hmf_tanhf"),
               dict(pure=False),
               dict(libpath="custom.bc")])
def test_unknown_external_contracts_are_rejected_without_changes(tmp_path, kwargs):
    module = parse(tmp_path, case_ir("pow", **kwargs))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["reason"] == "UNSUPPORTED_EXTERNAL_FUNCTION", report
    assert str(module) == before


@pytest.mark.parametrize("marker", ["", " {was_bool_to_int8 = false}"])
def test_bool_byte_layout_requires_exact_normalization_marker(tmp_path, marker):
    text = case_ir().replace(" {was_bool_to_int8 = true}", marker)
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "UNSUPPORTED_BYTE_OPERATION"
    assert str(module) == before


def test_i8_arithmetic_is_not_opened_by_byte_load_support(tmp_path):
    text = case_ir("byte").replace("%condition = arith.cmpi",
                                   "%sum8 = arith.addi %bytes, %bytezero : tensor<8xi8>\n    %condition = arith.cmpi")
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["reason"] == "UNSUPPORTED_BYTE_OPERATION", report


def test_external_f16_remains_outside_the_signature_subset(tmp_path):
    module = parse(tmp_path, case_ir("pow").replace("f32", "f16"))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["reason"] == "UNSUPPORTED_EXTERNAL_FUNCTION", report


def test_read_address_division_preserves_device_ssa(tmp_path):
    text = case_ir("tanh").replace(
        "%xp = tt.addptr %xs, %idx",
        "%quotient = arith.divsi %idx, %ns : tensor<8xi32>\n    %xp = tt.addptr %xs, %quotient")
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()


def test_pow_sanitizes_exponent_without_selecting_its_base(tmp_path):
    module = parse(tmp_path, case_ir("pow"))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report["compile_status"] == "REWRITTEN", report
    text = str(module)
    base, exponent = re.search(r'tt.extern_elementwise (%\w+), (%\w+)', text).groups()
    definitions = dict(re.findall(r'(%\w+) = ([^\n]+)', text))
    assert "hacc.block_batching_safe_input" not in definitions[base]
    assert "hacc.block_batching_safe_input" in definitions[exponent]
    assert module.verify()


@pytest.mark.parametrize("value,sanitize", [
    ("2.0", False),
    ("-3.0", False),
    ("0.0", False),
    ("0.5", False),
    ("2147483520.0", False),
    ("-2147483520.0", False),
    ("2147483648.0", True),
    ("-2147483648.0", True),
    ("0x7F800000", True),
    ("0xFF800000", True),
    ("0x7FC00000", True),
])
def test_pow_preserves_proven_safe_constant_exponent(tmp_path, value, sanitize):
    # The pow implementation casts abs(exponent) to signed i32. Safe literal
    # exponents need no inactive-lane select; unknown/unsafe inputs still do.
    text = case_ir("pow").replace(
        "%result = tt.extern_elementwise %xv, %yv", f"%exponent = arith.constant dense<{value}> : tensor<8xf32>\n"
        "    %result = tt.extern_elementwise %xv, %exponent")
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()
    rewritten = str(module)
    assert ("hacc.block_batching_safe_input" in rewritten) == sanitize
    # Preserve the external function and its math semantics.
    assert 'symbol = "__hmf_pow_fp32"' in rewritten
