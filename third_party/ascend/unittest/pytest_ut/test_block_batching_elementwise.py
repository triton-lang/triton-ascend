# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Native IR checks for observable W1/W2 calculations and exact lane widths."""
import json
import re

import pytest
from triton.backends.ascend.block_batching import IMPLEMENTATION_VERSION
from triton._C.libtriton import ascend, ir

from block_batching_op_cases import ELEMENTWISE_OPS, W2_OPS, case_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    path = tmp_path / "elementwise.ttir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    assert module.verify()
    return module


@pytest.mark.parametrize("op", ELEMENTWISE_OPS)
@pytest.mark.parametrize("factor", [3, 6])
@pytest.mark.parametrize("variant", ["UV", "VV"])
def test_registered_elementwise_combinations(tmp_path, op, factor, variant):
    module = parse(tmp_path, case_ir(op, variant))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["implementation_version"] == IMPLEMENTATION_VERSION
    assert report["lane_width"] == factor
    assert report["operations"][op] >= 1
    assert module.verify()
    assert f"tensor<{factor}x4x" in str(module)
    assert json.loads(report["guard_json"])["accesses"]


@pytest.mark.parametrize("op", W2_OPS)
def test_dangerous_operands_are_selected_before_evaluation(tmp_path, op):
    module = parse(tmp_path, case_ir(op))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    text = str(module)
    definitions = {}
    computation = None
    for line in text.splitlines():
        match = re.search(r"(%[\w]+) = (.*)", line)
        if not match:
            continue
        name, body = match.groups()
        lowered = ("arith.divui", "arith.remui") if op == "arith.ceildivui" else (op, )
        if any(body.startswith(name + " ") or body.startswith('"' + name + '"') for name in lowered):
            computation = body
            # Every original input to the dangerous operation must now be a
            # value defined by a select, not a select applied to its result.
            operands = re.findall(r"%[\w]+", body.split(":", 1)[0])
            assert operands
            assert all(definitions[value].startswith("arith.select ") for value in operands)
            assert all("hacc.block_batching_safe_input" in definitions[value] for value in operands)
        definitions[name] = body
    assert computation is not None


def test_unknown_attribute_diagnoses_original_operation(tmp_path):
    text = case_ir("arith.maximumf").replace('"arith.maximumf"(%v, %denomf)',
                                             '"arith.maximumf"(%v, %denomf) {unmodeled_semantics = true}')
    module = parse(tmp_path, text)
    original = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == "UNSUPPORTED_OPERATION_ATTRIBUTE"
    assert report["operation"] == "arith.maximumf" and report["location"]
    assert report["context"] == "BlockBody"
    assert str(module) == original


def test_unsigned_ceildiv_avoids_overflowing_expansion(tmp_path):
    module = parse(tmp_path, case_ir("arith.ceildivui"))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    text = str(module)
    assert "arith.ceildivui" not in text
    assert "arith.divui" in text and "arith.remui" in text
    assert "arith.cmpi ne" in text and "arith.extui" in text
    assert module.verify()


@pytest.mark.parametrize("op, marker", [("arith.cmpf", " = true"), ("arith.maximumf", "")])
def test_safety_hint_rejects_wrong_type_or_operation(tmp_path, op, marker):
    text = case_ir(op)
    if op == "arith.cmpf":
        text = text.replace("%predicate, %v, %onef :",
                            "%predicate, %v, %onef {hacc.block_batching_safe_input" + marker + "} :")
    else:
        text = text.replace('"arith.maximumf"(%v, %denomf)',
                            '"arith.maximumf"(%v, %denomf) {hacc.block_batching_safe_input}')
    module = parse(tmp_path, text)
    original = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report["compile_status"] == "REJECTED", report
    assert report["reason"] == "UNSUPPORTED_OPERATION_ATTRIBUTE", report
    assert str(module) == original
