# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host validation of the fixed op catalog, with optional native read-back."""
from collections import Counter
from pathlib import Path
import re

import pytest
from triton.backends.ascend.block_batching import IMPLEMENTATION_VERSION

from block_batching_op_cases import W1_OPS, W2_OPS

pytestmark = pytest.mark.backend("native")
CATALOG = Path(__file__).parents[2] / "include/TritonToGraph/BlockBatching/BatchOps.def"


def catalog():
    expression = r'BATCH_OP\("([^"\n]+)", ([BERS]), (\w+), (\w+), (\w+), (true|false)\)'
    rows = re.findall(expression, CATALOG.read_text())
    return {
        name:
        dict(spec_level=level, model=model, attribute_policy=attrs, inactive_input_policy=inactive,
             implementation="supported" if enabled == "true" else "rejected" if level == "R" else "deferred")
        for name, level, model, attrs, inactive, enabled in rows
    }, rows


def test_fixed_spec_catalog_is_complete_and_unique():
    entries, rows = catalog()
    assert len(entries) == len(rows) == 185
    assert Counter(row[1] for row in rows) == {"B": 87, "E": 66, "R": 20, "S": 12}
    assert sum(r[1] == "B" and r[-1] == "true" for r in rows) == 87
    assert all(r[-1] == "false" for r in rows if r[1] in ("E", "R") and r[0] not in ("tt.extern_elementwise", "tt.dot"))
    assert entries["tt.extern_elementwise"]["spec_level"] == "E"
    assert entries["tt.extern_elementwise"]["attribute_policy"] == "External"


def test_new_model_families_have_observable_operation_cases():
    entries, _ = catalog()
    assert len(W1_OPS) == 25 and len(W2_OPS) == 13
    assert len(set(W1_OPS + W2_OPS)) == 38
    for name in W1_OPS + W2_OPS:
        assert entries[name]["spec_level"] == "B"
        assert entries[name]["implementation"] == "supported"
        assert entries[name]["model"] == "Elementwise"


def test_dangerous_models_require_pre_evaluation_policy():
    entries, _ = catalog()
    for name in W2_OPS:
        assert entries[name]["inactive_input_policy"] != "Preserve"
    for name in ("math.exp", "math.exp2", "math.log", "math.log2", "math.sqrt", "math.rsqrt", "arith.divf",
                 "arith.remf", "tt.precise_divf", "tt.precise_sqrt"):
        assert entries[name]["inactive_input_policy"] == "IEEE"


def test_basic_region_families_have_conditional_models():
    entries, _ = catalog()
    for name in ("tt.scan", "scf.if", "scf.for"):
        assert entries[name]["spec_level"] == "B"
        assert entries[name]["implementation"] == "supported"
    for name in ("scf.while", "scf.parallel"):
        assert entries[name]["implementation"] == "rejected"


def test_native_manifest_matches_admission_catalog():
    extension = pytest.importorskip("triton._C.libtriton")
    from triton.backends.ascend.block_batching import IMPLEMENTATION_VERSION
    binding = extension.ascend.passes.ttir
    assert hasattr(binding, "block_batching_registry"), "rebuild the Ascend extension"
    native = binding.block_batching_registry()
    entries, _ = catalog()
    assert len(native) == len(entries)
    for row in native:
        expected = entries[row["op_name"]]
        assert all(row[key] == value for key, value in expected.items())
        assert row["implementation_version"] == IMPLEMENTATION_VERSION
