# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for the v3 schema and its independent v2 compatibility."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("_batch_program_grid",
                                              Path(__file__).parents[2] / "backend/program_grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)
pytestmark = pytest.mark.backend("native")


def test_native_v3_grid_contract(tmp_path):
    compiler = os.environ.get("CXX", "c++")
    if shutil.which(compiler) is None:
        pytest.skip("native launch-contract regression requires a C++ compiler")
    executable = tmp_path / "block_batching_grid_test"
    source = Path(__file__).with_name("block_batching_grid_test.cpp")
    headers = Path(__file__).parents[2] / "backend/launcher_src"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-I" + str(headers), "-o",
         str(executable)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "V1/V2 grid" in result.stdout


def contract(axis=0, factor=3, version=3):
    axes = [0, 1, 2] if version == 3 else [0, 1]
    return {
        "version":
        version, "extent_source":
        "runtime_original_grid", "hidden_extent_axes":
        axes, "hidden_argument_order":
        ["originalGrid" + a for a in "XYZ"[:len(axes)]], "hidden_argument_types": ["i32"] * len(axes), "transforms": [{
            "order":
            0, "kind":
            "ceil_div", "axis":
            axis, "factor":
            factor, "persistent_coverage":
            False, "grid_stride_abi_verified":
            False
        }]
    }


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("factor", [2, 3, 4, 6, 16])
def test_v3_tail_and_empty(axis, factor):
    schema = contract(axis, factor)
    assert grid.normalize_program_grid_transforms(json.dumps(schema)) == schema
    for original in [(10, 13, 7), (0, 13, 7), (10, 0, 7), (10, 13, 0)]:
        expected = list(original)
        expected[axis] = (original[axis] + factor - 1) // factor
        assert grid.apply_program_grid_transforms(original, schema) == tuple(expected)


@pytest.mark.parametrize("field,value", [
    ("version", True),
    ("version", 4),
    ("hidden_extent_axes", [False, 1, 2]),
    ("hidden_extent_axes", [0.0, 1, 2]),
    ("hidden_extent_axes", None),
    ("hidden_extent_axes", [0, 1]),
    ("hidden_argument_types", ["i32", "i32"]),
    ("hidden_argument_order", ["originalGridY", "originalGridX", "originalGridZ"]),
    ("transforms", []),
])
def test_v3_bad_schema(field, value):
    schema = contract()
    schema[field] = value
    with pytest.raises(grid.ProgramGridContractError):
        grid.normalize_program_grid_transforms(schema)


@pytest.mark.parametrize("field,value", [
    ("factor", True),
    ("factor", 1),
    ("factor", 1 << 31),
    ("axis", 3),
    ("axis", False),
    ("order", 1),
    ("persistent_coverage", True),
    ("grid_stride_abi_verified", True),
])
def test_v3_bad_transform(field, value):
    schema = contract()
    schema["transforms"][0][field] = value
    with pytest.raises(grid.ProgramGridContractError):
        grid.normalize_program_grid_transforms(schema)


def test_v3_repeated_transform_and_i32_limits():
    schema = contract()
    schema["transforms"].append(copy.deepcopy(schema["transforms"][0]))
    with pytest.raises(grid.ProgramGridContractError):
        grid.normalize_program_grid_transforms(schema)
    for original in [(-1, 1, 1), (1 << 31, 1, 1), (True, 1, 1), (65536, 65536, 2)]:
        with pytest.raises(grid.ProgramGridContractError):
            grid.apply_program_grid_transforms(original, contract())
    assert grid.apply_program_grid_transforms((2147483647, 1, 1), contract()) == (715827883, 1, 1)


def test_v2_independent_whitelist():
    assert grid.apply_program_grid_transforms((10, 7, 1), contract(0, 4, 2)) == (3, 7, 1)
    for axis, factor in [(0, 3), (1, 3), (2, 4)]:
        with pytest.raises(grid.ProgramGridContractError):
            grid.normalize_program_grid_transforms(contract(axis, factor, 2))
    with pytest.raises(grid.ProgramGridContractError):
        grid.apply_program_grid_transforms((0, 1, 1), contract(0, 4, 2))
