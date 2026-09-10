# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pure-Python acceptance tests for the versioned program-grid ABI."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def program_grid():
    path = Path(__file__).resolve().parents[2] / "backend" / "program_grid.py"
    module_name = "program_grid_contract_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _transform(order, axis, factor, logical_extent, *, persistent=False):
    return {
        "order": order,
        "kind": "ceil_div",
        "axis": axis,
        "factor": factor,
        "logical_extent": logical_extent,
        "persistent_coverage": persistent,
        "grid_stride_abi_verified": persistent,
    }


def _contract(*transforms, version=1):
    return {"version": version, "transforms": list(transforms)}


@pytest.mark.parametrize(
    ("logical_extent", "expected_programs"),
    [(1, 1), (7, 1), (8, 1), (9, 2), (64, 8), (65, 9)],
)
def test_iat_factor_eight_covers_boundary_head_extents_once(
    program_grid,
    logical_extent,
    expected_programs,
):
    """Exercise H=1/F-1/F/F+1/64/65 for the IAT tail-mask ABI."""
    factor = 8
    contract = _contract(_transform(0, 1, factor, logical_extent))
    assert program_grid.apply_program_grid_transforms(
        (3, logical_extent, 1),
        contract,
    ) == (3, expected_programs, 1)

    covered = [
        program * factor + lane
        for program in range(expected_programs)
        for lane in range(factor)
        if program * factor + lane < logical_extent
    ]
    assert covered == list(range(logical_extent))


def test_composable_transforms_support_same_and_different_axes(program_grid):
    same_axis = _contract(
        _transform(0, 0, 2, 65),
        _transform(1, 0, 3, 65),
    )
    assert program_grid.apply_program_grid_transforms((65, 1, 1), same_axis) == (11, 1, 1)

    different_axes = _contract(
        _transform(0, 0, 4, 17),
        _transform(1, 1, 3, 10),
    )
    assert program_grid.apply_program_grid_transforms((17, 10, 1), different_axes) == (5, 4, 1)


@pytest.mark.parametrize(
    "contract",
    [
        _contract(_transform(0, 0, 2, 8), version=2),
        _contract(_transform(0, 3, 2, 8)),
        _contract(_transform(0, 0, 1, 8)),
    ],
)
def test_contract_rejects_unknown_version_axis_and_factor(program_grid, contract):
    with pytest.raises(program_grid.ProgramGridContractError):
        program_grid.normalize_program_grid_transforms(contract)


def test_contract_rejects_unknown_or_mismatched_original_extent(program_grid):
    contract = _contract(_transform(0, 0, 4, 33))
    with pytest.raises(program_grid.ProgramGridContractError, match="logical_extent"):
        program_grid.apply_program_grid_transforms((32, 1, 1), contract)

    dynamic_extent = _contract(_transform(0, 0, 4, None))
    with pytest.raises(program_grid.ProgramGridContractError, match="logical_extent"):
        program_grid.normalize_program_grid_transforms(dynamic_extent)


def test_program_mapping_specialization_is_canonical_and_cache_serializable(program_grid):
    rule_mask = program_grid.INDEPENDENT_AXIS_TENSORIZE_RULE_BIT
    specialization = program_grid.make_program_grid_specialization((8, 65), rule_mask)

    assert specialization == {
        "version": 1,
        "grid": [8, 65, 1],
        "rule_mask": rule_mask,
    }
    assert program_grid.program_grid_specialization_enabled(rule_mask) is True
    assert program_grid.program_grid_specialization_enabled(0) is False
    assert program_grid.canonical_program_grid_specialization_json(specialization) == (
        '{"grid":[8,65,1],"rule_mask":512,"version":1}')


def test_runtime_scalar_specialization_is_canonical_and_cache_serializable(program_grid):
    specialization = program_grid.make_program_mapping_scalar_specialization([(2, 64), (10, -4)], )

    assert specialization == {
        "version": 1,
        "arguments": [
            {"index": 2, "value": 64},
            {"index": 10, "value": -4},
        ],
    }
    assert program_grid.canonical_program_mapping_scalar_specialization_json(
        specialization, ) == ('{"arguments":[{"index":2,"value":64},{"index":10,"value":-4}],'
                              '"version":1}')


def test_named_runtime_scalar_specialization_survives_frontend_argument_pruning(program_grid):
    specialization = program_grid.make_program_mapping_scalar_specialization([(4, "stride_out_token", 12288),
                                                                              (6, "num_splits", 2)], )

    assert specialization == {
        "version":
        2,
        "arguments": [
            {"index": 4, "name": "stride_out_token", "value": 12288},
            {"index": 6, "name": "num_splits", "value": 2},
        ],
    }
    assert program_grid.canonical_program_mapping_scalar_specialization_json(
        specialization, ) == ('{"arguments":[{"index":4,"name":"stride_out_token","value":12288},'
                              '{"index":6,"name":"num_splits","value":2}],"version":2}')


@pytest.mark.parametrize(
    "specialization",
    [
        {"version": 3, "arguments": [{"index": 2, "value": 64}]},
        {"version": 1, "arguments": []},
        {"version": 1, "arguments": [{"index": -1, "value": 64}]},
        {"version": 1, "arguments": [{"index": 2, "value": True}]},
        {
            "version": 1,
            "arguments": [
                {"index": 2, "value": 64},
                {"index": 2, "value": 65},
            ],
        },
        {
            "version": 2,
            "arguments": [{"index": 2, "name": "", "value": 64}],
        },
        {
            "version": 2,
            "arguments": [{"index": 2, "name": "stride", "value": 64}, {"index": 3, "name": "stride", "value": 65}],
        },
    ],
)
def test_runtime_scalar_specialization_rejects_invalid_contracts(
    program_grid,
    specialization,
):
    with pytest.raises(program_grid.ProgramGridContractError):
        program_grid.normalize_program_mapping_scalar_specialization(specialization)


def test_program_mapping_specialization_rejects_unknown_bits_and_noncanonical_grid(program_grid):
    with pytest.raises(program_grid.ProgramGridContractError, match="unsupported bits"):
        program_grid.normalize_program_mapping_rule_mask(1)
    with pytest.raises(program_grid.ProgramGridContractError, match="one to three"):
        program_grid.canonicalize_program_grid(())
    with pytest.raises(program_grid.ProgramGridContractError, match="positive integer"):
        program_grid.canonicalize_program_grid((1, True))
    with pytest.raises(program_grid.ProgramGridContractError, match="exactly three"):
        program_grid.normalize_program_grid_specialization({
            "version": 1,
            "grid": [8, 65],
            "rule_mask": 512,
        })


def test_callable_grid_is_resolved_twice_before_cache_and_must_be_reproducible(program_grid):
    calls = []

    def stable_grid(bound):
        calls.append(bound["extent"])
        return (bound["extent"], 16)

    assert program_grid.resolve_program_grid_for_specialization(stable_grid, {"extent": 65}) == (65, 16, 1)
    assert calls == [65, 65]

    values = iter(((8, 1, 1), (9, 1, 1)))
    with pytest.raises(program_grid.ProgramGridContractError, match="not reproducible"):
        program_grid.resolve_program_grid_for_specialization(lambda _bound: next(values), {})


@pytest.mark.parametrize("logical_tiles", (1, 7, 8, 9, 19))
def test_persistent_coverage_caps_only_verified_grid_stride_and_covers_every_tile_once(
    program_grid,
    logical_tiles,
):
    physical_cores = 8
    contract = _contract(_transform(0, 0, 2, logical_tiles * 2, persistent=True))
    actual_programs = program_grid.apply_program_grid_transforms(
        (logical_tiles * 2, 1, 1),
        contract,
        physical_core_count=physical_cores,
    )[0]
    assert actual_programs == min(logical_tiles, physical_cores)

    # The verified kernel ABI must stride by actual tt.num_programs(0), not by
    # the original launch grid.  That schedule visits every logical tile once.
    covered = [
        tile for program_id in range(actual_programs) for tile in range(program_id, logical_tiles, actual_programs)
    ]
    assert sorted(covered) == list(range(logical_tiles))
    assert len(covered) == len(set(covered))


def test_persistent_coverage_composes_with_a_nonpersistent_head_mapping(program_grid, ):
    """Only the verified token transform may cap its own launch axis."""
    contract = _contract(
        _transform(0, 1, 4, 16),
        _transform(1, 0, 2, 19, persistent=True),
    )

    # IAT first compresses 16 heads to four.  A core budget of eight then
    # gives the token grid two physical programs per head; the loop stride is
    # therefore two and every one of ceil_div(19, 2) logical token tiles is
    # owned exactly once for each mapped head.
    assert program_grid.apply_program_grid_transforms(
        (19, 16, 1),
        contract,
        physical_core_count=8,
    ) == (2, 4, 1)
    logical_tiles = (19 + 2 - 1) // 2
    covered = [tile for program_id in range(2) for tile in range(program_id, logical_tiles, 2)]
    assert sorted(covered) == list(range(logical_tiles))
    assert len(covered) == len(set(covered))


def test_nonpersistent_transform_never_caps_to_physical_parallelism(program_grid):
    contract = _contract(_transform(0, 0, 4, 68))
    assert program_grid.apply_program_grid_transforms(
        (68, 1, 1),
        contract,
        physical_core_count=8,
    ) == (17, 1, 1)


def test_contract_cache_identity_changes_with_transform_contents(program_grid):
    first = _contract(_transform(0, 0, 2, 32))
    second = _contract(_transform(0, 0, 4, 32))
    assert program_grid.canonical_program_grid_transforms_json(first) != \
        program_grid.canonical_program_grid_transforms_json(second)
