# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only UT for the integer-row program-grid protocol extension."""
from pathlib import Path
import runpy

import pytest

pytestmark = pytest.mark.backend("none")


def test_integer_row_grid_contract():
    # Load the checkout's pure-Python protocol, without importing an installed
    # Triton build or initializing an accelerator runtime.
    protocol = runpy.run_path(str(Path(__file__).resolve().parents[2] / "backend/program_grid.py"))
    normalize = protocol["normalize_program_grid_transforms"]
    apply = protocol["apply_program_grid_transforms"]
    error = protocol["ProgramGridContractError"]
    contract = dict(
        version=2,
        extent_source="runtime_original_grid",
        hidden_extent_axes=[0, 1],
        hidden_argument_order=["originalGridX", "originalGridY"],
        hidden_argument_types=["i32", "i32"],
        transforms=[
            dict(order=0, kind="ceil_div", axis=1, factor=64, persistent_coverage=False, grid_stride_abi_verified=False)
        ],
    )
    assert normalize(contract) == contract
    assert normalize(protocol["canonical_program_grid_transforms_json"](contract)) == contract
    # Preserve X/Z and round Y up, including just below/on/above a full batch.
    for extent, expected in ((1, 1), (63, 1), (64, 1), (65, 2)):
        assert apply((3, extent, 2), contract) == (3, expected, 2)
    assert protocol["get_persistent_transform"](contract) is None
    # The extension does not authorize Y/32, X/64 without persistent coverage,
    # or a persistent Y/64 launch.
    transform = contract["transforms"][0]
    for change in (dict(factor=32), dict(axis=0), dict(persistent_coverage=True, grid_stride_abi_verified=True)):
        with pytest.raises(error):
            normalize({**contract, "transforms": [{**transform, **change}]})
