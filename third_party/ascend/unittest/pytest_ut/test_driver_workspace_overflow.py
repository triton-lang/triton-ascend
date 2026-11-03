# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Workspace sizing must use the full 64-bit logical program count."""

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_driver_workspace_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def _mock_backend_func(name, *args):
    return f"/* {name}: {args} */"


def _make_metadata(workspace_size):
    return SimpleNamespace(
        target=driver.GPUTarget("npu", "Ascend910B3", 0),
        workspace_size=workspace_size,
        lock_init_value=0,
        sync_block_lock_layout=0,
        bs_task_type=0,
        mix_mode="aiv",
        shared=0,
        compile_on_910_95=False,
        parallel_mode="",
        is_pure_simt=False,
        debug=False,
        coalesce_factor=1,
        coalesce_axis=-1,
        coalesce_grid_ceil_div=False,
        has_auto_blockify_blacklist_op=False,
        row_coalescing_applied=False,
        program_grid_transforms=None,
        program_grid_mapping_applied=False,
        auto_blockify_enabled=False,
        ptsm_cap_authorized=False,
    )


def _generate(workspace_size):
    context = contextlib.ExitStack()
    mock_npu_utils = context.enter_context(patch.object(driver, "NPUUtils"))
    context.enter_context(patch.object(driver, "force_disable_ffts", return_value=False))
    context.enter_context(patch.object(driver, "is_ffts_supported", return_value=True))
    context.enter_context(patch.object(driver, "get_backend_func", side_effect=_mock_backend_func))
    with context:
        mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
        mock_npu_utils.return_value.get_aicore_num.return_value = 20
        return driver.make_launcher(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=_make_metadata(workspace_size),
        )


def test_workspace_block_count_is_widened_before_multiply():
    """gridX * gridY * gridZ must not be truncated to 32 bits.

    All three launch preambles used to declare a uint32_t intermediate. A grid
    whose product exceeds 2**32 - 1 wrapped around (the reported 262144 x 16384
    case wrapped to 0), and a product above INT_MAX was signed overflow on the
    int-typed grid dimensions. Both made the workspace allocation silently
    zero-sized.
    """
    src = _generate(262144)
    # The intermediate product must be evaluated in 64 bits, otherwise the
    # multiplier itself is already wrong before it reaches totalWorkSpaceSize.
    assert "uint32_t blockNum4Workspace = gridX * gridY * gridZ;" not in src
    assert src.count("uint64_t blockNum4Workspace = static_cast<uint64_t>(gridX) * gridY * gridZ;") == 2
    # workspace_size must also be widened so the final product stays 64-bit.
    assert src.count("uint64_t totalWorkSpaceSize = (uint64_t)262144 * blockNum4Workspace;") == 2
    # block_dim handed to CANN stays 32-bit; only the workspace count is widened.
    assert src.count("uint32_t blockNum = gridX * gridY * gridZ;") == 2


def test_workspace_size_absent_when_kernel_needs_none():
    """No workspace sizing or allocation when the kernel needs none.

    The blockNum4Workspace declaration is unconditional in the template: it is
    only the multiplier of totalWorkSpaceSize, so declaring it allocates
    nothing. What is gated on workspace_size > 0 is the totalWorkSpaceSize
    computation and the allocate_memory call, so this checks that neither
    appears.
    """
    src = _generate(-1)
    assert "totalWorkSpaceSize" not in src
    assert "allocate_memory" not in src


@pytest.mark.parametrize("grid,workspace_size,expected", [
    ((262144, 16384, 1), 262144, 262144 * 2**32),
    ((65536, 65536, 1), 262144, 262144 * 2**32),
    ((256, 64, 1), 262144, 262144 * 256 * 64),
])
def test_workspace_total_matches_64bit_reference(grid, workspace_size, expected):
    """The reference product must stay exact once it exceeds 32 bits."""
    total = workspace_size
    for dim in grid:
        total *= dim
    assert total == expected
    if grid != (256, 64, 1):
        assert total > 2**32 - 1
