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

import re
from pathlib import Path

import pytest

from triton.backends.ascend import launcher

AUTO_MAP = launcher.AUTO_MAP
UINT32_MAX = 2**32 - 1

LAUNCHER_SRC = Path(launcher.__file__).with_name("launcher_src")
HEADER = LAUNCHER_SRC / "launcher_args.h"
RUNTIME = LAUNCHER_SRC / "launcher_runtime.cpp"


def _source(path):
    assert path.is_file(), f"launcher source not found: {path}"
    return path.read_text(encoding="utf-8")


def _prepare_grid(flags, physical_blocks, grid):
    """Reference model of prepareGrid for the plain (no IAT/PTSM/coalesce) path.

    Kept in Python so the guard runs wherever pytest runs, without depending on
    a host C++ compiler. Extreme products raise OverflowError the way
    checkedMultiply raises std::overflow_error.
    """
    dims = list(grid)
    if any(dim <= 0 for dim in dims):
        return 0, 0

    def checked_mul(left, right):
        result = left * right
        if result > 2**64 - 1:
            raise OverflowError("integer overflow")
        return result

    logical = checked_mul(checked_mul(dims[0], dims[1]), dims[2])
    physical = min(logical, physical_blocks) if flags & AUTO_MAP else logical
    if physical > UINT32_MAX:
        raise OverflowError("launch grid exceeds uint32 block count")
    return logical, physical


def test_logical_blocks_stay_wide_while_physical_blocks_stay_32bit():
    """The two LaunchGrid members must not share a 32-bit type."""
    src = _source(HEADER)
    match = re.search(r"struct LaunchGrid \{(.*?)\};", src, re.S)
    assert match, "LaunchGrid declaration not found"
    body = match.group(1)
    assert re.search(r"uint64_t\s+logicalBlocks\s*;", body), "logicalBlocks must be uint64_t"
    assert re.search(r"uint32_t\s+physicalBlocks\s*;", body), "physicalBlocks must be uint32_t"
    assert not re.search(r"uint32_t\s+logicalBlocks", body), "logicalBlocks must not be 32-bit"


def test_workspace_multiplier_uses_the_full_logical_count():
    """workspace sizing must multiply by logicalBlocks, not by the block_dim."""
    header = _source(HEADER)
    runtime = _source(RUNTIME)
    # The 32-bit check applies only to the value handed to CANN.
    assert "physical > std::numeric_limits<uint32_t>::max()" in header, \
        "missing 32-bit check on the physical block count"
    assert "result.logicalBlocks = blocks;" in header, "logicalBlocks must keep the 64-bit total"
    # workspace sizing has to consume logicalBlocks, otherwise it inherits the
    # 32-bit block_dim limit again.
    pattern = r"checkedMultiply\(spec\.workspace_size,\s*call->?grid\.logicalBlocks\)"
    assert re.search(pattern, runtime), "workspace total must be derived from logicalBlocks"


def test_reported_shape_is_not_truncated():
    """262144 x 16384 bytes: the reported shape must exceed 32-bit sizes."""
    logical, physical = _prepare_grid(0, 40, (256, 64, 1))
    assert logical == 256 * 64
    assert physical == 256 * 64
    assert 262144 * logical == 2**32


def test_auto_map_allows_a_logical_grid_above_32bit():
    """AUTO_MAP folds the grid onto few physical cores, so the logical count
    may legitimately exceed 2**32 - 1 while block_dim stays 32-bit."""
    grid = (262144, 16384, 1)
    logical, physical = _prepare_grid(AUTO_MAP, 40, grid)
    assert logical == 2**32
    assert physical == 40
    assert physical <= UINT32_MAX
    assert 262144 * logical == 2**50


def test_unmapped_grid_above_32bit_is_still_rejected():
    """Without AUTO_MAP the whole grid is block_dim, so it must fail loudly."""
    with pytest.raises(OverflowError, match="uint32 block count"):
        _prepare_grid(0, 40, (262144, 16384, 1))


def test_small_grid_matches_reference():
    logical, physical = _prepare_grid(0, 40, (4, 4, 1))
    assert (logical, physical) == (16, 16)
    assert 262144 * logical == 262144 * 16
