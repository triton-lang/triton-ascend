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
"""Workspace sizing for issue #2797 on the native shared-runtime launcher."""
import shutil
import subprocess
from pathlib import Path

import pytest

from triton.backends.ascend import launcher, utils

AUTO_MAP = launcher.AUTO_MAP


@pytest.fixture(scope="module")
def workspace_probe(tmp_path_factory):
    """Build a header-only probe around prepareGrid + checkedMultiply."""
    source = Path(__file__).with_name("launcher_workspace_probe.cpp")
    output = tmp_path_factory.mktemp("workspace-probe") / "probe"
    compiler = shutil.which("c++") or utils._get_bisheng_path()
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-I" + str(Path(launcher.__file__).with_name("launcher_src")),
        str(source), "-o",
        str(output)
    ], check=True, capture_output=True, text=True)
    return output


def _run(probe, flags, physical, grid, workspace_size):
    out = subprocess.check_output(
        [str(probe), str(flags), str(physical), *map(str, grid),
         str(workspace_size)], text=True)
    return out.split()


def test_issue_shape_workspace_is_not_truncated(workspace_probe):
    """262144 x 16384 bytes: the reported shape must exceed 32-bit sizes."""
    values = _run(workspace_probe, 0, 40, (256, 64, 1), 262144)
    logical, physical, total = map(int, values)
    assert logical == 256 * 64
    assert physical == 256 * 64
    # 262144 * 16384 == 2**32, i.e. exactly one past the 32-bit limit.
    assert total == 2**32


def test_workspace_uses_64bit_blocks_under_auto_map(workspace_probe):
    """A logical grid above 2**32 blocks may still map onto few physical cores.

    Workspace must be reserved for every logical program; only the block count
    handed to CANN is limited to 32 bits, which AUTO_MAP already clamps.
    """
    grid = (262144, 16384, 1)
    total_logical = grid[0] * grid[1] * grid[2]
    assert total_logical == 2**32
    values = _run(workspace_probe, AUTO_MAP, 40, grid, 262144)
    logical, physical, total = map(int, values)
    assert logical == total_logical
    assert physical == 40
    assert total == 262144 * total_logical


def test_unmapped_grid_above_32bit_still_rejected(workspace_probe):
    """Without AUTO_MAP the full grid is the block_dim, so it must fail loudly."""
    with pytest.raises(subprocess.CalledProcessError):
        _run(workspace_probe, 0, 40, (262144, 16384, 1), 262144)


def test_small_grid_matches_reference(workspace_probe):
    values = _run(workspace_probe, 0, 40, (4, 4, 1), 262144)
    logical, physical, total = map(int, values)
    assert (logical, physical, total) == (16, 16, 262144 * 16)
