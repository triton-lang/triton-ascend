# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""pytest glue for tla_ut: map TLA_TEST_STAGE onto skips, provide fixtures."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.tla_test_utils import get_stages, npu_available


@pytest.fixture(scope="session", autouse=True)
def assign_npu(worker_id):
    """One NPU per xdist worker; no-op when no device (ir-only mode)."""
    if not npu_available():
        return
    import torch

    npu_id = 0 if worker_id == "master" else int(worker_id[2:]) % torch.npu.device_count()
    torch.npu.set_device(npu_id)


@pytest.fixture()
def output_dir():
    """Hook for test_ir(output_dir=...); None = script parity (no IR dump)."""
    return None


def pytest_collect_file(file_path, parent):
    """Collect every *.py as a test module (cases are named by op, not test_*)."""
    if file_path.suffix == ".py" and file_path.name != "conftest.py":
        return pytest.Module.from_parent(parent, path=file_path)
    return None


def pytest_collection_modifyitems(config, items):
    _, do_compile, do_run = get_stages()
    stage = os.environ.get("TLA_TEST_STAGE", "all")
    enabled = {"test_compile": do_compile, "test_run": do_run}
    for item in items:
        if not enabled.get(item.name, True):
            item.add_marker(pytest.mark.skip(reason=f"{item.name} skipped (TLA_TEST_STAGE={stage})"))
