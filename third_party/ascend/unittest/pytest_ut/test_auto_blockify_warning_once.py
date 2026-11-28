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
"""Tests for AutoBlockify safety warning suppression.

Warmup compiles the same kernel many times and the safety scan result does not
change between compilations.  The warning must therefore be emitted once per
``(kernel_name, blacklist_reasons)`` pair per process.
"""

import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytestmark = pytest.mark.backend("none")

UTILS_PATH = Path(__file__).resolve().parents[2] / "backend" / "utils.py"


def _load_utils(monkeypatch):
    """Load this checkout's utils.py without an installed Ascend backend."""
    triton_stub = types.ModuleType("triton")
    triton_stub.knobs = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "triton", triton_stub)

    backends_stub = types.ModuleType("triton.backends")
    backends_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "triton.backends", backends_stub)

    ascend_stub = types.ModuleType("triton.backends.ascend")
    ascend_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", ascend_stub)

    register_stub = types.ModuleType("triton.backends.ascend.backend_register")
    register_stub.backend_strategy_registry = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.backend_register", register_stub)

    pybind11_stub = types.ModuleType("pybind11")
    pybind11_stub.get_include = lambda: ""
    monkeypatch.setitem(sys.modules, "pybind11", pybind11_stub)

    spec = importlib.util.spec_from_file_location("backend_utils_auto_blockify_ut", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_warnings(utils, calls):
    output = io.StringIO()
    with redirect_stdout(output):
        for kernel_name, reasons in calls:
            utils._warn_auto_blockify_disabled(kernel_name, reasons)
    return output.getvalue()


def test_duplicate_warnings_are_suppressed(monkeypatch):
    utils = _load_utils(monkeypatch)

    output = _capture_warnings(utils, [
        ("token_bin_counts_and_mask_kernel", ["atomic operations"]),
        ("token_bin_counts_and_mask_kernel", ["atomic operations"]),
    ])

    assert output.count("[WARNING] AutoBlockify disabled") == 1
    assert "token_bin_counts_and_mask_kernel" in output


def test_new_kernel_or_new_reasons_still_warns(monkeypatch):
    utils = _load_utils(monkeypatch)

    output = _capture_warnings(utils, [
        ("kernel_a", ["atomic operations"]),
        ("kernel_a", ["unsupported dtype"]),
        ("kernel_b", ["atomic operations"]),
    ])

    assert output.count("[WARNING] AutoBlockify disabled") == 3
    assert "kernel_a" in output
    assert "kernel_b" in output


def test_empty_reasons_do_not_warn(monkeypatch):
    utils = _load_utils(monkeypatch)

    output = _capture_warnings(utils, [("kernel_a", [])])

    assert output == ""
