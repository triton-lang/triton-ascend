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
"""Guard the lazy backend initialization introduced by PR #1200.

A fixed import-time limit is unreliable when CI workers share CPU and I/O.
Check the deferred work directly in a fresh subprocess, independently of
modules already imported by pytest and its fixtures.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.backend("native")

_IMPORT_SCRIPT = """
import sys
import triton
from triton.backends.ascend import utils

assert utils._is_compile_on_910_95 is None, "import triton queried the device target"
assert triton.runtime.driver._default is None, "import triton initialized the default driver"
assert triton.runtime.driver._active is None, "import triton initialized the active driver"
loaded_runtimes = sorted({"acl", "torch", "torch_npu", "mindspore"} & sys.modules.keys())
assert not loaded_runtimes, f"import triton eagerly loaded runtime modules: {loaded_runtimes}"
"""


def test_import_triton_keeps_backend_initialization_lazy():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        # Bound hangs; this is not a performance threshold.
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
