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

import pytest

from triton.backends.ascend.testing import ProfilerResultMismatchError, _collect_prof_result


def _write_kernel_details(tmp_path):
    details = tmp_path / "kernel_details.csv"
    details.write_text("Name,Type,Duration(us)\nother_kernel,AI_CORE,10\n", encoding="utf-8")
    return str(tmp_path)


def test_collect_prof_result_empty_filter_returns_inf(tmp_path, capsys):
    base_dir = _write_kernel_details(tmp_path)

    result = _collect_prof_result(base_dir, [lambda: None], 5, 30, target_kernel_name="wanted")

    assert result == float("inf")
    assert "No profiling rows matched" in capsys.readouterr().out


def test_collect_prof_result_empty_filter_raises_on_mismatch(tmp_path):
    base_dir = _write_kernel_details(tmp_path)

    with pytest.raises(ProfilerResultMismatchError):
        _collect_prof_result(
            base_dir,
            [lambda: None],
            5,
            30,
            target_kernel_name="wanted",
            _raise_on_mismatch=True,
        )
