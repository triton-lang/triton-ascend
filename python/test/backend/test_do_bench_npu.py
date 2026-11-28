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

import pandas as pd
import pytest
from triton.backends.ascend.testing import _collect_prof_result


def _write_task_time(base_dir, task_times, control_rows):
    rows = []
    if control_rows:
        rows.append({"kernel_name": "PROFILING_ENABLE", "task_time(us)": 0})
    rows.extend({"kernel_name": "vector_add", "task_time(us)": task_time} for task_time in task_times)
    if control_rows:
        rows.append({"kernel_name": "PROFILING_DISABLE", "task_time(us)": 0})
    pd.DataFrame(rows).to_csv(base_dir / "task_time_0.csv", index=False)


def test_collect_prof_result_without_control_rows(tmp_path):
    task_times = list(range(10, 101, 10))
    _write_task_time(tmp_path, task_times, control_rows=False)

    result = _collect_prof_result(str(tmp_path), [lambda: None], num_warmup=3, num_active=7)

    assert result == pytest.approx(sum(task_times[3:]) / 7 / 1e3)


def test_collect_prof_result_with_control_rows(tmp_path):
    task_times = list(range(10, 101, 10))
    _write_task_time(tmp_path, task_times, control_rows=True)

    result = _collect_prof_result(str(tmp_path), [lambda: None], num_warmup=3, num_active=7)

    assert result == pytest.approx(sum(task_times[3:]) / 7 / 1e3)
