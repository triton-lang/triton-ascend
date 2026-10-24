# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import triton
from triton.backends.ascend.utils import is_compile_on_910_95


def require_dot_scaled_formats(a_format, b_format, scale_format=None):
    # A2/A3 support half operands and integer E8M0 scales. Omitted scales
    # do not impose a dtype requirement, including in batched dot tests.
    if a_format in ("fp16", "bf16") and b_format in ("fp16", "bf16") and scale_format in (None, "e8m0"):
        return
    # Query only at test execution; importing helpers or collecting tests
    # must not initialize a device.
    if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("FP4/FP8 operands and FP8 scales require Ascend 950")
