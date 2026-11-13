# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Common admission and actual-launch checks for feature tests."""
import pytest
import triton


@pytest.fixture(autouse=True)
def batching_target(monkeypatch):
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch.startswith(('Ascend950', 'Ascend910_95')):
        pytest.skip('Block batching is supported on Ascend950/Ascend910_95')
    # These are correctness tests, independent of the occupancy heuristic.
    monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', '0')


def assert_applied(kernel, factor):
    assert kernel.last_launch['runtime_status'] == 'APPLIED', (kernel.block_batching, kernel.last_launch)
    assert kernel.last_launch['applied_factor'] == factor, kernel.last_launch
