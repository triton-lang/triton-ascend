# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Common admission and actual-launch checks for feature tests."""
import pytest
import triton


@pytest.fixture(autouse=True)
def batching_target(monkeypatch):
    require_batching_target(monkeypatch)


def require_batching_target(monkeypatch, *, include_a3=False):
    arch = triton.runtime.driver.active.get_current_target().arch
    targets = ('Ascend950', 'Ascend910_95', 'Ascend910_93') if include_a3 else ('Ascend950', 'Ascend910_95')
    if not arch.startswith(targets):
        pytest.skip('This test requires ' + '/'.join(targets))
    # These are correctness tests, independent of the occupancy heuristic.
    monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', '0')


def assert_applied(kernel, factor):
    assert kernel.last_launch['runtime_status'] == 'APPLIED', (kernel.block_batching, kernel.last_launch)
    assert kernel.last_launch['applied_factor'] == factor, kernel.last_launch
