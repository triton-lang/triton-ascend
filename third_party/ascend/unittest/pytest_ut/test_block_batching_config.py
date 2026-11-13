"""Batching thresholds use validated available cores, without a second parser."""
from types import MethodType, SimpleNamespace

import pytest
from triton.backends.ascend import block_batching as batching
from triton.backends.ascend.driver import NPUUtils
from triton.runtime import driver


@pytest.fixture
def device(monkeypatch):
    utils = SimpleNamespace(get_device_core=lambda: (32, 64))
    for name in ('_get_npu_device_limit_form_env', 'get_device_properties', 'get_aicore_num', 'get_aivector_core_num'):
        setattr(utils, name, MethodType(getattr(NPUUtils, name), utils))
    monkeypatch.setattr(driver, '_active', SimpleNamespace(utils=utils))
    monkeypatch.delenv('TA_BLOCK_BATCHING_CORE_FLOOR', raising=False)
    monkeypatch.delenv('NPU_DEVICE_LIMIT', raising=False)
    return utils


@pytest.mark.parametrize('limit,expected', [(None, 64), ('16,32', 32), ('8,16', 16), (' 8, 16 ', 16)])
def test_core_floor_uses_available_aiv(device, monkeypatch, limit, expected):
    if limit is not None:
        monkeypatch.setenv('NPU_DEVICE_LIMIT', limit)
    assert batching._core_floor() == device.get_aivector_core_num() == expected


@pytest.mark.parametrize('limit', ['0,0', '33,66', '8,8', 'invalid', '8', '-1,-2'])
@pytest.mark.parametrize('override', [None, '0', '7'])
def test_core_floor_preserves_device_validation(device, monkeypatch, limit, override):
    monkeypatch.setenv('NPU_DEVICE_LIMIT', limit)
    if override is not None:
        monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', override)
    with pytest.raises(ValueError) as expected:
        device.get_aivector_core_num()
    with pytest.raises(ValueError) as actual:
        batching._core_floor()
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize('override', ['0', '7', '64'])
def test_core_floor_explicit_override_has_priority_after_validation(device, monkeypatch, override):
    monkeypatch.setenv('NPU_DEVICE_LIMIT', '8,16')
    monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', override)
    assert batching._core_floor() == int(override)


def test_core_floor_preserves_integer_override_validation(device, monkeypatch):
    monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', 'invalid')
    with pytest.raises(ValueError):
        batching._core_floor()
