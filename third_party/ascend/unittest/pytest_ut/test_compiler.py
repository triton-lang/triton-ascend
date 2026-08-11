import json
import os
import sys
from unittest.mock import MagicMock

import triton.backends.ascend.compiler as compiler


def _make_torch_npu_mock(cfg_dir):
    """Create a mock torch_npu module whose __file__ is in cfg_dir."""
    mock = MagicMock()
    mock.__file__ = os.path.join(cfg_dir, "__init__.py")
    return mock


def _write_acl_config(cfg_dir, config):
    """Write acl_default.json into cfg_dir."""
    cfg_path = os.path.join(cfg_dir, "acl_default.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    return cfg_path


def test_get_simt_stack_limit_default(monkeypatch):
    """Without torch_npu installed, should return default 1152."""
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_from_config(monkeypatch, tmp_path):
    """When acl_default.json has valid simt_stack_size, return it."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 2048}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 2048


def test_get_simt_stack_limit_no_StackSize_key(monkeypatch, tmp_path):
    """When acl_default.json has no StackSize key, return default 1152."""
    _write_acl_config(str(tmp_path), {"OtherConfig": {"foo": 1}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_no_simt_stack_size_key(monkeypatch, tmp_path):
    """When StackSize exists but simt_stack_size is absent, return default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {"other_field": 100}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_empty_StackSize(monkeypatch, tmp_path):
    """When StackSize is an empty dict, return default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_empty_config(monkeypatch, tmp_path):
    """When acl_default.json is an empty dict, return default 1152."""
    _write_acl_config(str(tmp_path), {})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_file_not_found(monkeypatch, tmp_path):
    """When acl_default.json does not exist, return default 1152."""
    # Do not write acl_default.json; only mock torch_npu.__file__
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_invalid_json(monkeypatch, tmp_path):
    """When acl_default.json is invalid JSON, return default 1152."""
    cfg_path = os.path.join(str(tmp_path), "acl_default.json")
    with open(cfg_path, "w") as f:
        f.write("{invalid json content}")

    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 1152


def test_get_simt_stack_limit_from_config_returns_int(monkeypatch, tmp_path):
    """Return value from config should be an integer."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 2048}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert isinstance(result, int)
    assert result == 2048


def test_get_simt_stack_limit_config_overrides_default(monkeypatch, tmp_path):
    """Config value should take precedence over default 1152."""
    _write_acl_config(str(tmp_path), {"StackSize": {"simt_stack_size": 9999}})
    mock_npu = _make_torch_npu_mock(str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch_npu", mock_npu)

    result = compiler.get_simt_stack_limit()
    assert result == 9999
    assert result != 1152
