import json
import os
import sys
from unittest.mock import MagicMock

import pytest
import triton.backends.ascend.compiler as compiler

pytestmark = pytest.mark.backend("cpu")


def test_cv_split_preserves_user_pipeline_policy():
    metadata = {
        "multibuffer": True,
        "set_workspace_multibuffer": 2,
        "has_auto_blockify_blacklist_op": False,
        "enable_mixed_cv": False,
        "disable_auto_inject_block_sync": False,
    }

    compiler._configure_cv_split_metadata(metadata)

    assert metadata["multibuffer"] is True
    assert metadata["set_workspace_multibuffer"] == 2
    assert metadata["has_auto_blockify_blacklist_op"] is False
    assert metadata["enable_mixed_cv"] is False
    assert metadata["disable_auto_inject_block_sync"] is True


@pytest.mark.parametrize(
    "dynamic, cv_split, target_is_a5, expected",
    [
        (False, False, True, (False, False)),
        (True, False, True, (False, True)),
        (False, True, True, (True, False)),
        # Auto mode: attempt CV split, then let DCVP observe the commit result
        # and run only if CV split kept the original module.
        (True, True, True, (True, True)),
        (True, True, False, (False, True)),
    ],
)
def test_cv_pipeline_selection(dynamic, cv_split, target_is_a5, expected):
    metadata = {
        "enable_dynamic_cv_pipeline": dynamic,
        "enable_cv_split_scheduling": cv_split,
    }
    assert compiler._select_cv_pipeline_policy(metadata, target_is_a5) == expected


def test_cv_split_a5_default_is_transactional_auto():
    """The two switches carry no value of their own.

    Both are left unset on the dataclass and resolved in `parse_options` from
    `is_compile_on_910_95()`, the same way `compile_on_910_95` itself is -- so
    on an A5 target both halves of the transactional default turn on, and on
    anything else neither does.

    This asserts that they are unset and that an explicit choice survives, not
    which callable the field happens to hold. Pinning the field default is what
    this test used to do, and it went stale the moment the resolution moved.
    """
    fields = compiler.NPUOptions.__dataclass_fields__
    assert fields["enable_cv_split_scheduling"].default is None
    assert fields["enable_dynamic_cv_pipeline"].default is None
    assert fields["cv_split_unroll_factor"].default == 4

    # Unset stays unset until parse_options fills it in.
    assert compiler.NPUOptions().enable_cv_split_scheduling is None
    assert compiler.NPUOptions().enable_dynamic_cv_pipeline is None

    # An explicit choice is never overwritten by that resolution.
    assert compiler.NPUOptions(enable_cv_split_scheduling=False).enable_cv_split_scheduling is False
    assert compiler.NPUOptions(enable_dynamic_cv_pipeline=True).enable_dynamic_cv_pipeline is True


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
