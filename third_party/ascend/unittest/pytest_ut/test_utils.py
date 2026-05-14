import os
import platform
from unittest.mock import patch, mock_open
import unittest
import pytest
from triton.backends.ascend.utils import (
    get_cann_version,
    get_machine_arch,
)


@patch.object(platform, "machine")
def test_get_machine_arch_normal_case(mock_machine):
    mock_machine.return_value = "x86_64"
    result = get_machine_arch()
    assert result == "x86_64"


@patch.object(platform, "machine")
def test_get_machine_arch_arm_case(mock_machine):
    mock_machine.return_value = "arm64"
    result = get_machine_arch()
    assert result == "aarch64"


@patch.object(platform, "machine")
def test_get_machine_arch_unknown_case(mock_machine):
    mock_machine.return_value = "unknown"
    with pytest.raises(KeyError):
        get_machine_arch()


def test_get_cann_version_normal_case():
    test_version = "9.0.0"
    test_inner_version = "v100rc00"
    test_file_context = f"version={test_version}\ninnerversion={test_inner_version}"
    with patch("builtins.open", mock_open(read_data=test_file_context)):
        result = get_cann_version()
        assert result == "CANN-" + test_version + "-" + test_inner_version


def test_get_cann_version_inner_empty_case():
    test_version = "9.0.0"
    test_file_context = f"version={test_version}\naaa=bbb"
    with patch("builtins.open", mock_open(read_data=test_file_context)):
        result = get_cann_version()
        assert result == "CANN-" + test_version


def test_get_cann_version_file_not_find_case():
    with patch("builtins.open") as mock_open_file:
        mock_open_file.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_cann_version()


def test_get_cann_version_empty_file_case():
    with patch("builtins.open", mock_open(read_data="")):
        with pytest.raises(ValueError):
            get_cann_version()
