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
import logging
import warnings

import pytest
import torch

from triton.backends.ascend import utils
from triton.backends.ascend.runtime import utils as runtime_utils


def test_get_logger():
    logger = utils.get_logger("test_utils", "INFO")
    assert logger.level == logging.INFO


def test_get_ascend_arch_from_env_is_deprecated(monkeypatch):
    monkeypatch.setattr(utils, "_WARNED_DEPRECATED_ASCEND_ENV_VARS", set(), raising=False)
    monkeypatch.setenv("TRITON_ASCEND_ARCH", "Ascend910_9599")
    with pytest.warns(FutureWarning, match=r"TRITON_ASCEND_ARCH.*deprecated and ignored"):
        result = utils.get_ascend_arch_from_env()
    assert result == ""


def test_deprecated_ascend_env_var_warns_only_once_per_process(monkeypatch):
    name = "TRITON_REGISTER_TENSOR_MSPROF"
    monkeypatch.setattr(utils, "_WARNED_DEPRECATED_ASCEND_ENV_VARS", set(), raising=False)
    monkeypatch.delenv(name, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        utils._warn_deprecated_ascend_env_var(name)
        monkeypatch.setenv(name, "1")
        utils._warn_deprecated_ascend_env_var(name)
        utils._warn_deprecated_ascend_env_var(name)

    assert len(caught) == 1


@pytest.mark.parametrize(
    "simt_option",
    [
        pytest.param("force_simt_only", id="simt-only"),
        pytest.param("force_simt_template", id="simt-template"),
    ],
)
def test_retained_simt_option_is_not_removed(simt_option):
    options = {simt_option: True}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = utils._remove_deprecated_npu_options(options)

    assert not caught
    assert normalized == options == {simt_option: True}


def test_explicit_compile_mode_preserves_retained_simt_option():
    options = {"compile_mode": "simd", "force_simt_only": True}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = utils._remove_deprecated_npu_options(options)

    assert not caught
    assert normalized == options


def test_retained_simt_option_is_preserved_after_in_place_normalization():
    options = {"force_simt_only": True}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = utils._remove_deprecated_npu_options(options, in_place=True)
        second = utils._remove_deprecated_npu_options(options, in_place=True)

    assert not caught
    assert first is second is options
    assert options == {"force_simt_only": True}


def test_get_byte_per_numel_supports_unsigned_integer_dtypes():
    assert runtime_utils.get_byte_per_numel(torch.uint16) == 2
    assert runtime_utils.get_byte_per_numel(torch.uint32) == 4
    assert runtime_utils.get_byte_per_numel(torch.uint64) == 8
