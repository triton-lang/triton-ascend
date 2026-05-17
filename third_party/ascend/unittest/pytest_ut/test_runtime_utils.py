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
import os
from triton.backends.ascend import utils
from triton.backends.ascend.runtime import utils as runtime_utils
import torch


def test_get_logger():
    logger = utils.get_logger("test_utils", "INFO")
    assert logger.level == logging.INFO


def test_get_ascend_arch_from_env():
    os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9599"
    result = utils.get_ascend_arch_from_env()
    assert result == "Ascend910_9599"


def test_get_byte_per_numel_supports_unsigned_integer_dtypes():
    assert runtime_utils.get_byte_per_numel(torch.uint16) == 2
    assert runtime_utils.get_byte_per_numel(torch.uint32) == 4
    assert runtime_utils.get_byte_per_numel(torch.uint64) == 8
