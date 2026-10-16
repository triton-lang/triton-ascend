# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

# Source: python/test/unit/language/test_libdevice.py at main-dev@396df6cb5b001314e36f22220be07a560de44664

import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_dividef as my_fast_dividef


def test_libdevice_rename(device):
    # mark the import as used by this test
    _ = my_fast_dividef

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device=device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)
