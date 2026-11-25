# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""Host-only contract tests for the scalar dtypes exported by ``tla``.

``tla`` re-exports Triton's scalar types so that kernels can spell buffer and
vector element types as ``tla.float32`` instead of ``tl.float32``.  The objects
must stay identical to the Triton ones, otherwise ``tla.buffer_type`` and
``tla.vector_type`` would reject them.
"""
import inspect

import pytest
import triton.language as tl
from triton.language.extra import tla

# tla export -> (Triton dtype name, primitive bitwidth)
DTYPES = {
    "bfloat16": ("bf16", 16),
    "float16": ("fp16", 16),
    "float32": ("fp32", 32),
    "int8": ("int8", 8),
    "int16": ("int16", 16),
    "int32": ("int32", 32),
    "int64": ("int64", 64),
    "uint8": ("uint8", 8),
    "uint16": ("uint16", 16),
    "uint32": ("uint32", 32),
    "uint64": ("uint64", 64),
}


@pytest.mark.parametrize("name", sorted(DTYPES))
def test_dtype_exports(name):
    expected_name, bits = DTYPES[name]
    ty = getattr(tla, name)
    assert name in tla.__all__
    assert ty is getattr(tl, name)
    assert isinstance(ty, tla.dtype)
    assert ty.name == expected_name
    assert ty.primitive_bitwidth == bits


def test_dtype_constructor():
    assert inspect.isclass(tla.dtype)
    assert tla.dtype is tl.dtype
    for name, (expected_name, _) in DTYPES.items():
        assert tla.dtype(expected_name) == getattr(tla, name)
    with pytest.raises(AssertionError):
        tla.dtype("float32")


@pytest.mark.parametrize("name", sorted(DTYPES))
def test_buffer_and_vector_element_types(name):
    ty = getattr(tla, name)
    assert tla.buffer_type((16, ), ty).element_ty is ty
    assert tla.vector_type(16, ty).element_ty is ty


def test_star_import():
    namespace = {}
    exec("from triton.language.extra.tla import *", namespace)
    for name in sorted(DTYPES):
        assert namespace[name] is getattr(tla, name)
    assert namespace["dtype"] is tla.dtype
