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

import linecache
import warnings

import pytest
import triton.language as tl
from triton.language.core import builtin
from triton.language.extra import tla
from triton.language.extra.tla import experimental
from triton.language.extra.tla.jit import TLASTSource

from utils.tla_test_utils import Options, compile_offline, make_context


@experimental
@builtin
def _experimental_abs(x, _semantic=None):
    return tla.abs(x, _semantic=_semantic)


@tla.jit
def _kernel(out, x, use_experimental: tl.constexpr):
    if use_experimental:
        tl.store(out, _experimental_abs(x))
    else:
        tl.store(out, tla.abs(x))


def test_experimental_call_warns_at_kernel_source():
    with pytest.warns(UserWarning, match=r"_experimental_abs is experimental") as caught:
        module, _ = compile_offline(_kernel, {"out": "*i32", "x": "i32"}, {"use_experimental": True})

    warning, = caught
    assert warning.category is UserWarning
    assert warning.filename == __file__
    assert "_experimental_abs(x)" in linecache.getline(warning.filename, warning.lineno)
    assert module.verify()


def test_unused_experimental_api_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        module, _ = compile_offline(_kernel, {"out": "*i32", "x": "i32"}, {"use_experimental": False})

    assert module.verify()


def test_experimental_warning_can_be_filtered():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=r"^TLA API .* is experimental", category=UserWarning)
        module, _ = compile_offline(_kernel, {"out": "*i32", "x": "i32"}, {"use_experimental": True})

    assert module.verify()


_experimental_cdiv = experimental(tla.cdiv)


@tla.jit
def _helper(x):
    return _experimental_cdiv(x, 2)


@tla.jit
def _kernel_with_helper(out, x):
    tl.store(out, _helper(x))


def test_tla_jit_source_warns_in_nested_helper():
    source = TLASTSource(_kernel_with_helper, {"out": "*i32", "x": "i32"})
    with pytest.warns(UserWarning, match=r"cdiv is experimental") as caught:
        module = source.make_ir(None, Options(), {}, {}, make_context())

    warning, = caught
    assert warning.filename == __file__
    assert "return _experimental_cdiv(x, 2)" in linecache.getline(warning.filename, warning.lineno)
    assert module.verify()
