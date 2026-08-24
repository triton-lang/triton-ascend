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

import os
import traceback

import pytest
import torch_npu  # noqa: F401  # Registers the "npu" device with PyTorch.

import triton
import triton.language as tl
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure


def format_exception(type, value, tb):
    list_msg = traceback.format_exception(type, value, tb, chain=False)
    return "\n".join(list_msg)


def test_err_undefined_variable():

    @triton.jit
    def kernel():
        a += 1  # noqa

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "is not defined" in err_msg, "error should mention the undefined variable"
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_binary_operator():

    @triton.jit
    def kernel():
        0 + "a"

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in err_msg, "error should point to the 0"
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_static_assert():

    @triton.jit
    def kernel():
        tl.static_assert(isinstance(0, tl.tensor))

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        assert isinstance(e.value, CompileTimeAssertionFailure)
        assert e.value.__cause__ is None
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        print(err_msg)
        assert "at 2:4:" in err_msg, "error should point to the static_assert call"
        assert "<source unavailable>" not in err_msg
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_unary_op():
    # Currently Triton can't evaluate `not` of a tuple at compile time. That's
    # acceptable, but the error message needs to point to the correct spot.
    @triton.jit
    def kernel():
        not (0, 0)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        assert e.value.__cause__ is None
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in err_msg, "error should point to the `not`"
        assert "<source unavailable>" not in err_msg
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_binary_op():

    @triton.jit
    def kernel():
        1.0 << 1

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in err_msg, "error should point to the 1.0"
        assert "<source unavailable>" not in err_msg
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


# This has to be defined as a top-level function; jit'ed functions can't call
# nested functions.
@triton.jit
def nested_call():
    xyz  # noqa


def test_err_in_nested_call():

    @triton.jit
    def kernel():
        # this is a comment to push nested_call() onto the next line
        nested_call()

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        inner_exc = e.value.__cause__
        inner = format_exception(inner_exc.__class__, inner_exc, inner_exc.__traceback__)
        assert "at 2:4:" in inner, "error should point to xyz"
        assert "<source unavailable>" not in inner
        assert "code_generator.py" not in inner

        outer = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 3:4" in outer, "error should point to the nested_call"
        assert "<source unavailable>" not in outer
        assert "code_generator.py" not in outer
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_builtin():

    # The root error here comes from core.py. Make sure the stacktrace reflects
    # this.
    @triton.jit
    def kernel():
        tl.expand_dims(None, -1)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        inner_exc = e.value.__cause__
        inner = format_exception(inner_exc.__class__, inner_exc, inner_exc.__traceback__)
        assert f"{os.sep}core.py" in inner, "error should point inside core.py"
        assert "code_generator.py" not in inner

        outer = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in outer, "error should point to expand_dims call"
        assert "<source unavailable>" not in outer
        assert "code_generator.py" not in outer
    except AssertionError as assertion_err:
        raise assertion_err from e.value


@triton.jit
def two_returns():
    return tl.arange(0, 4)
    return tl.arange(0, 8)


def test_two_returns_no_err():
    # This program is valid; `a` has shape (4,).
    @triton.jit
    def kernel():
        a = two_returns()
        a + tl.arange(0, 4)  # only works if we took the first return

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


def test_not_const_annotate_no_err():

    @triton.jit
    def kernel(N: int = 1):
        pass

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'N': 'i32'}, constexprs={}))


@triton.jit
def returns_branched_on_constexpr(N: tl.constexpr):
    if N == 0:
        return tl.arange(0, 4)
    # Ideally this would work even without the `else`, but we're not that smart
    # yet.
    else:
        return tl.arange(0, 8)


def test_returns_branched_on_constexpr():

    @triton.jit
    def kernel1(N: tl.constexpr):
        a = returns_branched_on_constexpr(N)
        a + tl.arange(0, 4)

    triton.compile(triton.compiler.ASTSource(fn=kernel1, signature={"N": "constexpr"}, constexprs={"N": 0}))

    @triton.jit
    def kernel2(N: tl.constexpr):
        a = returns_branched_on_constexpr(N)
        a + tl.arange(0, 8)

    triton.compile(triton.compiler.ASTSource(fn=kernel2, signature={"N": "constexpr"}, constexprs={"N": 1}))


@triton.jit
def returns_branched_on_non_constexpr(N: int):
    if N == 0:
        return tl.arange(0, 4)
    else:
        return tl.arange(0, 8)


def test_returns_branched_on_non_constexpr():

    @triton.jit
    def kernel(N: int):
        returns_branched_on_non_constexpr(N)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'N': 'i32'}, constexprs={}))

    try:
        assert "at 2:4:" in str(e.value), "error should point to the function call"
        assert "at 5:8:" in str(e.value.__cause__), "error should point to the second `return`"
    except AssertionError as assertion_err:
        raise assertion_err from e.value


GLOBAL = 42


def test_global_var_access():

    @triton.jit
    def kernel():
        a = GLOBAL  # noqa: F841

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
    assert "global variable" in str(e.value)


CONSTEXPR_ANNOTATED_GLOBAL: tl.constexpr = 42


def test_constexpr_annotated_global_var_access():

    @triton.jit
    def kernel():
        a = CONSTEXPR_ANNOTATED_GLOBAL  # noqa: F841

    # A Python annotation alone does not make this global accessible from JIT
    # code.
    try:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
        assert False, "Using a constexpr annotated global variable should not be allowed"
    except CompilationError as e:
        assert "Cannot access global variable" in str(e)


CONSTEXPR_GLOBAL = tl.constexpr(42)


def test_constexpr_global_var_access():

    @triton.jit
    def kernel():
        a = CONSTEXPR_GLOBAL  # noqa: F841

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


TYPE_ALIAS = tl.pointer_type(tl.int32)


def test_global_type_alias_access():

    @triton.jit
    def kernel():
        a = TYPE_ALIAS  # noqa: F841

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


def test_global_access_in_fn_default_arg():

    @triton.jit
    def kernel(a=GLOBAL):
        pass

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'a': "i32"}, constexprs={}))


def test_defaults_assign_no_err():

    @triton.jit
    def kernel(a=1, B: tl.constexpr = ""):
        pass

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'a': 'i32', 'B': 'constexpr'}, constexprs={'B': ""}))


extra_words = "These are extra words in the error message."


@triton.must_use_result(extra_words)
@triton.jit
def cube(x):
    return x * x * x


def test_unused_result():

    @triton.jit
    def evil_cube_kernel():
        a = tl.full((64, 64), 0.0, tl.float32)
        cube(a)

    @triton.jit
    def good_cube_kernel():
        a = tl.full((64, 64), 0.0, tl.float32)
        a = cube(a)

    triton.compile(triton.compiler.ASTSource(fn=good_cube_kernel, signature={}, constexprs={}))

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=evil_cube_kernel, signature={}, constexprs={}))

    expected_err_msg = "The result of cube is not being used. " + extra_words
    obtained_err_msg = str(e.value).split('\n')[-1]

    assert expected_err_msg == obtained_err_msg


def test_err_constexpr_and_do_not_specialize():

    @triton.jit(do_not_specialize=["N"])
    def kernel(N: tl.constexpr):
        pass

    with pytest.raises(CompilationError, match="N marked as constexpr and listed in do_not_specialize"):
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={"N": 5}))

    with pytest.raises(CompilationError, match="N marked as constexpr and listed in do_not_specialize"):
        kernel[(1, )](5)
