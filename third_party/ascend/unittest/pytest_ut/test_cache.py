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

import importlib.util
import itertools
import pathlib

import pytest
import torch
import torch_npu  # noqa: F401  # Registers the "npu" device with PyTorch.

import triton
import triton.language as tl


@triton.jit
def function_0(i):
    return i + 1


@triton.jit
def function_1(i):
    i = i + 1
    cond: tl.constexpr = True
    if cond:
        FN: tl.constexpr = function_2
    else:
        FN: tl.constexpr = function_0
    return FN(i)


@triton.jit
def function_2(i):
    i = i + 1
    return i


@triton.jit
def combine_fn(a, b):
    return COMBINE_OP  # noqa: F821


@triton.jit
def kernel(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit
def kernel_with_combine_fn(X, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    i = REDUCE_OR_SCAN(i, 0, combine_fn)  # noqa: F821
    tl.store(X, i)


def apply_src_change(target, old, new, to_modify):
    orig_src = to_modify.src
    kernel.hash = None
    function_0.hash = None
    function_1.hash = None
    function_2.hash = None
    try:
        to_modify._unsafe_update_src(orig_src.replace(old, new))
        return target.cache_key
    finally:
        to_modify._unsafe_update_src(orig_src)
        kernel.hash = None
        function_0.hash = None
        function_1.hash = None
        function_2.hash = None


def test_nochange():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 1', function_1)
    assert baseline == updated


def test_toplevel_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_1)
    assert baseline != updated


def test_nested1_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_2)
    assert baseline != updated


def test_nested2_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_0)
    assert baseline != updated


def test_combine_fn_change():
    # Test that tl.reduce and associative_scan calls include the combine_fn in
    # the hash.
    orig_combine_fn_src = combine_fn.src
    orig_kernel_src = kernel_with_combine_fn.src
    seen_keys = set()

    for reduce_or_scan, combine_op in itertools.product(
        ["tl.reduce", "tl.associative_scan"],
        ["a + b", "a * b"],
    ):
        combine_fn._unsafe_update_src(orig_combine_fn_src.replace("COMBINE_OP", combine_op))
        kernel_with_combine_fn._unsafe_update_src(orig_kernel_src.replace("REDUCE_OR_SCAN", reduce_or_scan))
        try:
            key = kernel_with_combine_fn.cache_key
        finally:
            combine_fn._unsafe_update_src(orig_combine_fn_src)
            kernel_with_combine_fn._unsafe_update_src(orig_kernel_src)

        assert key not in seen_keys
        seen_keys.add(key)


@triton.constexpr_function
def constexpr_flag_fn():
    return False


@triton.jit
def constexpr_fn_user(out):
    a: tl.constexpr = constexpr_flag_fn()
    tl.store(out, a)


def test_constexpr_fn_change():
    baseline = constexpr_fn_user.cache_key

    orig_src = constexpr_flag_fn.src
    new_src = orig_src.replace("False", "True")
    constexpr_flag_fn._unsafe_update_src(new_src)
    constexpr_fn_user.hash = None
    updated = constexpr_fn_user.cache_key
    assert baseline != updated

    constexpr_flag_fn._unsafe_update_src(orig_src)
    constexpr_fn_user.hash = None
    assert constexpr_fn_user.cache_key == baseline


def write_and_load_module(temp_file: pathlib.Path, code, num_extra_lines):
    temp_file.write_text(('# extra line\n' * num_extra_lines) + code)
    spec = importlib.util.spec_from_file_location("module.name", str(temp_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_line_numbers_invalidate_cache(tmp_path: pathlib.Path):
    from textwrap import dedent
    code = dedent("""
        import triton
        @triton.jit
        def test_kernel(i):
            i = i + 1
    """)
    temp_file0 = tmp_path / "test_changed_line_numbers_invalidate_cache0.py"
    orig_mod = write_and_load_module(temp_file0, code, 0)
    orig_cache_key = orig_mod.test_kernel.cache_key

    temp_file1 = tmp_path / "test_changed_line_numbers_invalidate_cache1.py"
    updated_mod = write_and_load_module(temp_file1, code, 1)
    updated_cache_key = updated_mod.test_kernel.cache_key
    assert orig_cache_key != updated_cache_key


def test_use_builtin():

    @triton.jit
    def builtin_kernel():
        a = float(0)  # noqa: F841

    # No error about the value of `float` changing.
    builtin_kernel[(1, )]()
    builtin_kernel[(1, )]()


def test_no_cache_module_as_global():

    @triton.jit
    def module_kernel():
        tl.arange(0, 16)

    module_kernel[(1, )]()
    # `tl` should not be entered into used_global_vals.
    assert not module_kernel.used_global_vals


@triton.jit
def no_cache_callable_inner():
    pass


def test_no_cache_callable():

    @triton.jit
    def callable_kernel():
        no_cache_callable_inner()

    callable_kernel[(1, )]()
    # `no_cache_callable_inner` should not be entered into used_global_vals.
    assert not callable_kernel.used_global_vals


device = "npu"
GLOBAL_DEFAULT_ARG = 1
GLOBAL_VAR = tl.constexpr(1)
GLOBAL = 42  # noqa: N816
CONSTEXPR_GLOBAL = tl.constexpr(42)
BUILTIN_AS_GLOBAL = tl.int32


def test_kernel_global_var_change():
    global GLOBAL_VAR

    previous_global = GLOBAL_VAR
    GLOBAL_VAR = tl.constexpr(1)
    try:

        @triton.jit
        def global_var_kernel(X):
            tl.store(X, GLOBAL_VAR)

        x = torch.empty(1, dtype=torch.int32, device=device)
        global_var_kernel[(1, )](x)
        assert x == torch.ones_like(x)

        GLOBAL_VAR = 2
        with pytest.raises(RuntimeError) as error:
            global_var_kernel[(1, )](x)

        assert "global variable" in str(error.value).lower()
    finally:
        GLOBAL_VAR = previous_global


def test_local_shadows_global():
    global GLOBAL

    previous_global = GLOBAL
    GLOBAL = 42
    try:

        @triton.jit
        def local_shadow_kernel():
            _, GLOBAL = 0, 0  # noqa: N806
            a = GLOBAL  # noqa: F841

        # No error because the local `GLOBAL` is distinct from the module-level
        # value changed between launches.
        local_shadow_kernel[(1, )]()
        GLOBAL = 43
        local_shadow_kernel[(1, )]()
    finally:
        GLOBAL = previous_global


def test_local_does_not_shadow_global():
    global CONSTEXPR_GLOBAL

    previous_global = CONSTEXPR_GLOBAL
    CONSTEXPR_GLOBAL = tl.constexpr(42)
    try:

        @triton.jit
        def local_constexpr_kernel():
            a = CONSTEXPR_GLOBAL  # noqa: F823, F841
            _, CONSTEXPR_GLOBAL = 0, 0  # noqa: F841, N806

        local_constexpr_kernel[(1, )]()
        CONSTEXPR_GLOBAL = tl.constexpr(43)

        # The first read uses the module global even though the kernel later
        # assigns a local with the same name, so changing the global is an
        # error on the next launch.
        with pytest.raises(RuntimeError):
            local_constexpr_kernel[(1, )]()
    finally:
        CONSTEXPR_GLOBAL = previous_global


def test_cache_builtin_as_global():
    global BUILTIN_AS_GLOBAL

    previous_global = BUILTIN_AS_GLOBAL
    BUILTIN_AS_GLOBAL = tl.int32
    try:

        @triton.jit
        def builtin_global_kernel():
            x = BUILTIN_AS_GLOBAL  # noqa: F841

        builtin_global_kernel[(1, )]()

        BUILTIN_AS_GLOBAL = tl.int64
        with pytest.raises(RuntimeError) as error:
            builtin_global_kernel[(1, )]()

        assert "global variable" in str(error.value).lower()
    finally:
        BUILTIN_AS_GLOBAL = previous_global


def test_cache_closure():

    def make_closure(cst):

        @triton.jit
        def closure():
            tl.full((16, ), cst, dtype=tl.int32)

        return closure

    cst = tl.constexpr(42)
    closure = make_closure(cst)

    closure[(1, )]()
    cst.value = 43
    with pytest.raises(RuntimeError) as error:
        closure[(1, )]()

    assert "cst has changed since we compiled this kernel, from constexpr[42] to constexpr[43]" in str(error.value)


def test_kernel_default_arg():
    global GLOBAL_DEFAULT_ARG

    previous_default = GLOBAL_DEFAULT_ARG
    GLOBAL_DEFAULT_ARG = 1
    try:

        @triton.jit
        def kernel(X, i: tl.constexpr = GLOBAL_DEFAULT_ARG):
            tl.store(X, i)

        x = torch.empty(1, dtype=torch.int32, device=device)
        kernel[(1, )](x)
        assert x == torch.ones_like(x)

        # Changing the global variable must not change the default value that
        # was captured when kernel was defined.
        GLOBAL_DEFAULT_ARG = 2
        kernel[(1, )](x)
        assert x == torch.ones_like(x)

        device_id = torch.npu.current_device()
        assert len(kernel.device_caches[device_id][0]) == 1
    finally:
        GLOBAL_DEFAULT_ARG = previous_default


def test_constexpr_cache_invalidation_recreated():

    def test_run(val):
        VAL = tl.constexpr(val)

        @triton.jit
        def kernel(out):
            tl.store(out, VAL)

        out = torch.zeros(1, device=device)
        kernel[(1, )](out)
        return out.item()

    assert test_run(123) == 123
    assert test_run(123) == 123
    assert test_run(1234) == 1234
    assert test_run(1234) == 1234


def test_jit_warmup_cache():

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    args = [
        torch.randn(32, dtype=torch.float32, device=device),
        torch.randn(32, dtype=torch.float32, device=device),
        torch.randn(32, dtype=torch.float32, device=device),
        32,
    ]
    device_id = torch.npu.current_device()
    assert len(kernel_add.device_caches[device_id][0]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.device_caches[device_id][0]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.device_caches[device_id][0]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.device_caches[device_id][0]) == 1


def test_jit_debug():

    @triton.jit
    def debug_kernel(tmp):
        tl.device_assert(tl.load(tmp) == 1, "tmp == 1")

    device_id = torch.npu.current_device()
    tmp = torch.tensor([1], dtype=torch.int32, device=device)
    assert len(debug_kernel.device_caches[device_id][0]) == 0
    debug_kernel[(1, )](tmp, debug=False)
    assert len(debug_kernel.device_caches[device_id][0]) == 1
    debug_kernel[(1, )](tmp, debug=True)
    assert len(debug_kernel.device_caches[device_id][0]) == 2
    bins = list(debug_kernel.device_caches[device_id][0].values())
    assert bins[0].asm['ttir'] != bins[1].asm['ttir']
