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
"""
Tests for compile_mode as a tunable autotune parameter.

Three scenarios for how the user supplies compile_mode:
  1. Not specified anywhere -> autotune explores both 'simd' and 'simt_only'
     candidate modes (on 910_95 hardware), each tagged on the generated configs.
  2. Specified in @triton.autotune Config kwargs -> the user-supplied configs
     come through with their declared compile_mode; the autotuner also adds
     auto-generated configs spanning both modes (since the call site has no
     compile_mode).
  3. Specified at the kernel call site (kernel[grid](x, compile_mode=...))
     -> autotune restricts the candidate set to that single mode.
"""

import os
import contextlib
from unittest import mock
import sys

import pytest
import torch
import triton
import triton.backends.ascend.runtime  # trigger _patch_autotune
from triton.runtime.jit import JITFunction
from triton.tools.get_ascend_devices import is_compile_on_910_95

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# Skip the whole file on non-SIMT-capable hardware. The autotuner only emits
# 'simt_only' candidate configs on 910_95-class devices; on A3 (e.g. 910B)
# bishengir-compile lacks the SIMT flags (--enable-triton-ir-compile,
# --pure-simt, --num-warps, --threads-per-warp, --shared-mem-dynamic-size)
# and rejects them at compile time. The whole compile_mode-tunable contract
# is moot on non-SIMT hardware, so skip.
if not is_compile_on_910_95:
    pytest.skip(
        "compile_mode tunable feature requires 910_95-class hardware "
        "(SIMT-capable bishengir-compile); skipping on this runner.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def spy_run_kwarg(name):
    """Capture the value of kwarg `name` seen by JITFunction.run on every
    invocation. Any value passed via either call-site kwargs or
    config.all_kwargs() is recorded; absence is recorded as None.
    """
    seen = []
    real_run = JITFunction.run

    def spying_run(self, *args, grid, warmup, **kwargs):
        seen.append(kwargs.get(name))
        return real_run(self, *args, grid=grid, warmup=warmup, **kwargs)

    with mock.patch.object(JITFunction, 'run', spying_run):
        yield seen


def spy_compile_modes():
    """Capture the compile_mode kwarg value seen by JITFunction.run."""
    return spy_run_kwarg('compile_mode')


# ---------------------------------------------------------------------------
# Scenario 1: no compile_mode specified anywhere
# ---------------------------------------------------------------------------


def test_no_compile_mode_explores_both_modes():
    """When no compile_mode is supplied, autotune generates configs in both
    'simd' and 'simt_only' on SIMT-capable hardware and benchmarks them
    together.
    """

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_compile_modes() as seen:
        kernel[(1, )](x)

    modes = {m for m in seen if m is not None}
    assert 'simd' in modes, f"Expected 'simd' in benchmarked modes, got: {modes}"
    assert 'simt_only' in modes, f"Expected 'simt_only' in benchmarked modes, got: {modes}"


# ---------------------------------------------------------------------------
# Scenario 2: compile_mode declared in @triton.autotune Config kwargs
# ---------------------------------------------------------------------------


def test_compile_mode_in_autotune_configs():
    """User-supplied configs in @triton.autotune carry their own compile_mode.
    They flow through unchanged. The autotuner does not strip or override
    them.
    """

    @triton.autotune(configs=[
        triton.Config({'compile_mode': 'simd'}),
        triton.Config({'compile_mode': 'simt_only'}),
    ], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_compile_modes() as seen:
        kernel[(1, )](x)

    modes = {m for m in seen if m is not None}
    # Both modes from user_configs should appear in the benchmarked set.
    assert 'simd' in modes, f"User-supplied SIMD config should run with 'simd', got: {modes}"
    assert 'simt_only' in modes, f"User-supplied SIMT config should run with 'simt_only', got: {modes}"


def test_no_pin_modeless_user_config_defaults_to_simd():
    """When the user supplies a mix of explicit-mode and no-mode configs in
    @triton.autotune AND does not pin compile_mode at the call site, the
    no-mode config falls through to NPUOptions' default ('simd'), while
    explicit-mode configs run in their declared mode.
    """

    @triton.autotune(
        configs=[
            triton.Config({'compile_mode': 'simt_only'}),
            triton.Config({}),  # no compile_mode -> NPUOptions default = simd
        ], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_compile_modes() as seen:
        kernel[(1, )](x)

    # Explicit SIMT config runs as SIMT.
    assert 'simt_only' in seen, f"Explicit SIMT config should run as SIMT, got: {seen}"
    # No-mode config arrives at JIT.run with no compile_mode kwarg
    # (NPUOptions then applies its default of 'simd').
    assert None in seen, (f"No-mode user_config should arrive without compile_mode in kwargs, "
                          f"got: {seen}")


def test_user_pin_propagates_to_modeless_user_configs():
    """When the user pins compile_mode at the call site, that pin propagates
    to user_configs that do not declare their own compile_mode. User_configs
    that DO declare a compile_mode keep their declared value (config wins).
    Verifies that the call-sit pin honours user intent across all configs.
    """

    @triton.autotune(
        configs=[
            triton.Config({'compile_mode': 'simd'}),  # explicit, should stay SIMD
            triton.Config({}),  # no mode, should inherit pin
        ], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_compile_modes() as seen:
        kernel[(1, )](x, compile_mode='simt_only')

    modes = {m for m in seen if m is not None}
    # Explicit SIMD config wins over the call-site pin (config.kwargs takes
    # precedence on autotune-managed keys).
    assert 'simd' in modes, (f"Explicit-mode config should keep its declared compile_mode, got: {modes}")
    # No-mode config inherits the call-site pin.
    assert 'simt_only' in modes, (f"No-mode user_config should inherit the call-site pin 'simt_only', "
                                  f"got: {modes}")


# ---------------------------------------------------------------------------
# Scenario 3: compile_mode passed at the kernel call site
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pinned_mode", ["simd", "simt_only"])
def test_compile_mode_at_call_site_restricts_to_pinned(pinned_mode):
    """When the user pins compile_mode at the kernel call site, autotune
    restricts the candidate set to that single mode. No configs in any
    other mode are benchmarked.
    """

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_compile_modes() as seen:
        kernel[(1, )](x, compile_mode=pinned_mode)

    modes = {m for m in seen if m is not None}
    assert modes == {pinned_mode}, (f"Pinning compile_mode='{pinned_mode}' should restrict the candidate "
                                    f"set to that mode only, got: {modes}")


def test_force_simt_only_kwarg_is_alias_for_simt_pinned():
    """The legacy force_simt_only=True kwarg is treated as an alias for
    compile_mode='simt_only': the candidate set shrinks to SIMT only.
    """

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_compile_modes() as seen:
        kernel[(1, )](x, force_simt_only=True)

    modes = {m for m in seen if m is not None}
    assert modes == {'simt_only'}, (f"force_simt_only=True should pin to SIMT, got: {modes}")


# ---------------------------------------------------------------------------
# Cache scoping: pinned and auto runs do not share entries
# ---------------------------------------------------------------------------


def test_auto_and_pinned_runs_have_separate_cache_entries():
    """An auto-search run and a user-pinned run on the same args produce
    distinct cache entries: the cache key encodes the user constraint.
    """

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")

    kernel[(1, )](x)
    after_auto = len(kernel.cache)

    kernel[(1, )](x, compile_mode='simd')
    after_simd = len(kernel.cache)

    kernel[(1, )](x, compile_mode='simt_only')
    after_simt = len(kernel.cache)

    assert after_auto >= 1, "Auto run should populate the cache"
    assert after_simd > after_auto, (f"Pinning compile_mode='simd' should add a separate cache entry "
                                     f"(auto={after_auto}, after_simd={after_simd})")
    assert after_simt > after_simd, (f"Pinning compile_mode='simt_only' should add another separate entry "
                                     f"(after_simd={after_simd}, after_simt={after_simt})")


# ---------------------------------------------------------------------------
# Pinned runs are repeatable: a second pinned call hits cache
# ---------------------------------------------------------------------------


def test_pinned_run_repeats_hit_cache():
    """A second call with the same pinned compile_mode hits the cache and
    does not produce a new entry.
    """

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")

    kernel[(1, )](x, compile_mode='simt_only')
    after_first = len(kernel.cache)

    kernel[(1, )](x, compile_mode='simt_only')
    after_second = len(kernel.cache)

    assert after_first == after_second, (f"Repeated pinned call should reuse the cached entry "
                                         f"(after_first={after_first}, after_second={after_second})")


# ---------------------------------------------------------------------------
# simt_stack_limit: per-config tag must not collide with a call-site override
# ---------------------------------------------------------------------------


def test_simt_stack_limit_override_at_call_site():
    """A call-site simt_stack_limit is honored on SIMT configs and does not
    collide with the per-config tag the autotuner attaches.

    Regression: _gen_tile_configs tags every SIMT config with simt_stack_limit
    in its kwargs. simt_stack_limit must therefore be autotune-managed in
    _make_kernel_call — otherwise a call-site value lands in both meta and
    config.kwargs and the conflict check raises
    ValueError("Conflicting meta-parameters: simt_stack_limit").
    """

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_run_kwarg('simt_stack_limit') as seen:
        # Before the fix this raised inside _make_kernel_call before any run.
        kernel[(1, )](x, compile_mode='simt_only', simt_stack_limit=16384)

    assert 16384 in seen, (f"Call-site simt_stack_limit=16384 should reach the kernel on the "
                           f"SIMT path, got: {seen}")


def test_simt_stack_limit_defaults_when_unset():
    """When the user does not pass simt_stack_limit, SIMT configs still carry
    the autotuner's default so it reaches the compiler deterministically."""

    @triton.autotune(configs=[], key=[])
    @triton.jit
    def kernel(X):
        pass

    x = torch.randn(1, device="npu")
    with spy_run_kwarg('simt_stack_limit') as seen:
        kernel[(1, )](x, compile_mode='simt_only')

    # 8192 is AutoTilingTuner.simt_stack_limit, tagged on every SIMT config.
    assert 8192 in seen, (f"SIMT configs should carry the default simt_stack_limit=8192 "
                          f"when the user does not override it, got: {seen}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
