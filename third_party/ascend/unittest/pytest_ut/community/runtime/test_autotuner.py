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

# Source: python/test/unit/runtime/test_autotuner.py at main-dev@396df6cb5b001314e36f22220be07a560de44664

import torch

import triton
import triton.language as tl
import pytest

import pathlib
import uuid
from triton._internal_testing import is_cuda


def do_bench(kernel_call, quantiles, use_cuda_graph=False):
    if use_cuda_graph:
        return triton.testing.do_bench_cudagraph(kernel_call, quantiles=quantiles)
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_kwargs(use_cuda_graph: bool, device: str):
    if use_cuda_graph and not torch.cuda.is_available():
        pytest.xfail("CUDA is not available")

    M, N = 1024, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.autotune(configs=configs, key=["M"],
                     do_bench=lambda kernel, quantiles: do_bench(kernel, quantiles, use_cuda_graph))
    @triton.jit
    def _kernel(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
    _kernel[grid](dst, src, N, M, N)
    # the key word args could be in arbitrary order.
    _kernel[grid](dst=dst, src=src, M=M // 2, stride_m=N, BLOCK_SIZE_N=N)
    assert len(_kernel.cache) == 2


def test_no_do_bench(device: str):
    M, N = 1024, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.autotune(configs=configs, key=["M"])
    @triton.jit
    def _kernel(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
    _kernel[grid](dst, src, N, M, N)
    assert len(_kernel.cache) == 1


@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
def test_restore(pass_kwargs_to_kernel, device):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    if pass_kwargs_to_kernel:
        _kernel[grid](src=src, N=N)
    else:
        _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))


def test_hooks(device):
    # Autotuner's pre- and post- hooks should be called the same number of times
    N = 4096
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 4096}), triton.Config(kwargs={'BLOCK_SIZE': 32})]

    values = {"counter": 0, "has_exception": False}

    def _pre_hook(*args, **kwargs):
        values["counter"] += 1

    def _post_hook(*args, exception):
        values["counter"] -= 1
        if exception is not None:
            values["has_exception"] = True
        assert values["counter"] == 0

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench, pre_hook=_pre_hook, post_hook=_post_hook)
    @triton.heuristics({"N_STAGES": lambda nargs: 100 if nargs['N'] == 4096 else 4})
    @triton.jit
    def _kernel(src, N, N_STAGES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        max_iters = tl.cdiv(N, BLOCK_SIZE)
        for _ in tl.range(max_iters, num_stages=N_STAGES):
            x = tl.load(src + offsets, mask=offsets < N)
            tl.store(src + offsets, x, mask=offsets < N)
            offsets += BLOCK_SIZE

    _kernel[(1, )](src, N)

    # On NVIDIA GPUs:
    # The tuning knob `num_stages` can be set by users.
    # This will cause out of resources when N_STAGES = 100
    # shared memory bytes = N_STAGES * BLOCK_SIZE * sizeof(float)
    # On AMD GPUs:
    # `num_stages` is a fixed value of 2, so it won't cause out of resources
    if triton.runtime.driver.active.get_current_target().backend == "cuda":
        assert values["has_exception"] is True
    else:
        assert values["has_exception"] is False


@pytest.mark.parametrize('with_perf_model', [False, True])
def test_prune_configs(with_perf_model: bool, device: str):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)
    records = {}

    def early_config_prune(configs, named_args, **kwargs):
        records['run_early_config_prune'] = True
        if "N" in kwargs and kwargs["N"] == 1024:
            records['capture_kwargs'] = True
        if "dst" in named_args and "src" in named_args and len(named_args) == 2:
            records['capture_named_args'] = True
        return [configs[0]]

    def perf_model(*args, **kwargs):
        records['run_perf_model'] = True
        return kwargs['BLOCK_SIZE']

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    if with_perf_model:
        prune_configs_by = {'perf_model': perf_model, 'top_k': 1}
    else:
        prune_configs_by = {'early_config_prune': early_config_prune}

    @triton.autotune(configs=configs, key=['N'], prune_configs_by=prune_configs_by, do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)
    torch.testing.assert_close(src, dst)
    if with_perf_model:
        assert len(records) == 1
        assert records['run_perf_model']
    else:
        assert len(records) == 3
        assert records['run_early_config_prune']
        assert records['capture_kwargs']
        assert records['capture_named_args']


def test_prune_all_configs(device):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    def early_config_prune(configs, named_args, **kwargs):
        return []

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    prune_configs_by = {'early_config_prune': early_config_prune}

    @triton.autotune(configs=configs, key=['N'], prune_configs_by=prune_configs_by)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    try:
        _kernel[grid](dst, src, N=N)
        pytest.fail("Expected exception was not thrown.")
    except triton.TritonError as e:
        assert e is not None and str(
            e
        ) == "Autotuner error: No valid autotuner configs after pruning. `early_config_prune` should return at least one config."
