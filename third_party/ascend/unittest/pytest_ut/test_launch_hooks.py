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

import torch_npu  # noqa: F401  # Registers the "npu" device with PyTorch.

import triton


def test_metadata():
    used_hook = False

    def _launch_metadata(grid, kernel, args):
        return {"grid": grid, "value": args["x"]}

    def hook(launch_metadata):
        nonlocal used_hook
        metadata = launch_metadata.get()
        assert metadata["grid"] == (1, 3, 2)
        assert metadata["value"] == 6
        used_hook = True

    @triton.jit(launch_metadata=_launch_metadata)
    def kernel(x):
        pass

    triton.knobs.runtime.launch_enter_hook.add(hook)
    try:
        kernel[(1, 3, 2)](6)
        assert used_hook
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(hook)


def test_load_hook():
    used_start_hook = False
    start_hash = None

    def hook_start(module, function, name, metadata_group, hash):
        nonlocal used_start_hook, start_hash
        start_hash = hash
        used_start_hook = True

    used_end_hook = False
    end_hash = None

    def hook_end(module, function, name, metadata_group, hash):
        nonlocal used_end_hook, end_hash
        end_hash = hash
        used_end_hook = True

    @triton.jit
    def kernel(x):
        pass

    triton.knobs.runtime.kernel_load_start_hook.add(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end)
    try:
        kernel[(1, 3, 2)](6)
        assert used_start_hook
        assert used_end_hook
        assert start_hash == end_hash
    finally:
        triton.knobs.runtime.kernel_load_start_hook.remove(hook_start)
        triton.knobs.runtime.kernel_load_end_hook.remove(hook_end)


def test_multiple_hooks():
    start0 = False
    end0 = False
    start1 = False
    end1 = False

    def hook_start0(module, function, name, metadata_group, hash):
        nonlocal start0
        start0 = True

    def hook_end0(module, function, name, metadata_group, hash):
        nonlocal end0
        end0 = True

    def hook_start1(module, function, name, metadata_group, hash):
        nonlocal start1
        start1 = True

    def hook_end1(module, function, name, metadata_group, hash):
        nonlocal end1
        end1 = True

    triton.knobs.runtime.kernel_load_start_hook.add(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end1)

    @triton.jit
    def kernel(x):
        pass

    try:
        kernel[(1, )](6)
        assert start0
        assert end0
        assert start1
        assert end1
    finally:
        triton.knobs.runtime.kernel_load_start_hook.remove(hook_start0)
        triton.knobs.runtime.kernel_load_end_hook.remove(hook_end0)
        triton.knobs.runtime.kernel_load_start_hook.remove(hook_start1)
        triton.knobs.runtime.kernel_load_end_hook.remove(hook_end1)
