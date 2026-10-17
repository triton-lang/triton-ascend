# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
from collections import Counter

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend import driver


def profiler_wrapper(fn, *args):
    result_path = "./result_profiling"
    skip_first = 10
    wait = 0
    warmup = 3
    active = 30
    repeat = 1
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False, data_simplification=False)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,
                                                 skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path), record_shapes=True,
            profile_memory=False, with_stack=False, with_flops=False, with_modules=False,
            experimental_config=experimental_config) as prof:
        stream.synchronize()
        for _ in range(skip_first + (wait + warmup + active) * repeat):
            fn(*args)
            prof.step()
        stream.synchronize()


@triton.jit
def triton_kernel_add(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
    idx = tl.arange(0, XS)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + idx, tmp2)


@triton.jit
def triton_kernel_or(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
    idx = tl.arange(0, XS)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 | tmp1
    tl.store(out_ptr0 + idx, tmp2)


def triton_add_func(x0, x1, N):
    y0 = torch.empty_like(x0)
    triton_kernel_add[1, 1, 1](y0, x0, x1, N)
    return y0


def triton_or_func(x0, x1, N):
    y0 = torch.empty_like(x0)
    triton_kernel_or[1, 1, 1](y0, x0, x1, N)
    return y0


# ==================== Pytest Test ====================
@pytest.mark.parametrize("dtype, low, high", [
    (torch.float32, 0, 1),
    (torch.float16, 0, 1),
    (torch.bfloat16, 0, 1),
    (torch.int64, 1, 100),
    (torch.int32, 1, 100),
    (torch.int16, 1, 100),
    (torch.int8, 1, 100),
    (torch.bool, 0, 2),
])
def test_elementwise_ops(dtype, low, high):
    N = 1024
    test_case_is_inductor = False

    if dtype == torch.bool:
        x0 = torch.randint(low=low, high=high, size=(N, )).bool().npu()
        x1 = torch.randint(low=low, high=high, size=(N, )).bool().npu()
        triton_cal = triton_or_func(x0, x1, N)
        ref = x0 | x1
    else:
        if dtype.is_floating_point:
            x0 = torch.rand((N, ), dtype=dtype).npu()
            x1 = torch.rand((N, ), dtype=dtype).npu()
        else:
            x0 = torch.randint(low=low, high=high, size=(N, ), dtype=dtype).npu()
            x1 = torch.randint(low=low, high=high, size=(N, ), dtype=dtype).npu()

        triton_cal = triton_add_func(x0, x1, N)
        ref = x0 + x1

    torch.testing.assert_close(triton_cal, ref)

    def wrapper():
        _ = triton_add_func(x0, x1, N) if dtype != torch.bool else triton_or_func(x0, x1, N)

    profiler_wrapper(wrapper)


@triton.jit
def _profiling_unused_arguments(unused_head, x, n, unused_middle, BLOCK: tl.constexpr, y, state, unused_tail):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < n, other=0)
    previous = tl.load(state + offsets, offsets < n, other=0)
    tl.store(y + offsets, values + 1, offsets < n)
    tl.store(state + offsets, previous + values, offsets < n)


@pytest.mark.parametrize("taskqueue", [False, True])
def test_profiler_preserves_unused_pointer_positions(tmp_path, monkeypatch, taskqueue):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    x = torch.arange(128, dtype=torch.float32, device="npu")
    y = torch.empty((16, 8), dtype=torch.float32, device="npu")
    state = torch.zeros((4, 32), dtype=torch.float32, device="npu")
    unused = [torch.empty(shape, device="npu") for shape in [(3, 5), (7, 9), (11, )]]
    args = (unused[0], x, 128, unused[1], 128, y, state, unused[2])
    compiled = _profiling_unused_arguments.warmup(*args, grid=(1, ))
    compiled._init_handles()
    # Bind a launcher under this test's taskqueue setting even on a JIT cache hit.
    instance = driver.NPULauncher(compiled.src, compiled.metadata)
    stream = driver.NPUDriver().get_current_stream()

    def run(arguments):
        instance(1, 1, 1, stream, compiled.function, compiled.packed_metadata, None, None, None, *arguments)

    torch.npu.synchronize()

    # Warm up without profiling, then reuse the same callable with new shapes.
    run(args)
    torch.npu.synchronize()
    state.zero_()
    torch.npu.synchronize()
    reshaped_args = (unused[0], x.view(8, 16), 128, unused[1], 128, y.view(128), state.view(2, 64), unused[2])
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(tmp_path))):
        run(args)
        run(reshaped_args)
        torch.npu.synchronize()

    assert torch.equal(y.flatten().cpu(), x.cpu() + 1)
    assert torch.equal(state.flatten().cpu(), x.cpu() * 2)
    reports = list(tmp_path.rglob("kernel_details.csv"))
    assert len(reports) == 1, f"Expected one profiler report, found {reports}"
    with reports[0].open(newline="") as report:
        rows = [row for row in csv.DictReader(report) if compiled.metadata.kernel_name in row["Name"]]
    assert len(rows) == 2, rows

    def shapes(value):
        # CANN keeps literal quotes around the shape list inside the CSV field.
        return tuple(tuple(int(dim) for dim in shape.split(",")) for shape in value.strip('"').split(";") if shape)

    reported = Counter((shapes(row["Input Shapes"]), shapes(row["Output Shapes"])) for row in rows)
    expected = Counter([
        (((128, ), (4, 32)), ((16, 8), (4, 32))),
        (((8, 16), (2, 64)), ((128, ), (2, 64))),
    ])
    assert reported == expected
    # Hidden workspace/lock arguments and the scalar/constexpr do not add slots.
    assert compiled.metadata.tensor_kinds == [-1, 0, -1, 1, 2, -1]
