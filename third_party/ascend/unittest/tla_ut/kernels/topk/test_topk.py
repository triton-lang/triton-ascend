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
"""End-to-end radix selection checks against stable CPU sorting."""

import json
import os
from pathlib import Path
from statistics import median

import pytest
import torch
from triton.errors import TritonError
from triton.runtime import driver

from kernels.utils import load_kernel
from utils.tla_test_utils import check_kernel_binary, compile_offline, lower_offline


def _kernels():
    return load_kernel(Path(__file__).with_name("topk_kernel.py"))


def _inputs(rows, n, dtype, pattern):
    generator = torch.Generator().manual_seed(23)
    if pattern == "ties":
        x = torch.randint(-7, 8, (rows, n), generator=generator).to(dtype)
        if dtype == torch.float32:
            x[:, :4] = torch.tensor([-0.0, 0.0, -float("inf"), float("inf")])
    elif pattern == "equal":
        x = torch.full((rows, n), -3, dtype=dtype)
    elif not dtype.is_floating_point:
        limits = torch.iinfo(dtype)
        x = torch.randint(limits.min, limits.max + 1, (rows, n), generator=generator, dtype=dtype)
        x[:, :2] = torch.tensor([limits.min, limits.max], dtype=dtype)
    else:
        x = torch.randn((rows, n), generator=generator).to(dtype)
    return x


cases = pytest.mark.parametrize(("rows", "n", "k", "dtype", "largest", "pattern"), [
    (4, 33, 1, torch.float32, True, "ties"),
    (5, 257, 32, torch.float32, True, "random"),
    (3, 2048, 32, torch.int32, False, "random"),
    (4, 1031, 31, torch.float32, False, "ties"),
    (2, 129, 64, torch.int32, True, "equal"),
    (4, 4096, 64, torch.float32, True, "random"),
    (129, 1023, 10, torch.float32, True, "ties"),
    (4, 511, 65, torch.float16, True, "ties"),
    (5, 513, 129, torch.bfloat16, False, "random"),
    (4, 513, 33, torch.int8, False, "random"),
    (4, 511, 65, torch.uint8, True, "random"),
])


def test_compile():
    x = torch.empty((1, 256), dtype=torch.float32, device="npu")
    values = torch.empty((1, 32), dtype=x.dtype, device=x.device)
    indices = torch.empty(values.shape, dtype=torch.int32, device=x.device)
    check_kernel_binary(_kernels().native_topk.warmup(x, values, indices, 1, 256, 32, True, grid=(1, )))


def test_ir(tmp_path):
    module, context = compile_offline(
        _kernels().native_topk, {"X": "*fp32", "Values": "*fp32", "Indices": "*i32", "ROWS": "i32"}, {
            "N": 257, "K": 32, "LARGEST": True, "LAST_N": 0, "WIDE_INDEX": False, "Previous": None, "TILE_INDICES":
            False, "FinalValues": None, "FinalIndices": None, "TILES": 0
        }, output_dir=tmp_path)
    ir = str(lower_offline(module, context, output_dir=tmp_path))
    assert ir.count("ave.hir.chistv2") == 8
    assert "ave.hir.vsqz" in ir
    assert "ave.hir.vselr" in ir
    assert "ave.hir.vscatter" in ir
    assert "tt.histogram" not in ir


@cases
@pytest.mark.parametrize("wide_index", [False, True])
def test_run(rows, n, k, dtype, largest, pattern, wide_index):
    cpu = _inputs(rows, n, dtype, pattern)
    x = cpu.npu()
    values = torch.empty((rows, k), dtype=dtype, device=x.device)
    indices = torch.full((rows, k), -1, dtype=torch.int64 if wide_index else torch.int32, device=x.device)
    storage = indices.view(torch.int32) if wide_index else indices
    blocks = min(rows, driver.active.utils.get_aivector_core_num())
    _kernels().native_topk[(blocks, )](x, values, storage, rows, n, k, largest, WIDE_INDEX=wide_index)

    idx = indices.cpu().long()
    assert bool(((0 <= idx) & (idx < n)).all())
    torch.testing.assert_close(values.cpu(), cpu.gather(1, idx), rtol=0, atol=0)
    expected = torch.argsort(cpu, dim=1, descending=largest, stable=True)[:, :k]
    torch.testing.assert_close(idx.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tiled_tail(dtype):
    cpu = torch.randint(-7, 8, (10003, ), generator=torch.Generator().manual_seed(23)).to(dtype)
    k = 17
    actual = _kernels().topk(cpu.npu(), k)
    _check_matrix(cpu, torch.topk(cpu, k).values, actual, -1, True)


def test_tiled_concurrent_streams():
    cpu = torch.randn(131091, generator=torch.Generator().manual_seed(23)).to(torch.float16)
    aligned = cpu.npu()
    unaligned = torch.cat((cpu[:1], -cpu)).npu()[1:]
    inputs = ((aligned, cpu), (unaligned, -cpu))
    torch.npu.synchronize()
    streams = (torch.npu.Stream(), torch.npu.Stream())
    outputs = []
    topk = _kernels().topk
    for stream in streams + streams:
        with torch.npu.stream(stream):
            for x, source in inputs:
                for _ in range(2):
                    outputs.append((topk(x, 17), source))
    for stream in streams:
        stream.synchronize()
    for result, source in outputs:
        _check_matrix(source, torch.topk(source, 17).values, result, -1, True)


@pytest.mark.skipif(os.environ.get("TLA_TOPK_PERF") != "1", reason="opt-in serial NPU benchmark")
def test_perf():
    rows, n, k = 64, 4096, 32
    cpu = _inputs(rows, n, torch.float32, "random")
    x = cpu.npu()
    values = torch.empty((rows, k), device=x.device, dtype=x.dtype)
    indices = torch.empty((rows, k), device=x.device, dtype=torch.int32)
    ref_indices = torch.empty((rows, k), device=x.device, dtype=torch.int64)
    kernels = _kernels()

    blocks = min(rows, driver.active.utils.get_aivector_core_num())

    def native():
        kernels.native_topk[(blocks, )](x, values, indices, rows, n, k, True)

    def radix():
        kernels.radix_topk[(rows, )](x, values, indices, n, k, n, True)

    def torch_topk():
        torch.topk(x, k, sorted=False, out=(values, ref_indices))

    expected = torch.topk(cpu, k, dim=-1).values
    for name, call, out_indices in (("native", native, indices), ("radix", radix, indices), ("torch", torch_topk,
                                                                                             ref_indices)):
        values.fill_(float("nan"))
        out_indices.fill_(-1)
        call()
        idx = out_indices.cpu().long()
        assert bool(((0 <= idx) & (idx < n)).all())
        actual = values.cpu()
        torch.testing.assert_close(actual, cpu.gather(1, idx), rtol=0, atol=0)
        torch.testing.assert_close(actual.sort(descending=True).values, expected, rtol=0, atol=0)
        for _ in range(5):
            call()
        torch.npu.synchronize()
        samples = []
        for _ in range(5):
            start = torch.npu.Event(enable_timing=True)
            stop = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                call()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000 / 50)
        print(f"{name}: {median(samples):.2f} us/call, samples={samples}")


# (shape, dtype, k, dim, largest, low, high)
MATRIX = (
    ((1048576, ), torch.float16, 10, -1, True, -1, 1),
    ((2048, 2048), torch.int8, 100, -1, True, -127, 127),
    ((4096, 4096), torch.bfloat16, 64, -1, True, -3, 3),
    ((8192, 8192), torch.int32, 256, 0, True, -10000, 10000),
    ((4096, 8192), torch.int64, 512, -1, False, -100000, 100000),
    ((8192, 8192), torch.float32, 1024, 0, True, -1000, 1000),
    ((1023, 1023), torch.float16, 10, -1, True, -0.1, 0.1),
    ((1009, 1021), torch.float32, 50, 0, False, -1, 2),
    ((1537, 769), torch.bfloat16, 128, -1, True, -5, 10),
    ((363, 367, 373), torch.int32, 20, 1, True, -50, 100),
    ((2049, 513), torch.float16, 256, -1, True, -65504, 65504),
    ((3, 7, 13, 4001), torch.float32, 2000, -1, True, -88, 88),
    ((2, 7, 256, 256), torch.bfloat16, 7, 1, False, -0.01, 0.01),
    ((1000003, ), torch.float32, 100, -1, True, -1e38, 1e38),
    ((11, 13, 17, 67, 67), torch.float16, 17, 2, True, -6e-5, 6e-5),
    ((3, 7, 11, 13, 1013), torch.int64, 10, -1, True, 0, 3),
    ((512, 2049), torch.float32, 512, -1, True, -0.5, 0.5),
    ((255, 8193), torch.bfloat16, 128, 0, False, -1, 3),
    ((4097, 511), torch.uint8, 256, -1, True, 0, 255),
    ((2, 511, 2049), torch.float16, 100, 1, True, -3, 6),
)


def _matrix_input(case_id, shape, dtype, low, high):
    generator = torch.Generator().manual_seed(20261113 + case_id)
    if dtype.is_floating_point:
        return torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator).to(dtype)
    return torch.randint(low, high + 1, shape, dtype=dtype, generator=generator)


def _check_matrix(cpu, expected, output, dim, largest):
    values, indices = (tensor.cpu() for tensor in output)
    assert values.shape == indices.shape == expected.shape
    assert values.dtype == cpu.dtype and indices.dtype == torch.int64
    assert bool(((0 <= indices) & (indices < cpu.shape[dim])).all())
    assert bool(torch.diff(indices.sort(dim).values, dim=dim).ne(0).all())
    assert torch.equal(values, cpu.gather(dim, indices))
    assert torch.equal(values.sort(dim, descending=largest).values, expected)


def _latency(call):
    for _ in range(5):
        call()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)

    def group(iterations):
        start.record()
        for _ in range(iterations):
            call()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1000 / iterations

    iterations = min(80, max(10, int(30_000 / max(group(10), 1))))
    return [round(group(iterations), 3) for _ in range(7)]


def _implementations(shape, dtype, k, dim):
    n = shape[dim]
    if k == n:
        return ("full", )
    if n > 8192:
        return ("tiles-2048", "tiles-4096", "tiles-8192")
    if dtype in (torch.int8, torch.uint8):
        return ("radix", "byte")
    if dtype == torch.int64:
        return ("radix", )
    return ("radix", "sort")


@pytest.mark.skipif(os.environ.get("TLA_TOPK_MATRIX") != "1", reason="opt-in full Ascend 950 TopK matrix")
@pytest.mark.parametrize("case_id", range(1, 21))
def test_matrix(case_id):
    shape, dtype, k, dim, largest, low, high = MATRIX[case_id - 1]
    cpu = _matrix_input(case_id, shape, dtype, low, high)
    x = cpu.npu()
    expected = torch.topk(cpu, k, dim=dim, largest=largest, sorted=True).values
    torch_sorted = k == shape[dim]
    out = (x.new_empty(expected.shape), x.new_empty(expected.shape, dtype=torch.int64))
    kernels = _kernels()

    def torch_topk():
        return torch.topk(x, k, dim=dim, largest=largest, sorted=torch_sorted, out=out)

    def candidate(name):
        implementation, _, chunk = name.partition("-")
        return kernels.topk(x, k, dim, largest, implementation=implementation, chunk=int(chunk or 4096),
                            bounded_int64=dtype == torch.int64 and -(2**31) <= low <= high < 2**31)

    _check_matrix(cpu, expected, torch_topk(), dim, largest)
    baseline = _latency(torch_topk)
    print(
        "TOPK_RESULT",
        json.dumps({
            "case": case_id, "path": "torch", "median_us": median(baseline), "samples_us": baseline, "sorted":
            torch_sorted
        }), flush=True)
    paths = ("auto", )
    if os.environ.get("TLA_TOPK_COMPARE") == "1":
        paths += _implementations(shape, dtype, k, dim)
    for name in paths:
        try:
            _check_matrix(cpu, expected, candidate(name), dim, largest)
            samples = _latency(lambda: candidate(name))
        except (TritonError, RuntimeError, AssertionError, ValueError) as error:
            print("TOPK_RESULT", json.dumps({"case": case_id, "path": name, "error": str(error)}), flush=True)
            if name == "auto":
                raise
        else:
            print("TOPK_RESULT",
                  json.dumps({"case": case_id, "path": name, "median_us": median(samples), "samples_us": samples}),
                  flush=True)
