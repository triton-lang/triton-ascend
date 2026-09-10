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
"""Numerical/ownership tests for the shared launcher on real Ascend devices."""
import ctypes
import gc
import importlib.util
import struct

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend import driver, launcher
from triton.compiler import ASTSource

pytestmark = pytest.mark.backend("torch_npu")


@triton.jit
def _copy_add(x, y, n, value, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(x + offsets, offsets < n, other=0)
    tl.store(y + offsets, data + value, offsets < n)


@triton.jit
def _scalar_echo(out, a, b, c, d):
    tl.store(out + 0, a.to(tl.int64))
    tl.store(out + 1, b.to(tl.int64))
    tl.store(out + 2, c.to(tl.int64))
    tl.store(out + 3, d.to(tl.int64))


@triton.jit
def _mixed_argument_echo(out, small, fractional, source, short, wide, BIAS: tl.constexpr, tail):
    tl.store(out + 0, small.to(tl.int64))
    tl.store(out + 1, fractional.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(out + 2, tl.load(source).to(tl.int64) + BIAS)
    tl.store(out + 3, short.to(tl.int64))
    tl.store(out + 4, wide.to(tl.int64))
    tl.store(out + 5, tail.to(tl.int64))


@triton.jit
def _grid_echo(out):
    x = tl.program_id(0)
    y = tl.program_id(1)
    z = tl.program_id(2)
    ny = tl.num_programs(1)
    nz = tl.num_programs(2)
    index = (x * ny + y) * nz + z
    tl.store(out + index, x * 100 + y * 10 + z)


def _native(compiled):
    # Initialize the binary handle, then create a plan under the current test's
    # environment. This avoids a JIT cache hiding taskqueue changes.
    compiled._init_handles()
    return driver.NPULauncher(compiled.src, compiled.metadata)


def _run(instance, compiled, args, grid=(1, 1, 1), stream=None):
    if stream is None:
        stream = driver.NPUDriver().get_current_stream()
    instance(*grid, stream, compiled.function, compiled.packed_metadata, None, None, None, *args)


def _c_launch(instance, compiled, values, grid=(1, 1, 1)):
    library = ctypes.CDLL(instance.get_launcher_so_path())
    launch = library.triton_launch_kernel
    launch.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    launch.restype = None
    pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(value) for value in values))
    sizes = (ctypes.c_size_t * len(values))(*(ctypes.sizeof(value) for value in values))
    launch(compiled.packed_metadata['kernel_name'].encode(), compiled.function,
           driver.NPUDriver().get_current_stream(), *grid, None, None, 0, None, pointers, sizes, len(values))
    return library


@pytest.mark.parametrize("taskqueue", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
@pytest.mark.parametrize("n", [257, 8193])
def test_shared_launcher_values_and_export(monkeypatch, taskqueue, dtype, n):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    grid = (triton.cdiv(n, 128), 1, 1)
    x = torch.arange(n, dtype=dtype, device="npu")
    y = torch.full_like(x, -1)
    compiled = _copy_add.warmup(x, y, n, 3, BLOCK=128, grid=(triton.cdiv(n, 128), ))
    instance = _native(compiled)
    args = (x, y, n, 3, 128)
    _run(instance, compiled, args, grid=grid)
    torch.npu.synchronize()
    assert torch.equal(y.cpu(), x.cpu() + 3)
    if n == 257 and dtype == torch.float32 and taskqueue:
        # The historical path is also importable as a Python extension.
        module_spec = importlib.util.spec_from_file_location("__triton_launcher", instance.get_launcher_so_path())
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        y.fill_(-3)
        module.launch(*grid,
                      driver.NPUDriver().get_current_stream(), compiled.function, compiled.packed_metadata, None, None,
                      None, *args)
        torch.npu.synchronize()
        assert torch.equal(y.cpu(), x.cpu() + 3)
    y.fill_(-2)
    # Only non-constexpr values belong to the C ABI. Both inputs and their
    # pointed-to temporary arrays go out of scope before synchronization.
    _c_launch(instance, compiled,
              [ctypes.c_void_p(x.data_ptr()),
               ctypes.c_void_p(y.data_ptr()),
               ctypes.c_int32(n),
               ctypes.c_int32(3)], grid=grid)
    gc.collect()
    torch.npu.synchronize()
    assert torch.equal(y.cpu(), x.cpu() + 3)


@pytest.mark.parametrize("unsigned", [False, True])
def test_mixed_scalar_abi_matches_device_and_c_entry(unsigned):
    kinds = ["u8", "u16", "u32", "u64"] if unsigned else ["i8", "i16", "i32", "i64"]
    values = [250, 65530, 2**31 + 5, 2**48 + 17] if unsigned else [-7, -1234, 34342, -(2**40) + 7]
    ctypes_types = [ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64
                    ] if unsigned else [ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64]
    signature = dict(zip(["out", "a", "b", "c", "d"], ["*i64", *kinds]))
    compiled = triton.compile(ASTSource(_scalar_echo, signature=signature, constexprs={}))
    output = torch.full((4, ), -1, dtype=torch.int64, device="npu")
    instance = _native(compiled)
    _run(instance, compiled, (output, *values))
    torch.npu.synchronize()
    assert output.cpu().tolist() == values
    output.fill_(-2)
    _c_launch(instance, compiled,
              [ctypes.c_void_p(output.data_ptr()), *(kind(value) for kind, value in zip(ctypes_types, values))])
    torch.npu.synchronize()
    assert output.cpu().tolist() == values


@pytest.mark.parametrize("taskqueue", [False, True])
def test_mixed_arguments_and_constexpr_reach_both_entries(monkeypatch, taskqueue):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    signature = dict(
        zip(_mixed_argument_echo.arg_names, ["*i64", "i8", "fp32", "*i32", "i16", "u64", "constexpr", "i32"]))
    compiled = triton.compile(ASTSource(_mixed_argument_echo, signature=signature, constexprs={"BIAS": 13}))
    output = torch.full((6, ), -1, dtype=torch.int64, device="npu")
    source = torch.tensor([17], dtype=torch.int32, device="npu")
    small, fractional, short, wide, tail = -7, 1.25, -1234, 2**48 + 17, -98765
    expected = [small, struct.unpack("=i", struct.pack("=f", fractional))[0], 30, short, wide, tail]
    instance = _native(compiled)
    _run(instance, compiled, (output, small, fractional, source, short, wide, 13, tail))
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected
    output.fill_(-2)
    values = [
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int8(small),
        ctypes.c_float(fractional),
        ctypes.c_void_p(source.data_ptr()),
        ctypes.c_int16(short),
        ctypes.c_uint64(wide),
        ctypes.c_int32(tail)
    ]
    _c_launch(instance, compiled, values)
    # C argument storage can be reused after submission, including with taskqueue enabled.
    for index in (1, 2, 4, 5, 6):
        values[index].value = 0
    del values
    gc.collect()
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected


@pytest.mark.parametrize("ceil_div", [False, True])
def test_coalesced_grid_reaches_both_entries(ceil_div):
    grid = (2, 17 if ceil_div else 16, 3)
    ny = 5 if ceil_div else 4
    # Keep the uncoalesced capacity so a missed transform fails by comparison,
    # without allowing an incorrect launcher to write past the allocation.
    output = torch.full((2 * grid[1] * 3,), -1, dtype=torch.int32, device="npu")
    compiled = _grid_echo.warmup(output, grid=(2, ny, 3), rule_mask=0)
    compiled._init_handles()
    # The echo kernel consumes the transformed grid directly. Override only the
    # launch policy to test coalescing independently of compiler transformations.
    metadata = compiled.metadata._replace(coalesce_axis=1, coalesce_factor=4, coalesce_grid_ceil_div=ceil_div,
                                          row_coalescing_applied=True, auto_blockify_enabled=False)
    instance = driver.NPULauncher(compiled.src, metadata)
    expected = [100 * x + 10 * y + z for x in range(2) for y in range(ny) for z in range(3)]
    _run(instance, compiled, (output, ), grid=grid)
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected + [-1] * (output.numel() - len(expected))
    output.fill_(-2)
    _c_launch(instance, compiled, [ctypes.c_void_p(output.data_ptr())], grid=grid)
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected + [-2] * (output.numel() - len(expected))


def test_three_dimensional_grid_and_empty_launch():
    output = torch.full((24, ), -1, dtype=torch.int32, device="npu")
    compiled = _grid_echo.warmup(output, grid=(2, 3, 4))
    instance = _native(compiled)
    _run(instance, compiled, (output, ), grid=(2, 3, 4))
    torch.npu.synchronize()
    assert output.cpu().tolist() == [100 * x + 10 * y + z for x in range(2) for y in range(3) for z in range(4)]
    output.fill_(-1)
    _run(instance, compiled, (output, ), grid=(0, 3, 4))
    torch.npu.synchronize()
    assert output.cpu().tolist() == [-1] * 24


def test_streams_and_queued_plan_ownership(monkeypatch):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", "true")
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    pairs = []
    for index, stream in enumerate(streams):
        with torch.npu.stream(stream):
            x = torch.arange(1024, dtype=torch.float32, device="npu")
            y = torch.empty_like(x)
            compiled = _copy_add.warmup(x, y, 1024, index + 2, BLOCK=128, grid=(8, ))
            instance = _native(compiled)
            for _ in range(50):
                _run(instance, compiled, (x, y, 1024, index + 2, 128), grid=(8, 1, 1))
            pairs.append((x, y, index + 2))
            del instance
    gc.collect()
    for stream in streams:
        stream.synchronize()
    for x, y, value in pairs:
        assert torch.equal(y.cpu(), x.cpu() + value)


def test_python_argument_and_hook_errors():
    x = torch.arange(16, dtype=torch.float32, device="npu")
    y = torch.full_like(x, -1)
    compiled = _copy_add.warmup(x, y, 16, 2, BLOCK=16, grid=(1, ))
    instance = _native(compiled)
    with pytest.raises(TypeError):
        _run(instance, compiled, (x, y))
    with pytest.raises((TypeError, AttributeError)):
        _run(instance, compiled, (object(), y, 16, 2, 16))
    with pytest.raises(OverflowError):
        _run(instance, compiled, (x, y, 16, 2, 16), grid=(2**40, 1, 1))

    def fail(metadata):
        raise RuntimeError("hook sentinel")

    with pytest.raises(RuntimeError, match="hook sentinel"):
        instance(1, 1, 1,
                 driver.NPUDriver().get_current_stream(), compiled.function, compiled.packed_metadata, None, fail, None,
                 x, y, 16, 2, 16)
    torch.npu.synchronize()
    assert torch.equal(y.cpu(), torch.full((16, ), -1.0))


def test_multiple_signatures_share_runtime():
    x = torch.arange(16, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    first = _copy_add.warmup(x, y, 16, 2, BLOCK=16, grid=(1, ))
    grid_out = torch.empty(1, dtype=torch.int32, device="npu")
    second = _grid_echo.warmup(grid_out, grid=(1, ))
    left, right = _native(first), _native(second)
    assert left._runtime_path == right._runtime_path
    assert left.launch is not right.launch
    assert left._so_launcher_path is None and right._so_launcher_path is None
    _run(left, first, (x, y, 16, 2, 16))
    _run(right, second, (grid_out, ))
    torch.npu.synchronize()
    assert torch.equal(y.cpu(), x.cpu() + 2)
    assert grid_out.item() == 0


@pytest.mark.parametrize("taskqueue", [False, True])
def test_cube_vector_workspace(monkeypatch, record_property, taskqueue):
    from test_workspace_usage import matmul_mul_kernel

    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    # Integer-valued fp16 inputs give an exact CPU oracle. A2 emits a nonzero
    # workspace requirement; CANN on 950 may omit the workspace callback.
    generator = torch.Generator().manual_seed(11)
    m, n, k = 256, 192, 32
    a, b, c = [torch.randint(-2, 3, shape, generator=generator).half() for shape in ((m, k), (k, n), (m, n))]
    expected = (a.float() @ b.float() * c.float()).half()
    a, b, c = (value.npu() for value in (a, b, c))
    out = torch.empty_like(c)
    compiled = matmul_mul_kernel[(4, 3)](a, b, c, out, m, n, k, *a.stride(), *b.stride(), *c.stride(), *out.stride(),
                                         BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.npu.synchronize()
    workspace = getattr(compiled.metadata, "workspace_size", 0)
    if not compiled.metadata.compile_on_910_95:
        assert workspace > 0
    assert _native(compiled).launch_spec.workspace_size == max(workspace, 0)
    record_property("workspace_bytes_per_block", workspace)
    assert torch.equal(out.cpu(), expected)


@triton.jit
def _nested_copy(out, values):
    x, pair = values
    n, bias = pair
    offsets = tl.arange(0, 16)
    data = tl.load(x + offsets, offsets < n, other=0)
    tl.store(out + offsets, data + bias, offsets < n)


def test_nested_tuple_arguments():
    x = torch.arange(16, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    _nested_copy[(1, )](out, (x, (16, 3)))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), x.cpu() + 3)


def test_compile_only_does_not_submit(monkeypatch):
    x = torch.arange(16, dtype=torch.float32, device="npu")
    out = torch.full_like(x, -1)
    compiled = _copy_add.warmup(x, out, 16, 2, BLOCK=16, grid=(1, ))
    monkeypatch.setenv("TRITON_COMPILE_ONLY", "true")
    instance = _native(compiled)
    _run(instance, compiled, (x, out, 16, 2, 16))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), torch.full((16, ), -1.0))
