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
def _folded_argument_echo(out, first, middle, last, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Bound all writes even with an incorrect grid read from a broken launcher.
    active = (pid < 3) & (tl.program_id(1) == 0) & (tl.program_id(2) == 0)
    tl.store(out + pid * 6 + 0, first, active)
    tl.store(out + pid * 6 + 1, middle, active)
    tl.store(out + pid * 6 + 2, last + BLOCK, active)
    tl.store(out + pid * 6 + 3, tl.num_programs(0), active)
    tl.store(out + pid * 6 + 4, tl.num_programs(1), active)
    tl.store(out + pid * 6 + 5, tl.num_programs(2), active)


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
@pytest.mark.parametrize("folded", ["first", "last"])
def test_value_folded_arguments_preserve_values_and_grid(monkeypatch, tmp_path, taskqueue, folded):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    # Match Inductor: a folded scalar stays i32, and a tile constexpr is absent
    # from the signature altogether. Folding must precede ABI layout creation.
    signature = {"out": "*i32", "first": "i32", "middle": "i32", "last": "i32"}
    # ASTSource.hash() currently omits constant keys, so isolate specializations
    # that fold the same value at different argument positions.
    with triton.knobs.cache.scope():
        triton.knobs.cache.dir = str(tmp_path / "cache")
        compiled = triton.compile(
            ASTSource(_folded_argument_echo, signature=signature, constexprs={folded: 196, "BLOCK": 8}))
    values = {"first": 5, "middle": 7, "last": 11}
    values[folded] = 196
    output = torch.full((3, 6), -1, dtype=torch.int32, device="npu")
    expected = [[values["first"], values["middle"], values["last"] + 8, 3, 1, 1]] * 3
    instance = _native(compiled)
    args = (output, *values.values())
    grid = (3, 1, 1)
    _run(instance, compiled, args, grid=grid)
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected

    output.fill_(-2)
    module_spec = importlib.util.spec_from_file_location("__triton_launcher", instance.get_launcher_so_path())
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.launch(*grid,
                  driver.NPUDriver().get_current_stream(), compiled.function, compiled.packed_metadata, None, None,
                  None, *args)
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected

    output.fill_(-3)
    _c_launch(instance, compiled, [
        ctypes.c_void_p(output.data_ptr()), *(ctypes.c_int32(value) for name, value in values.items() if name != folded)
    ], grid=grid)
    torch.npu.synchronize()
    assert output.cpu().tolist() == expected


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
    output = torch.full((2 * grid[1] * 3, ), -1, dtype=torch.int32, device="npu")
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


def _workspace_case(seed=11):
    from test_workspace_usage import matmul_mul_kernel

    # Integer-valued fp16 inputs give an exact CPU oracle. A2 emits a nonzero
    # workspace requirement; CANN on 950 may omit the workspace callback.
    generator = torch.Generator().manual_seed(seed)
    m, n, k = 256, 192, 32
    a, b, c = [torch.randint(-2, 3, shape, generator=generator).half() for shape in ((m, k), (k, n), (m, n))]
    expected = (a.float() @ b.float() * c.float()).half()
    a, b, c = (value.npu() for value in (a, b, c))
    out = torch.empty_like(c)
    args = (a, b, c, out, m, n, k, *a.stride(), *b.stride(), *c.stride(), *out.stride())
    compiled = matmul_mul_kernel.warmup(*args, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, grid=(4, 3))
    compiled._init_handles()
    workspace = getattr(compiled.metadata, "workspace_size", 0)
    if not compiled.metadata.compile_on_910_95:
        assert workspace > 0
    return compiled, (*args, 64, 64, 32), expected


@pytest.mark.parametrize("taskqueue", [False, True])
def test_cube_vector_workspace(monkeypatch, record_property, taskqueue):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    compiled, args, expected = _workspace_case()
    instance = _native(compiled)
    workspace = getattr(compiled.metadata, "workspace_size", 0)
    assert instance.launch_spec.workspace_size == max(workspace, 0)
    record_property("workspace_bytes_per_block", workspace)
    _run(instance, compiled, args, grid=(4, 3, 1))
    torch.npu.synchronize()
    assert torch.equal(args[3].cpu(), expected)


def _workspace_instance(compiled):
    from types import SimpleNamespace

    # Exercise host ownership on 950 too, even when its device binary does not
    # consume workspace. The non-SIMT ABI always reserves the workspace slot.
    metadata = compiled.metadata._asdict()
    metadata["workspace_size"] = max(metadata.get("workspace_size", 0), 65536)
    return driver.NPULauncher(compiled.src, SimpleNamespace(**metadata))


@pytest.mark.parametrize("taskqueue", [False, True])
@pytest.mark.parametrize("sync_each", [False, True])
def test_workspace_is_released_after_execution(monkeypatch, record_property, taskqueue, sync_each):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    compiled, args, expected = _workspace_case()
    instance = _workspace_instance(compiled)
    record_property("device_workspace_bytes_per_block", getattr(compiled.metadata, "workspace_size", 0))
    record_property("launch_workspace_bytes_per_block", instance.launch_spec.workspace_size)
    _run(instance, compiled, args, grid=(4, 3, 1))
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    for _ in range(64):
        _run(instance, compiled, args, grid=(4, 3, 1))
        if sync_each:
            torch.npu.synchronize()
    torch.npu.synchronize()
    after = torch.npu.memory_allocated()
    record_property("retained_workspace_growth_bytes", after - before)
    assert torch.equal(args[3].cpu(), expected)
    # Allow minor framework bookkeeping, but not one workspace per invocation.
    # Check active allocations rather than allocator-reserved memory.
    assert after <= before + 1024**2


def test_workspace_streams_and_queued_plan_ownership(monkeypatch, record_property):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", "true")
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    cases, instances = [], []
    for index, stream in enumerate(streams):
        compiled, args, expected = _workspace_case(seed=11 + index)
        with torch.npu.stream(stream):
            stream_id = driver.NPUDriver().get_current_stream()
        cases.append((compiled, args, expected, stream_id))
        instances.append(_workspace_instance(compiled))
    torch.npu.synchronize()
    for instance, (compiled, args, _, stream_id) in zip(instances, cases):
        _run(instance, compiled, args, grid=(4, 3, 1), stream=stream_id)
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    for _ in range(32):
        for instance, (compiled, args, _, stream_id) in zip(instances, cases):
            # Submit from the default stream's context to explicit, distinct
            # launch streams. Workspace reuse must follow the launch stream.
            _run(instance, compiled, args, grid=(4, 3, 1), stream=stream_id)
        scratch = torch.empty(65536 * 12, dtype=torch.uint8, device="npu")
        scratch.fill_(165)
        del scratch
    del instance
    instances.clear()
    gc.collect()
    torch.npu.synchronize()
    after = torch.npu.memory_allocated()
    record_property("retained_workspace_growth_bytes", after - before)
    for _, args, expected, _ in cases:
        assert torch.equal(args[3].cpu(), expected)
    assert after <= before + 1024**2


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


@triton.jit
def _original_grid_echo(out, original_x, original_y):
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0) & (tl.program_id(2) == 0):
        tl.store(out + 0, original_x)
        tl.store(out + 1, original_y)
        tl.store(out + 2, tl.num_programs(0))
        tl.store(out + 3, tl.num_programs(1))
        tl.store(out + 4, tl.num_programs(2))


@pytest.mark.parametrize("taskqueue", [False, True])
@pytest.mark.parametrize("sequence", [
    [(1, 16, False)],
    [(0, 64, True)],
    [(1, 16, False), (0, 4, True)],
])
def test_dynamic_original_grid_abi_reaches_both_entries(monkeypatch, taskqueue, sequence):
    from types import SimpleNamespace
    from triton.backends.ascend.program_grid import apply_program_grid_transforms
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    # Explicit scalar parameters occupy the same two ABI slots as compiler-
    # inserted originalGridX/Y. This observes the launcher's bytes on device
    # independently of whether a particular optimization happens to match.
    compiled = triton.compile(
        ASTSource(_original_grid_echo, signature={"out": "*i32", "original_x": "i32", "original_y": "i32"},
                  constexprs={}), options={"rule_mask": 0})
    compiled._init_handles()
    contract = dict(
        version=2, extent_source="runtime_original_grid", hidden_extent_axes=[0, 1],
        hidden_argument_order=["originalGridX", "originalGridY"], hidden_argument_types=["i32", "i32"], transforms=[
            dict(order=i, kind="ceil_div", axis=axis, factor=factor, persistent_coverage=persistent,
                 grid_stride_abi_verified=persistent) for i, (axis, factor, persistent) in enumerate(sequence)
        ])
    metadata = compiled.metadata._replace(program_grid_mapping_applied=True, program_grid_transforms=contract,
                                          row_coalescing_applied=False, auto_blockify_enabled=False,
                                          ptsm_cap_authorized=any(t[2] for t in sequence))
    source = SimpleNamespace(fn=compiled.src.fn, signature={0: "*i32"})
    instance = driver.NPULauncher(source, metadata)
    output = torch.full((5, ), -1, dtype=torch.int32, device="npu")
    # Reuse one plan and one exported stub for changing, non-divisible grids.
    for grid in [(1, 1, 1), (4097, 17, 3), (129, 65, 1)]:
        transformed = apply_program_grid_transforms(grid, contract,
                                                    physical_core_count=instance.launch_spec.physical_blocks)
        expected = [*grid[:2], *transformed]
        output.fill_(-1)
        _run(instance, compiled, (output, ), grid=grid)
        torch.npu.synchronize()
        assert output.cpu().tolist() == expected
        output.fill_(-2)
        _c_launch(instance, compiled, [ctypes.c_void_p(output.data_ptr())], grid=grid)
        torch.npu.synchronize()
        assert output.cpu().tolist() == expected
