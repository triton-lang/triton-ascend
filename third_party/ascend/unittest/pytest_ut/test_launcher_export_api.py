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
"""Exercise the versioned C ABI instead of inspecting generated C++ text.

Numerical execution through the legacy exported symbol is covered alongside
the Python entry in test_launcher_runtime.py.
"""
import ctypes as ct
from contextlib import contextmanager

import pytest

from triton.backends.ascend import driver, launcher


class Spec(ct.Structure):
    _fields_ = [("version", ct.c_uint32), ("struct_size", ct.c_uint32),
                ("flags", ct.c_uint64), ("workspace_size", ct.c_uint64),
                ("ordered_locks", ct.c_uint64), ("unordered_locks", ct.c_uint64),
                ("lock_init_value", ct.c_int64), ("participant_factor", ct.c_uint32),
                ("physical_blocks", ct.c_uint32), ("coalesce_factor", ct.c_uint32),
                ("coalesce_axis", ct.c_int32), ("task_type", ct.c_uint32),
                ("mix_ratio", ct.c_uint32), ("shared_mem_dynamic_size", ct.c_uint32)]


class ArgType(ct.Structure):
    _fields_ = [("kind", ct.c_uint32), ("dtype", ct.c_int32)]


class Request(ct.Structure):
    _fields_ = [("version", ct.c_uint32), ("struct_size", ct.c_uint32),
                ("kernel_name", ct.c_char_p), ("function", ct.c_void_p), ("stream", ct.c_void_p),
                ("grid", ct.c_int32 * 3), ("shapes_data", ct.POINTER(ct.c_int64)),
                ("shape_dims", ct.POINTER(ct.c_int)), ("tensor_kinds", ct.POINTER(ct.c_int)),
                ("num_tensors", ct.c_int)]


@pytest.fixture(scope="module")
def native_api():
    _, path = launcher.get_runtime(driver.NPUUtils().get_so_path())
    api = ct.CDLL(path)
    api.triton_npu_create_plan_v1.argtypes = [ct.POINTER(Spec), ct.POINTER(ArgType), ct.c_size_t,
                                            ct.POINTER(ct.c_char), ct.c_size_t]
    api.triton_npu_create_plan_v1.restype = ct.c_void_p
    api.triton_npu_destroy_plan_v1.argtypes = [ct.c_void_p]
    api.triton_npu_destroy_plan_v1.restype = None
    api.triton_npu_launch_v1.argtypes = [ct.c_void_p, ct.POINTER(Request), ct.POINTER(ct.c_void_p),
                                       ct.POINTER(ct.c_size_t), ct.c_size_t,
                                       ct.POINTER(ct.c_char), ct.c_size_t]
    api.triton_npu_launch_v1.restype = ct.c_int
    return api


def spec():
    return Spec(1, ct.sizeof(Spec), 0, 0, 0, 0, 0, 1, 20, 1, -1, 1, 0, 0)


def request():
    return Request(version=1, struct_size=ct.sizeof(Request), kernel_name=b"abi_contract",
                   grid=(ct.c_int32 * 3)(0, 1, 1))


@contextmanager
def plan(api, kinds, **changes):
    config = spec()
    for field, value in changes.items():
        setattr(config, field, value)
    types = (ArgType * len(kinds))(*(ArgType(kind, -1) for kind in kinds))
    error = ct.create_string_buffer(256)
    handle = api.triton_npu_create_plan_v1(ct.byref(config), types, len(types), error, len(error))
    assert handle, error.value
    try:
        yield handle
    finally:
        api.triton_npu_destroy_plan_v1(handle)


@pytest.mark.parametrize("field,value", [("version", 2), ("struct_size", 1), ("flags", 1 << 60),
                                         ("coalesce_factor", 0), ("coalesce_axis", -2), ("coalesce_axis", 3),
                                         ("participant_factor", 0), ("task_type", 0), ("task_type", 5),
                                         ("mix_ratio", 65536)])
def test_c_api_rejects_incompatible_or_invalid_plan(native_api, field, value):
    config = spec()
    setattr(config, field, value)
    error = ct.create_string_buffer(256)
    handle = native_api.triton_npu_create_plan_v1(ct.byref(config), None, 0, error, len(error))
    assert not handle
    assert error.value


@pytest.mark.parametrize("flags", [0, launcher.FFTS, launcher.PURE_SIMT, launcher.FFTS | launcher.PURE_SIMT])
@pytest.mark.parametrize("kinds", [[], [launcher.CONSTEXPR], [launcher.CONSTEXPR, launcher.CONSTEXPR]])
def test_c_api_empty_signature_and_constexpr(native_api, kinds, flags):
    with plan(native_api, kinds, flags=flags) as handle:
        req = request()
        error = ct.create_string_buffer(256)
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), None, None, 0, error, len(error)) == 0
        assert error.value == b""


@pytest.mark.parametrize("kind,value_type,value", [
    (launcher.I8, ct.c_int8, -7), (launcher.U8, ct.c_uint8, 250),
    (launcher.I16, ct.c_int16, -1234), (launcher.U16, ct.c_uint16, 65530),
    (launcher.I32, ct.c_int32, -34342), (launcher.U32, ct.c_uint32, 2**31 + 5),
    (launcher.I64, ct.c_int64, -(2**40) + 7), (launcher.U64, ct.c_uint64, 2**48 + 17),
    (launcher.F32, ct.c_float, 1.25), (launcher.F64, ct.c_double, -3.75),
    (launcher.POINTER, ct.c_void_p, 0x123456789ABCDEF0),
], ids=["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64", "pointer"])
def test_c_api_checks_argument_count_and_width(native_api, kind, value_type, value):
    with plan(native_api, [kind]) as handle:
        req = request()
        value = value_type(value)
        pointers = (ct.c_void_p * 1)(ct.addressof(value))
        sizes = (ct.c_size_t * 1)(ct.sizeof(value))
        error = ct.create_string_buffer(256)
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), pointers, sizes, 1, error, len(error)) == 0
        sizes[0] = ct.sizeof(value) + 1
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), pointers, sizes, 1, error, len(error)) != 0
        assert b"width" in error.value
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), pointers, sizes, 0, error, len(error)) != 0
        assert b"count" in error.value


def test_c_api_constexpr_does_not_consume_value_buffer(native_api):
    kinds = [launcher.CONSTEXPR, launcher.I16, launcher.CONSTEXPR, launcher.F32]
    with plan(native_api, kinds) as handle:
        req = request()
        values = (ct.c_int16(-1234), ct.c_float(1.25))
        pointers = (ct.c_void_p * 2)(*(ct.addressof(value) for value in values))
        sizes = (ct.c_size_t * 2)(*(ct.sizeof(value) for value in values))
        error = ct.create_string_buffer(256)
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), pointers, sizes, 2, error, len(error)) == 0
        assert error.value == b""


@pytest.mark.parametrize("grid", [(0, 3, 4), (3, -1, 4), (3, 4, 0)])
def test_c_api_nonpositive_grid_does_not_submit(native_api, grid):
    with plan(native_api, []) as handle:
        # No function or stream: an empty launch must return before device submission.
        req = request()
        req.grid[:] = grid
        error = ct.create_string_buffer(256)
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), None, None, 0, error, len(error)) == 0
        assert error.value == b""


@pytest.mark.parametrize("config,grid,message", [
    ({"coalesce_axis": 1, "coalesce_factor": 4}, (3, 17, 5), b"not divisible"),
    ({}, (65536, 65536, 1), b"uint32 block count"),
    ({}, (2**31 - 1, 2**31 - 1, 2**31 - 1), b"size overflow"),
], ids=["nondivisible", "uint32-blocks", "uint64-product"])
def test_c_api_invalid_grid_is_rejected_before_submission(native_api, config, grid, message):
    with plan(native_api, [], **config) as handle:
        req = request()
        req.grid[:] = grid
        error = ct.create_string_buffer(256)
        assert native_api.triton_npu_launch_v1(handle, ct.byref(req), None, None, 0, error, len(error)) != 0
        assert message in error.value


def test_c_api_request_version_and_bounded_error(native_api):
    with plan(native_api, []) as handle:
        req = request()
        req.version = 2
        error = (ct.c_char * 8)(*b"XXXXXXXX")
        ret = native_api.triton_npu_launch_v1(handle, ct.byref(req), None, None, 0, error, 4)
        assert ret != 0
        assert bytes(error)[3] == 0
        assert bytes(error)[4:] == b"XXXX"
