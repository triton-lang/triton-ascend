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

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_driver_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def _make_launcher(monkeypatch, global_scratch_size, global_scratch_align, launch):
    monkeypatch.setenv("TRITON_COMPILE_ONLY", "false")
    monkeypatch.setenv("TRITON_REGISTER_TENSOR_MSPROF", "false")
    metadata = SimpleNamespace(
        mix_mode="aiv",
        shared=0,
        global_scratch_size=global_scratch_size,
        global_scratch_align=global_scratch_align,
    )
    fake_module = SimpleNamespace(launch=launch)
    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    with patch.object(driver.NPULauncher, "_make_launcher_stub_path", return_value="/tmp/fake_launcher.so"), \
            patch("importlib.util.spec_from_file_location", return_value=fake_spec), \
            patch("importlib.util.module_from_spec", return_value=fake_module):
        src = SimpleNamespace(fn=SimpleNamespace(arg_names=[]), signature={})
        return driver.NPULauncher(src, metadata)


def _mock_backend_func(name, *args):
    return f"/* {name}: {args} */"


def _make_launcher_source(monkeypatch, *, force_simt_only, global_scratch_size=0, workspace_size=0):
    metadata = SimpleNamespace(
        target=driver.GPUTarget("npu", "Ascend910B", 0),
        workspace_size=workspace_size,
        lock_init_value=0,
        lock_num=0,
        bs_task_type=0,
        mix_mode="aiv",
        shared=0,
        global_scratch_size=global_scratch_size,
        global_scratch_align=1,
        compile_on_910_95=False,
        parallel_mode="simt" if force_simt_only else "",
        force_simt_only=force_simt_only,
        shared_mem_dynamic_size=122880,
        debug=False,
        coalesce_factor=1,
        coalesce_axis=-1,
    )
    monkeypatch.setenv("TRITON_DEVICE_PRINT", "true")
    monkeypatch.setattr(driver, "extract_device_print_code_from_cann", lambda: "/* print stub */")
    with patch.object(driver, "NPUUtils") as mock_npu_utils, \
            patch.object(driver, "get_backend_func", side_effect=_mock_backend_func), \
            patch.object(driver, "is_ffts_supported", return_value=False), \
            patch.object(driver, "force_disable_ffts", return_value=False), \
            patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False):
        mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
        mock_npu_utils.return_value.get_aicore_num.return_value = 20
        return driver.make_launcher({}, {0: "*fp32"}, metadata)


def _exported_launcher_source(src):
    start = src.index("void triton_launch_kernel(")
    return src[start:src.index("static void _launch(", start)]


def _assert_order(text, *items):
    offsets = [text.index(item) for item in items]
    assert offsets == sorted(offsets)


def test_npu_launcher_allocates_global_scratch(monkeypatch):
    buffer = object()
    allocate = Mock(return_value=buffer)
    allocator = Mock()
    allocator.get.return_value = allocate
    monkeypatch.setattr(driver._allocation, "_allocator", allocator)
    launch = Mock(return_value=0)
    launcher = _make_launcher(monkeypatch, global_scratch_size=64, global_scratch_align=16, launch=launch)
    packed_metadata = {"hash": "h"}

    launcher(2, 3, 4, 99, 123, packed_metadata, None, None, None, "kernel-arg")

    allocate.assert_called_once_with(1536, 16, 99)
    launch.assert_called_once_with(
        2,
        3,
        4,
        99,
        123,
        buffer,
        None,
        packed_metadata,
        None,
        None,
        None,
        "kernel-arg",
    )


def test_npu_launcher_skips_zero_sized_global_scratch(monkeypatch):
    allocator = Mock()
    monkeypatch.setattr(driver._allocation, "_allocator", allocator)
    launch = Mock(return_value=0)
    launcher = _make_launcher(monkeypatch, global_scratch_size=0, global_scratch_align=1, launch=launch)
    packed_metadata = {"hash": "h"}

    launcher(1, 1, 1, 99, 123, packed_metadata, None, None, None)

    allocator.get.assert_not_called()
    launch.assert_called_once_with(1, 1, 1, 99, 123, None, None, packed_metadata, None, None, None)


def test_npu_launcher_skips_global_scratch_for_empty_grid(monkeypatch):
    allocator = Mock()
    monkeypatch.setattr(driver._allocation, "_allocator", allocator)
    launch = Mock(return_value=0)
    launcher = _make_launcher(monkeypatch, global_scratch_size=64, global_scratch_align=16, launch=launch)
    packed_metadata = {"hash": "h"}

    launcher(0, 3, 4, 99, 123, packed_metadata, None, None, None)

    allocator.get.assert_not_called()
    launch.assert_called_once_with(0, 3, 4, 99, 123, None, None, packed_metadata, None, None, None)


def test_make_launcher_threads_scratch_through_pure_simt_abi(monkeypatch):
    src = _make_launcher_source(monkeypatch, force_simt_only=True)

    parse = src[src.index("if(!PyArg_ParseTuple("):]
    assert 'args, "iiiKKOOOOOOO"' in parse
    _assert_order(parse, "&global_scratch_obj", "&profile_scratch_obj", "&packedMetadata")

    conversions = src[src.index("void *global_scratch = 0;"):src.index("// get kernel_name")]
    _assert_order(
        conversions,
        "global_scratch_obj != Py_None",
        "getPointer(global_scratch_obj, -1)",
        "profile_scratch_obj != Py_None",
        "getPointer(profile_scratch_obj, -1)",
    )

    internal = src[src.index("static void _launch("):]
    signature = internal[:internal.index(") {")]
    _assert_order(signature, "global_scratch", "profile_scratch", "arg0")

    struct_start = internal.index("struct __attribute__((packed))")
    initializer_start = internal.index("} args = {", struct_start)
    declarations = internal[struct_start:initializer_start]
    initializer = internal[initializer_start:internal.index("};", initializer_start)]
    _assert_order(declarations, "arg0", "gridX", "gridY", "gridZ", "global_scratch", "profile_scratch", "DTData")
    _assert_order(
        initializer,
        "static_cast<void*>(arg0)",
        "static_cast<int32_t>(gridX)",
        "static_cast<void*>(global_scratch)",
        "static_cast<void*>(profile_scratch)",
        "static_cast<void*>(DTData)",
    )

    call_start = src.index("_launch(kernelName")
    call = src[call_start:src.index(");", call_start)]
    _assert_order(call, "global_scratch", "profile_scratch", "ptr_info0.dev_ptr")

    exported = _exported_launcher_source(src)
    _assert_order(exported, "global_scratch_offset", "profile_scratch_offset", "dtdata_offset")


def test_make_launcher_omits_scratch_from_non_simt_device_layout(monkeypatch):
    src = _make_launcher_source(monkeypatch, force_simt_only=False)

    parse = src[src.index("if(!PyArg_ParseTuple("):]
    _assert_order(parse, "&global_scratch_obj", "&profile_scratch_obj", "&packedMetadata")

    internal = src[src.index("static void _launch("):]
    struct_start = internal.index("struct __attribute__((packed))")
    initializer_start = internal.index("} args = {", struct_start)
    declarations = internal[struct_start:initializer_start]
    assert "DTData" in declarations
    assert "global_scratch" not in declarations
    assert "profile_scratch" not in declarations

    exported = _exported_launcher_source(src)
    assert "dtdata_offset" in exported
    assert "global_scratch_offset" not in exported
    assert "profile_scratch_offset" not in exported


def test_exported_launcher_rejects_nonzero_global_scratch_before_setup(monkeypatch):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", "true")
    src = _make_launcher_source(
        monkeypatch,
        force_simt_only=True,
        global_scratch_size=64,
        workspace_size=32,
    )
    exported = _exported_launcher_source(src)
    header = exported.split("{", 1)[0]
    assert "global_scratch" not in header

    diagnostic = 'fprintf(stderr, "Error: triton_launch_kernel does not support nonzero global scratch\\n");'
    diagnostic_offset = exported.index(diagnostic)
    guard_return = exported.index("return;", diagnostic_offset)
    assert diagnostic_offset < guard_return
    for setup_marker in ("tensorShapes", "allocate_memory", "DebugTunnel::Open", "aclrtLaunchKernelWithHostArgs("):
        assert guard_return < exported.index(setup_marker)
