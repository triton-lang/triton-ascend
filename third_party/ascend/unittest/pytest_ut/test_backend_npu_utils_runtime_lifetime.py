import ctypes

import pytest


def _load_retained_tensor_api():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
    from triton.backends.ascend.driver import NPUUtils

    try:
        torch.npu.set_device(0)
        torch.empty((1,), device="npu")
    except Exception as exc:
        pytest.skip(f"NPU runtime is unavailable: {exc}")

    lib = ctypes.CDLL(NPUUtils().npu_utils_mod.__file__)
    lib.triton_allocate_workspace.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.triton_allocate_workspace.restype = ctypes.c_void_p
    lib.triton_allocate_sync_block_lock.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.triton_allocate_sync_block_lock.restype = ctypes.c_void_p
    lib.triton_release_retained_tensor.argtypes = [ctypes.c_void_p]
    lib.triton_release_retained_tensor.restype = None
    return torch, get_raw_stream, lib


def test_npu_utils_prints_retained_tensor_release_lifecycle(capfd):
    torch, get_raw_stream, lib = _load_retained_tensor_api()

    workspace_handle = ctypes.c_void_p()
    print("[TEST] create workspace tensor space", flush=True)
    workspace_ptr = lib.triton_allocate_workspace(4096, ctypes.byref(workspace_handle))
    assert workspace_ptr
    assert workspace_handle.value

    print("[TEST] release workspace tensor handle", flush=True)
    lib.triton_release_retained_tensor(workspace_handle)

    sync_handle = ctypes.c_void_p()
    print("[TEST] create sync_block_lock tensor space", flush=True)
    sync_ptr = lib.triton_allocate_sync_block_lock(
        64,
        ctypes.c_void_p(get_raw_stream(0)),
        ctypes.byref(sync_handle),
    )
    assert sync_ptr
    assert sync_handle.value

    print("[TEST] release sync_block_lock tensor handle", flush=True)
    lib.triton_release_retained_tensor(sync_handle)
    torch.npu.synchronize()

    captured = capfd.readouterr()
    text = captured.out + captured.err
    assert "[TEST] release workspace tensor handle" in text
    assert "[TEST] release sync_block_lock tensor handle" in text
    assert "[TRITON_NPU_TENSOR_LIFETIME] action=create kind=workspace" in text
    assert "[TRITON_NPU_TENSOR_LIFETIME] action=release kind=workspace" in text
    assert "[TRITON_NPU_TENSOR_LIFETIME] action=create kind=sync_block_lock" in text
    assert "[TRITON_NPU_TENSOR_LIFETIME] action=release kind=sync_block_lock" in text
