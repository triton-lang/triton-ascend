from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"


def test_npu_utils_uses_releaseable_tensor_handles():
    src = (BACKEND_DIR / "npu_utils.cpp").read_text()

    assert "retained_tensors" not in src
    assert "retained_tensor_mutex" not in src
    assert "struct RetainedTensorHandle" in src
    assert "triton_release_retained_tensor" in src
    assert "new RetainedTensorHandle" in src
    assert "delete retained;" in src


def test_npu_utils_logs_tensor_handle_lifecycle():
    src = (BACKEND_DIR / "npu_utils.cpp").read_text()

    assert "[TRITON_NPU_TENSOR_LIFETIME]" in src
    assert 'logRetainedTensor("create", retained);' in src
    assert 'logRetainedTensor("release", retained);' in src
    assert 'retainTensor(std::move(tensor), handle, "workspace", size)' in src
    assert 'retainTensor(std::move(tensor), handle, "sync_block_lock", size)' in src


def test_launcher_keeps_tensor_handles_until_launch_finishes():
    driver_src = (BACKEND_DIR / "driver.py").read_text()
    register_src = (BACKEND_DIR / "backend_register.py").read_text()

    assert "triton_release_retained_tensor_t" in driver_src
    assert 'dlsym(handle, "triton_release_retained_tensor")' in driver_src
    assert "std::shared_ptr<void>" in driver_src
    assert "release_npu_tensor_handle" in driver_src
    assert "workspace_handle" in driver_src
    assert "syncBlockLock_handle" in driver_src

    assert "g_allocate_workspace({size}, &workspace_handle)" in register_src
    assert "g_allocate_sync_block_lock({size}, {stream}, &syncBlockLock_handle)" in register_src


def test_npu_utils_load_binary_accepts_legacy_combined_kernel_name():
    driver_src = (BACKEND_DIR / "driver.py").read_text()

    assert "def load_binary(self, name, kernel, shared, device, mix_mode=None):" in driver_src
    assert 'name, mix_mode = name.rsplit("_", 1)' in driver_src
