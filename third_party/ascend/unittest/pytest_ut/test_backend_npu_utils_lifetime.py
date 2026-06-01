from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
PYTEST_DIR = Path(__file__).resolve().parent


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


def test_runtime_lifecycle_case_explicitly_prints_release():
    runtime_case = PYTEST_DIR / "test_backend_npu_utils_runtime_lifetime.py"

    assert runtime_case.exists()
    src = runtime_case.read_text()

    assert 'print("[TEST] release workspace tensor handle", flush=True)' in src
    assert 'print("[TEST] release sync_block_lock tensor handle", flush=True)' in src
    assert "triton_release_retained_tensor(workspace_handle)" in src
    assert "triton_release_retained_tensor(sync_handle)" in src
    assert "[TRITON_NPU_TENSOR_LIFETIME] action=release kind=workspace" in src
    assert "[TRITON_NPU_TENSOR_LIFETIME] action=release kind=sync_block_lock" in src


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


def test_commit1_preserves_launcher_compatibility_boundaries():
    driver_src = (BACKEND_DIR / "driver.py").read_text()
    utils_src = (BACKEND_DIR / "utils.py").read_text()

    assert "void triton_launch_kernel(" in driver_src
    assert "const void* kernel_args[] = {" in driver_src
    assert "const size_t arg_sizes[] = {" in driver_src
    assert "rtKernelLaunch(func, blockNum, static_cast<void*>(launch_args.data()), launch_args.size(), NULL, stream);" in driver_src
    assert "rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);" not in driver_src
    assert "default_parallel = " not in utils_src
    assert 'return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "true").lower() in ("true", "1")' in utils_src


def test_commit5_restores_cpu_tensor_address_guard_only_for_device_pointers():
    driver_src = (BACKEND_DIR / "driver.py").read_text()

    assert "aclrtPointerGetAttributes" in driver_src
    assert "ACL_MEM_LOCATION_TYPE_DEVICE" in driver_src
    assert "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)" in driver_src
    assert "attributes.location.type != 4" not in driver_src
    assert "ACL_MEM_LOCATION_TYPE_HOST_NUMA" not in driver_src


def test_taskqueue_path_uses_stable_function_wrapper_and_symbol_guards():
    driver_src = (BACKEND_DIR / "driver.py").read_text()
    register_src = (BACKEND_DIR / "backend_register.py").read_text()

    assert "std::function<rtError_t()> launch_call = [=]() -> rtError_t" in driver_src
    assert "auto launch_call = [=]() -> rtError_t" not in driver_src
    assert "static bool npu_utils_init_attempted = false;" in driver_src
    assert "static bool npu_utils_init_ok = false;" in driver_src
    assert 'fprintf(stderr, "Error: required npu_utils symbols are unavailable\\\\n");' in driver_src
    assert "g_allocate_workspace ? g_allocate_workspace" in register_src
    assert "g_allocate_sync_block_lock ? g_allocate_sync_block_lock" in register_src
    assert 'if (!g_async_launch)' in register_src
    assert 'fprintf(stderr, "Error: triton_async_launch is unavailable\\\\n")' in register_src
