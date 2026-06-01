from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"


def test_npu_utils_uses_releaseable_tensor_handles():
    src = (BACKEND_DIR / "npu_utils.cpp").read_text()

    assert "retained_tensors" not in src
    assert "retained_tensor_mutex" not in src
    assert "struct RetainedTensorHandle" in src
    assert "triton_release_retained_tensor" in src
    assert "new RetainedTensorHandle" in src
    assert "delete static_cast<RetainedTensorHandle*>" in src


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

    assert "aclrtPointerGetAttributes" not in driver_src
    assert "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)" not in driver_src
    assert "void triton_launch_kernel(" in driver_src
    assert "const void* kernel_args[] = {" in driver_src
    assert "const size_t arg_sizes[] = {" in driver_src
    assert "rtKernelLaunch(func, blockNum, static_cast<void*>(launch_args.data()), launch_args.size(), NULL, stream);" in driver_src
    assert "rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);" not in driver_src
    assert "default_parallel = " not in utils_src
    assert 'return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "true").lower() in ("true", "1")' in utils_src


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
