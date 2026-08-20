import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_driver_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def _mock_backend_func(name, *args):
    return f"/* {name}: {args} */"


def _make_metadata():
    return SimpleNamespace(
        target=driver.GPUTarget("npu", "Ascend910B3", 0),
        workspace_size=0,
        lock_init_value=0,
        lock_num=0,
        bs_task_type=0,
        mix_mode="aiv",
        shared=0,
        compile_on_910_95=False,
        parallel_mode="",
        force_simt_only=False,
        debug=False,
        coalesce_factor=1,
        coalesce_axis=-1,
        coalesce_grid_ceil_div=False,
        has_auto_blockify_blacklist_op=False,
        row_coalescing_applied=False,
        enable_auto_blockify=None,
    )


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_exposes_triton_launch_kernel(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32", 2: "i32"},
        metadata=_make_metadata(),
    )

    assert 'void triton_launch_kernel(' in src
    assert 'const void* const* kernel_args, const size_t* arg_sizes, int num_args' in src
    assert 'std::vector<std::vector<char>> copied_kernel_args;' in src
    assert 'std::vector<size_t> launch_arg_sizes;' in src
    assert 'std::vector<char> launch_args(total_size, 0);' in src
    assert 'memcpy(launch_args.data() + grid_offset, &gridX, sizeof(int32_t));' in src
    _mock_ffts.assert_called_once_with("Ascend910B3")


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_resolves_npu_utils_from_active_cache_root(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    cache_key = "NPU_UTILS_CACHE_KEY"

    def make_utils(cache_root):
        return SimpleNamespace(
            get_aivector_core_num=lambda: 40,
            get_aicore_num=lambda: 20,
            npu_utils_mod=SimpleNamespace(__file__=f"{cache_root}/{cache_key}/npu_utils.so"),
        )

    producer_utils = make_utils("/producer/cache")
    consumer_utils = make_utils("/consumer/cache")
    # make_launcher currently reads NPUUtils once for core counts and once for
    # the loaded module path.
    mock_npu_utils.side_effect = [producer_utils, producer_utils, consumer_utils, consumer_utils]

    producer_src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=_make_metadata(),
    )
    consumer_src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=_make_metadata(),
    )

    assert producer_src == consumer_src
    assert "/producer/cache" not in producer_src
    assert "/consumer/cache" not in consumer_src
    assert '#include <cstdlib>' in producer_src
    assert 'const char* cache_root = std::getenv("TRITON_CACHE_DIR");' in producer_src
    assert f'npu_utils_path = std::string(cache_root) + "/{cache_key}/npu_utils.so";' in producer_src
    assert 'const char* triton_home = std::getenv("TRITON_HOME");' in producer_src
    assert f'npu_utils_path = std::string(base) + "/.triton/cache/{cache_key}/npu_utils.so";' in producer_src


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_shrinks_coalesced_grid_for_both_launch_paths(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.coalesce_factor = 16
    metadata.coalesce_axis = 1

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert src.count("gridY = gridY / 16;") == 2
    assert src.count("ChunkCoalescing: grid[1] not divisible by coalesce_factor 16") == 2


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_uses_ceil_div_for_row_coalescing(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.coalesce_factor = 4
    metadata.coalesce_axis = 2
    metadata.coalesce_grid_ceil_div = True

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert src.count("gridZ = (gridZ + 4 - 1) / 4;") == 2
    assert "ChunkCoalescing: grid[2] not divisible" not in src


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_enables_91095_simt_for_sls_mixed_parallel_mode(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    """SLS-created indirect ops keep the original mixed-SIMT launch path."""
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.compile_on_910_95 = True
    metadata.parallel_mode = "mix_simd_simt"
    metadata.shared_mem_dynamic_size = 221184

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    # Both the ABI and local generated launch paths use the 910_95 SIMT launch
    # API only when the T2L result advertises SIMT in parallel_mode.
    assert src.count("aclrtLaunchKernelWithHostArgs") == 2


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_block_cap_uses_only_env_and_blacklist(
    _mock_backend_func_patch,
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    """The launcher cap is the 895 E && !B contract, independent of Row."""
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    cap = "blockNum = std::min(blockNum, (uint32_t)40);"

    # E = TRITON_ALL_BLOCKS_PARALLEL, B = blacklist, R = Row coalescing.
    # The Row result must not leak into the launcher predicate: only E && !B
    # controls whether both generated launch paths contain the physical-core
    # cap.  (O is intentionally absent here; it is a compiler argv option.)
    for env_enabled, blacklisted, row_applied in product((False, True), (False, True), (False, True)):
        metadata = _make_metadata()
        metadata.row_coalescing_applied = row_applied
        metadata.has_auto_blockify_blacklist_op = blacklisted
        with patch.object(
                driver,
                "_is_auto_map_parallel_blocks_enabled",
                return_value=env_enabled,
        ):
            src = driver.make_launcher(
                constants={},
                signature={0: "*fp32", 1: "*fp32"},
                metadata=metadata,
            )
        case = f"E={env_enabled}, B={blacklisted}, R={row_applied}"
        expected_per_launch_path = 1 if env_enabled and not blacklisted else 0
        c_abi_launch, cpp_launch = src.split("static void _launch(", maxsplit=1)
        assert c_abi_launch.count(cap) == expected_per_launch_path, case
        assert cpp_launch.count(cap) == expected_per_launch_path, case


@patch("importlib.util.module_from_spec")
@patch("importlib.util.spec_from_file_location")
@patch.object(driver, "make_npu_launcher_stub", return_value="/tmp/fake_launcher.so")
@patch.object(driver, "make_launcher", return_value="// wrapper src")
@patch.object(driver, "generate_npu_header_src", return_value="// header src")
def test_npu_launcher_exposes_launcher_so_path(
    mock_header_src,
    mock_wrapper_src,
    mock_launcher_stub,
    mock_spec_from_file_location,
    mock_module_from_spec,
):
    fake_module = SimpleNamespace(launch=object())
    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    mock_module_from_spec.return_value = fake_module
    mock_spec_from_file_location.return_value = fake_spec
    src = SimpleNamespace(
        constants={"input_ptr": 1},
        signature={"input_ptr": "*fp32", "numel": "i32"},
        fn=SimpleNamespace(arg_names=["input_ptr", "numel"]),
    )
    metadata = _make_metadata()

    launcher = driver.NPULauncher(src, metadata)

    assert launcher.so_launcher_path == "/tmp/fake_launcher.so"
    assert mock_launcher_stub.call_count == 1
    assert launcher.get_launcher_so_path() == "/tmp/fake_launcher.so"
    assert mock_header_src.call_count == 1
    assert mock_wrapper_src.call_count == 1
    assert mock_launcher_stub.call_count == 1
    mock_wrapper_src.assert_called_with(
        {0: 1},
        {0: "*fp32", 1: "i32"},
        metadata,
    )
    mock_launcher_stub.assert_called_with("// header src", "// wrapper src", False)
