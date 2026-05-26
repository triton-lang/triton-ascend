import importlib.util
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


def _make_metadata():
    return SimpleNamespace(
        workspace_size=0,
        lock_init_value=0,
        lock_num=0,
        bs_task_type=0,
        mix_mode="aiv",
        compile_on_910_95=False,
        parallel_mode="",
        force_simt_only=False,
        debug=False,
    )


@patch.object(driver, "NPUUtils")
@patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False)
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B")
def test_generate_npu_wrapper_src_exposes_triton_launch_kernel(
    _mock_arch,
    _mock_ffts,
    _mock_disable_ffts,
    _mock_auto_map,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.generate_npu_wrapper_src(
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
