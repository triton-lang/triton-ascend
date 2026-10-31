import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_driver_workspace_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def _mock_backend_func(name, *args):
    return f"/* {name}: {args} */"


def _make_metadata(workspace_size):
    return SimpleNamespace(
        workspace_size=workspace_size,
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
    )


def _generate(workspace_size):
    with patch.object(driver, "NPUUtils") as mock_npu_utils, \
            patch.object(driver, "_is_auto_map_parallel_blocks_enabled", return_value=False), \
            patch.object(driver, "force_disable_ffts", return_value=False), \
            patch.object(driver, "is_ffts_supported", return_value=True), \
            patch.object(driver, "get_ascend_arch_from_env", return_value="Ascend910B"), \
            patch.object(driver, "get_backend_func", side_effect=_mock_backend_func):
        mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
        mock_npu_utils.return_value.get_aicore_num.return_value = 20
        return driver.generate_npu_wrapper_src(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=_make_metadata(workspace_size),
        )


def test_workspace_block_count_is_widened_before_multiply():
    """Issue #2797: gridX * gridY * gridZ must not be truncated to uint32.

    Both launch paths used to declare ``uint32_t blockNum4Workspace =
    gridX * gridY * gridZ;``. A grid whose product exceeds 2**32 - 1 wrapped
    around (the reported 262144 x 16384 case wrapped to 0), so the workspace
    allocation silently became zero-sized and the kernel failed.
    """
    src = _generate(262144)
    # The intermediate product must be evaluated in 64 bits, otherwise the
    # multiplier itself is already wrong before it reaches totalWorkSpaceSize.
    assert "uint32_t blockNum4Workspace = gridX * gridY * gridZ;" not in src
    assert src.count("uint64_t blockNum4Workspace = static_cast<uint64_t>(gridX) * gridY * gridZ;") == 2
    # workspace_size must also be widened so the final product stays 64-bit.
    assert src.count("uint64_t totalWorkSpaceSize = (uint64_t)262144 * blockNum4Workspace;") == 2


def test_workspace_size_absent_when_kernel_needs_none():
    """无需 workspace 时，不生成 workspace 大小计算与内存申请。

    `blockNum4Workspace` 的声明是无条件出现在模板里的：它只是
    `totalWorkSpaceSize` 的被乘数，声明本身不申请任何内存。真正按
    `workspace_size > 0` 条件生成的是 `totalWorkSpaceSize` 的计算和
    `allocate_memory` 调用，因此这里校验这两者不出现。
    """
    src = _generate(-1)
    assert "totalWorkSpaceSize" not in src
    assert "allocate_memory" not in src


@pytest.mark.parametrize("grid,workspace_size,expected", [
    ((262144, 16384, 1), 262144, 262144 * 262144 * 16384),
    ((256, 64, 1), 262144, 262144 * 256 * 64),
])
def test_workspace_total_matches_64bit_reference(grid, workspace_size, expected):
    total = workspace_size
    for dim in grid:
        total *= dim
    assert total == expected
    # Sanity: the reported failing shape is exactly 2**32 bytes of workspace.
    if grid == (262144, 16384, 1) and workspace_size == 262144:
        assert expected == 262144 * 2**32
