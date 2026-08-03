from pathlib import Path

import pytest

from triton.backends.ascend.runtime import costmodel_runtime


@pytest.mark.parametrize(
    ("arch", "filename"),
    [
        ("Ascend910_9362", "ascend_910b.json"),
        ("Ascend910_9589", "ascend_davidv100.json"),
    ],
)
def test_installed_wheel_contains_default_costmodel_profiles(arch, filename):
    module_path = Path(costmodel_runtime.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[4]
    if module_path.is_relative_to(repo_root):
        pytest.skip("installed-wheel packaging test")

    resolved = Path(costmodel_runtime._resolve_hardware_config(target_arch=arch))
    expected = module_path.parent.parent / "costmodel" / "configs" / filename
    assert resolved == expected
    assert resolved.is_file()
