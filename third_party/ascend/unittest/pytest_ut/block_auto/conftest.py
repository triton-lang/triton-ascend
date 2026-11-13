"""CPU-only block-auto tests; device integration lives in pytest_ut itself."""
import pytest


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """Override the device fixture when collected with the full Ascend suite."""
