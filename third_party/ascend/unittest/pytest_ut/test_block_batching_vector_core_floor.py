"""AIV-only batching must preserve the available vector-core task domain."""
from types import SimpleNamespace
import pytest
from triton.backends.ascend import block_batching as batching
from triton.runtime import driver

pytestmark = pytest.mark.backend("native")


@pytest.mark.parametrize('aic,aiv', [(28, 56), (16, 32), (8, 16)])
def test_vector_kernel_floor_uses_vector_cores(monkeypatch, aic, aiv):
    utils = SimpleNamespace(get_aicore_num=lambda: aic, get_aivector_core_num=lambda: aiv)
    monkeypatch.setattr(driver, '_active', SimpleNamespace(utils=utils))
    monkeypatch.delenv('TA_BLOCK_BATCHING_CORE_FLOOR', raising=False)
    assert batching._core_floor() == aiv
