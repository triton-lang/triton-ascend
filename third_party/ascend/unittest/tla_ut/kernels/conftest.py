"""Opt-in profiling for composite TLA kernels."""

import os

import pytest

from utils.tla_test_utils import get_stages


@pytest.fixture
def perf_output(request, tmp_path):
    if not get_stages()[2] or os.environ.get("TLA_RUN_PERF") != "1":
        pytest.skip("Set TLA_TEST_STAGE=run and TLA_RUN_PERF=1 to profile")
    if hasattr(request.config, "workerinput"):
        pytest.skip("Run performance tests without pytest-xdist")
    return tmp_path


@pytest.fixture(scope="module")
def npu_runtime():
    if not get_stages()[2]:
        pytest.skip("Device stage is disabled")
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("Requires Ascend 950")
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch, torch_npu
    finally:
        torch.npu.matmul.allow_hf32 = previous
