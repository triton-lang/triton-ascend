# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
from triton.backends.ascend.block_batching import BatchRequest, analyze_request
from triton.backends.ascend.compiler import NPUOptions

pytestmark = pytest.mark.backend("native")


@pytest.mark.parametrize("factor", [0, -1, True, False, 1.0, "2", None])
def test_invalid_factor(factor):
    with pytest.raises(ValueError, match="block_batching_factor"):
        NPUOptions(arch="Ascend950", block_batching_factor=factor)


@pytest.mark.parametrize("axis", [-2, 3, True, False, 1.0, "0", None])
def test_invalid_axis(axis):
    with pytest.raises(ValueError, match="block_batching_axis"):
        NPUOptions(arch="Ascend950", block_batching_axis=axis)


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 6, 16, 2**40])
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_explicit_request_is_preserved(factor, axis):
    options = NPUOptions(arch="Ascend950", block_batching_factor=factor, block_batching_axis=axis)
    assert options.block_batching_factor == factor
    assert options.block_batching_axis == axis


def test_options_cache_identity():
    options = [
        NPUOptions(arch="Ascend950", block_batching_factor=f, block_batching_axis=a)
        for f in (1, 2, 3, 6)
        for a in (-1, 0, 1, 2)
    ]
    assert len({option.hash() for option in options}) == len(options)
    assert NPUOptions(arch="Ascend950").hash() == NPUOptions(arch="Ascend950", block_batching_factor=1,
                                                             block_batching_axis=-1).hash()


def test_disabled_and_unrepresentable_requests_do_not_enter_cpp():
    # An object with no binding methods ensures these paths are host-only.
    assert analyze_request(None, BatchRequest(), object())["compile_status"] == "DISABLED"
    report = analyze_request(None, BatchRequest(2**40), object())
    assert report["reason"] == "FACTOR_OUT_OF_RANGE"
    assert report["requested_factor"] == 2**40
    assert report["applied_factor"] == 1


def test_missing_cpp_binding_is_an_environment_error():
    with pytest.raises(RuntimeError, match=r"matching rebuilt Ascend C\+\+ extension"):
        analyze_request(None, BatchRequest(3, 0), object())
