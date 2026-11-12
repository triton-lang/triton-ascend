# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
from test_block_batching_elementwise import parse
from block_batching_op_cases import BASIC_COMPLETION_OPS, case_ir
from triton._C.libtriton import ascend

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("op", BASIC_COMPLETION_OPS)
def test_remaining_basic_ops_are_observed_before_rewrite(tmp_path, op):
    module = parse(tmp_path, case_ir(op))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["operations"].get(op, 0)
    assert module.verify()
