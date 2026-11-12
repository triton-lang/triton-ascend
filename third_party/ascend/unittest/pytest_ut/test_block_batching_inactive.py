# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import re
import pytest
from test_block_batching_memory import parse
from block_batching_inactive_cases import IEEE_OPS, inactive_ir
from triton._C.libtriton import ascend

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize('op', IEEE_OPS)
def test_unconstrained_fp_has_no_artificial_input_select(tmp_path, op):
    module = parse(tmp_path, inactive_ir(op))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    text = str(module)
    assert 'arith.select' not in text
    # One common live-mask broadcast is shared by load and store. Pure
    # arithmetic must not materialize unused mask tensors.
    assert len(re.findall(r'tt.broadcast .*: tensor<3x1xi1> -> tensor<3x4xi1>', text)) == 1
    assert module.verify()


@pytest.mark.parametrize('kind', ['log_to_int', 'exp_to_int'])
def test_fp_special_inactive_values_are_cut_before_integer_conversion(tmp_path, kind):
    module = parse(tmp_path, inactive_ir(kind))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    text = str(module)
    # log(0/negative) or overflowing exp is allowed in a dead lane, but must
    # never reach the FP-to-integer conversion unselected.
    selected = re.findall(r'(%\w+) = arith.select ', text)
    assert len(selected) == 1
    assert re.search(r'arith.fptosi ' + re.escape(selected[0]) + r'\s*:', text)
    assert module.verify()


def test_lift_reuse_cannot_escape_a_sibling_region(tmp_path):
    module = parse(tmp_path, inactive_ir('sibling_lifts'))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert module.verify()  # global memoization would violate dominance here
    text = str(module)
    assert 'scf.if' in text and 'else' in text


def test_ieee_policy_does_not_admit_nnan_promises(tmp_path):
    text = inactive_ir('math.exp').replace('"math.exp"(%v)', '"math.exp"(%v) {fastmath = #arith.fastmath<nnan>}')
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REJECTED'
    assert report['reason'] == 'UNSUPPORTED_OPERATION_ATTRIBUTE'
    assert str(module) == before
