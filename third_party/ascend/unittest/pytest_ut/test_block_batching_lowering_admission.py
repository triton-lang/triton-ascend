# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""VV admission follows the lowered memory path, including scan/reduce bodies."""
from copy import deepcopy

import pytest
from triton.backends.ascend import block_batching as batching

pytestmark = pytest.mark.backend('native')


def lowered_report(operation, loads=0, stores=0):
    return dict(
        owner='block_batching', cv_mode='disabled', lane_width=2,
        operations={'tt.load': 1, 'tt.store': 1, operation: 1}, layout={
            'lowering_observation':
            dict(scope='module', stage='ttadapter', indirect_load_ops=loads, indirect_store_ops=stores,
                 memref_copy_ops=3)
        })


@pytest.mark.parametrize('operation', ['arith.mulf', 'tt.reduce', 'tt.scan'])
@pytest.mark.parametrize('loads,stores', [(1, 0), (0, 1), (2, 1)])
def test_unsupported_lowered_memory_cannot_bypass_admission(operation, loads, stores):
    report = lowered_report(operation, loads, stores)
    before = deepcopy(report)
    assert batching.vv_memory_lowering_refusal(report) == 'VV_INDIRECT_MEMORY_NOT_SUPPORTED'
    assert report == before


@pytest.mark.parametrize('operation', ['tt.reduce', 'tt.scan', 'scf.for', 'arith.mulf'])
def test_structured_memory_keeps_existing_compute_support(operation):
    report = lowered_report(operation)
    report['lane_scans_lowered'] = 2 if operation == 'tt.scan' else 0
    assert batching.vv_memory_lowering_refusal(report) is None


@pytest.mark.parametrize('change', [dict(owner='none'), dict(lane_width=1), dict(cv_mode='independent_batch_v1')])
def test_lowering_refusal_is_owned_by_vv_batching(change):
    report = lowered_report('tt.reduce', loads=1, stores=1)
    report.update(change)
    assert batching.vv_memory_lowering_refusal(report) is None


@pytest.mark.parametrize('observation', [
    None, {}, {'stage': 'ttadapter'}, {'stage': 'ttadapter', 'scope': 'module', 'indirect_load_ops': 0}, {
        'stage': 'ttadapter', 'scope': 'module', 'indirect_load_ops': -1, 'indirect_store_ops': 0
    }, {'stage': 'ttir', 'scope': 'module', 'indirect_load_ops': 0, 'indirect_store_ops': 0},
    {'stage': 'ttadapter', 'scope': 'module', 'indirect_load_ops': False, 'indirect_store_ops': 0}
])
def test_missing_lowering_proof_is_not_zero_indirect_memory(observation):
    report = lowered_report('tt.reduce')
    report['layout']['lowering_observation'] = observation
    assert batching.vv_memory_lowering_refusal(report) == 'VV_MEMORY_LOWERING_NOT_PROVEN'
