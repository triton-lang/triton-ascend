# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Enumerated oracles for gapped byte domains and affine loop envelopes."""
import copy
import itertools
import random

import pytest
from triton.backends.ascend.block_batching_guard import GuardFailure
from triton.backends.ascend.block_batching_guard_v2 import ByteDomain, check_guard_v2

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def c(value):
    return ['const', value, 64]


def add(a, b):
    return ['addi', 64, a, b]


def mul(a, b):
    return ['muli', 64, a, b]


def strided_access(strides=(256, 64, 8), *, write=True):
    p, r, s = strides
    offset = add(mul(['pid', 0, 32], c(p)), add(mul(['coord', 0, 2, 32], c(r)), mul(['coord', 1, 3, 32], c(s))))
    return dict(root=0, shape=[2, 3], offset=offset, mask=None, bytes=4, write=write, volatile=False,
                pointer_checks=[offset], loops=[])


def test_gapped_reads_and_writes_have_a_conservative_byte_interval():
    for write in (False, True):
        spec = strided_access(write=write)
        assert ByteDomain([], (2, 1, 1), spec).interval(4096) == (4096, 4436, write)
    read = strided_access(write=False)
    out = dict(root=1, shape=[], offset=c(0), mask=None, bytes=4, write=True)
    guard = dict(version=2, axis=0, accesses=[read, out])
    # A gap is deliberately NOT used to relax aliasing between accesses.
    assert check_guard_v2(guard, [4096, 4100], (1, 1, 1), 2) == (False, 'WRITE_ALIAS')
    assert check_guard_v2(guard, [4096, 8192], (1, 1, 1), 2)[0]


@pytest.mark.parametrize('strides,reason', [
    ((256, 64, 2), 'OVERLAPPING_BLOCK_WRITES'),
    ((256, 16, 8), 'OVERLAPPING_BLOCK_WRITES'),
    ((0, 64, 8), 'OVERLAPPING_BLOCK_WRITES'),
    ((256, 64, -8), 'NEGATIVE_STRIDE'),
])
def test_sparse_layout_still_rejects_overlap_and_negative_strides(strides, reason):
    with pytest.raises(GuardFailure, match=reason):
        ByteDomain([], (2, 1, 1), strided_access(strides)).interval(4096)


def test_gapped_write_proof_against_enumerated_bytes():
    rng = random.Random(202611041)
    accepted = 0
    for _ in range(400):
        strides = tuple(rng.randint(0, 96) * 4 for _ in range(3))
        spec = strided_access(strides)
        grid = (rng.randint(1, 4), 1, 1)
        try:
            interval = ByteDomain([], grid, spec).interval(4096)
        except GuardFailure:
            continue
        accepted += 1
        touched = set()
        for p, r, s in itertools.product(range(grid[0]), range(2), range(3)):
            addr = 4096 + p * strides[0] + r * strides[1] + s * strides[2]
            region = set(range(addr, addr + 4))
            assert touched.isdisjoint(region)
            assert interval[0] <= addr and addr + 4 <= interval[1]
            touched.update(region)
    assert accepted > 20


def varying_upper_access():
    offset = add(mul(['pid', 0, 32], c(2048)),
                 add(mul(['pid', 1, 32], c(256)), add(mul(['iv', 0, 32], c(32)), mul(['coord', 0, 4, 32], c(8)))))
    return dict(
        root=0, shape=[4], offset=offset, mask=None, bytes=4, write=True, volatile=False, pointer_checks=[offset],
        loops=[
            dict(id=0, bits=32, lower=['const', 0, 32], upper=['pid_rem', 1, 4, 32], step=['const', 1, 32]),
        ])


def test_nonmerged_program_loop_upper_is_proved_by_a_superset():
    spec = varying_upper_access()
    domain = ByteDomain([], (2, 4, 1), spec)
    interval = domain.interval(4096)
    touched = set()
    for p, q in itertools.product(range(2), range(4)):
        for iv, local in itertools.product(range(q % 4), range(4)):
            addr = 4096 + p * 2048 + q * 256 + iv * 32 + local * 8
            region = set(range(addr, addr + 4))
            assert touched.isdisjoint(region)
            touched.update(region)
            assert interval[0] <= addr and addr + 4 <= interval[1]
    # p_other=0 has zero iterations; an envelope may include those phantom
    # accesses, which can only cause conservative rejection, never acceptance.
    assert interval[0] < min(touched)
    assert domain.loop_counts == (3, )


def test_loop_bound_split_and_arithmetic_are_not_ignored():
    spec = varying_upper_access()
    with pytest.raises(GuardFailure, match='NON_RECTANGULAR_PROGRAM_SPLIT'):
        ByteDomain([], (2, 3, 1), spec)
    spec['loops'][0]['upper'] = ['addi', 32, ['pid_rem', 1, 4, 32], ['const', (1 << 31) - 1, 32]]
    with pytest.raises(GuardFailure, match='INDEX_OVERFLOW'):
        ByteDomain([], (2, 4, 1), spec)


def test_loop_envelope_does_not_relax_vv_axis_absorption_or_aliases():
    spec = varying_upper_access()
    invalid = copy.deepcopy(spec)
    invalid['loops'][0]['lower'] = ['pid', 0, 32]
    invalid['loops'][0]['step'] = ['grid', 0, 32]
    with pytest.raises(GuardFailure, match='NON_UNIFORM_LOOP_DOMAIN'):
        ByteDomain([], (2, 4, 1), invalid)
    output = strided_access()
    output['root'] = 1
    assert not check_guard_v2(dict(version=2, axis=0, accesses=[spec, output]), [4096, 4096], (2, 4, 1), 2)[0]


def test_all_zero_trip_envelopes_have_no_effect():
    spec = varying_upper_access()
    spec['loops'][0]['upper'] = ['subi', 32, ['pid_rem', 1, 4, 32], ['const', 3, 32]]
    assert ByteDomain([], (2, 4, 1), spec).interval(4096) is None
