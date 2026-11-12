"""Exact loop/local flattening for masked final K tiles, without mask erasure."""
import copy

import pytest
from triton.backends.ascend.block_batching_guard_v2 import ByteDomain, check_guard_v2

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def c(value):
    return ['const', value, 32]


def add(a, b):
    return ['addi', 32, a, b]


def mul(a, b):
    return ['muli', 32, a, b]


def tail_read():
    local = ['coord', 0, 4, 32]
    linear = add(mul(['iv', 0, 32], c(4)), local)
    offset = mul(add(mul(['pid', 0, 32], c(6)), linear), c(4))
    return dict(root=0, offset=offset, mask=['slt', linear, c(6), 32], shape=[4], bytes=4, write=False, volatile=False,
                pointer_checks=[offset], loops=[dict(id=0, bits=32, lower=c(0), upper=c(2), step=c(1))])


@pytest.mark.parametrize('length', [1, 4, 5, 6, 7, 8, 9])
def test_loop_tail_interval_matches_enumerated_original_accesses(length):
    read = tail_read()
    read['mask'][2] = c(length)
    actual = [
        4096 + p * 24 + (i * 4 + k) * 4 for p in range(2) for i in range(2) for k in range(4) if i * 4 + k < length
    ]
    interval = ByteDomain([], (2, 1, 1), read).interval(4096)
    assert interval == (min(actual), max(actual) + 4, False)


def test_loop_tail_does_not_create_a_false_alias_with_adjacent_output():
    read = tail_read()
    output = dict(root=1, offset=mul(add(mul(['pid', 0, 32], c(6)), ['coord', 0, 6, 32]), c(4)), mask=None, shape=[6],
                  bytes=4, write=True, volatile=False, pointer_checks=[], loops=[])
    spec = dict(version=2, axis=0, accesses=[read, output])
    assert check_guard_v2(spec, [4096, 4096 + 48], (2, 1, 1), 2)[0]
    assert not check_guard_v2(spec, [4096, 4096 + 44], (2, 1, 1), 2)[0]
    overlap = copy.deepcopy(output)
    overlap['offset'] = mul(add(mul(['pid', 0, 32], c(5)), ['coord', 0, 6, 32]), c(4))
    assert not check_guard_v2(dict(spec, accesses=[read, overlap]), [4096, 8192], (2, 1, 1), 2)[0]


def test_nonmatching_address_stride_keeps_conservative_full_loop_box():
    read = tail_read()
    read['offset'] = mul(add(mul(['pid', 0, 32], c(32)), add(mul(['iv', 0, 32], c(8)), ['coord', 0, 4, 32])), c(4))
    read['pointer_checks'] = [read['offset']]
    # Predicate covers i*4+k<6, but address uses i*8+k. No contiguous collapse.
    assert ByteDomain([], (2, 1, 1), read).interval(4096) == (4096, 4096 + (32 + 8 + 3) * 4 + 4, False)
