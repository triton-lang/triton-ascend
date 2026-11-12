# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host tests for joint effects using the existing guard schema and code."""
from copy import deepcopy
import itertools

import pytest

from test_block_batching_guard_v2 import module as guard, arg, const, pid, coord, add, mul

pytestmark = pytest.mark.backend("native")


def joint_spec():
    base = mul(mul(["extsi", 64, pid(0), 32], arg(2)), const(4))
    accesses = []
    for column in range(3):
        offset = add(base, const(4 * column))
        accesses.append(
            dict(root=0, shape=[], bytes=4, write=True, offset=offset, mask=None, pointer_checks=[base, offset],
                 loops=[]))
    accesses[0]["shape"] = [3]
    accesses[0]["offset"] = add(base, mul(coord(0, 3), const(4)))
    accesses[1]["mask"] = accesses[2]["mask"] = ["false"]
    accesses.append(
        dict(root=1, shape=[], bytes=4, write=False, offset=mul(["extsi", 64, pid(0), 32], const(4)), mask=None,
             pointer_checks=[], loops=[]))
    return dict(version=2, axis=0, accesses=accesses)


@pytest.mark.parametrize("rows,stride,factor", itertools.product([1, 2, 7, 512], [0, 1, 2, 3, 5], [2, 3]))
def test_joint_injectivity_matches_explicit_byte_sets(rows, stride, factor):
    writes = [set(range(4 * (r * stride + c), 4 * (r * stride + c + 1))) for r in range(rows) for c in range(3)]
    disjoint = sum(map(len, writes)) == len(set().union(*writes))
    result = guard.check_guard(joint_spec(), [0x100000, 0x200000, stride], (rows, 1, 1), factor)
    assert result[0] == disjoint


def test_joint_proof_cache_tracks_stride_and_pointer_changes():
    cached = guard.RuntimeGuard(joint_spec(), 3)
    args = [0x100000, 0x200000, 3]
    assert cached(args, (7, 1, 1))[0]
    args[2] = 2
    assert cached(args, (7, 1, 1)) == (False, "OVERLAPPING_BLOCK_WRITES")
    args[2] = 3
    args[1] = args[0] + 4
    assert cached(args, (7, 1, 1)) == (False, "WRITE_ALIAS")
    args[1] = 0x200000
    assert cached(args, (7, 1, 1))[0]


def test_covered_members_still_check_original_arithmetic():
    spec = joint_spec()
    # A narrow original intermediate must not disappear when its effect is
    # covered by the leader. This also remains a cached scalar dependency.
    spec["accesses"][2]["pointer_checks"].append(["addi", 32, arg(3, 32), const(1, 32)])
    cached = guard.RuntimeGuard(spec, 3)
    assert cached([0x100000, 0x200000, 3, 0], (7, 1, 1))[0]
    assert cached([0x100000, 0x200000, 3, (1 << 31) - 1], (7, 1, 1)) == (False, "INDEX_OVERFLOW")


def test_layout_access_ids_remain_valid_for_covered_members():
    spec = deepcopy(joint_spec())
    spec["layout_checks"] = [
        dict(version=1, access_id=2, axis=0, factor=3, root=0, bytes=4, shape=[3], loops=[], positive=[arg(2)],
             checks=[const(0)], pointer_checks=[const(0), const(8)])
    ]
    assert guard.check_guard(spec, [0x100000, 0x200000, 3], (7, 1, 1), 3)[0]


def test_joint_effect_does_not_hide_other_output_aliases():
    spec = joint_spec()
    spec["accesses"].append(
        dict(root=3, shape=[], bytes=4, write=True, offset=mul(pid(0), const(4)), mask=None, pointer_checks=[],
             loops=[]))
    args = [0x100000, 0x200000, 3, 0x100008]
    assert guard.check_guard(spec, args, (7, 1, 1), 3) == (False, "WRITE_ALIAS")
