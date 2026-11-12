# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for the generated symbolic-layout arithmetic contract."""
import pytest
from test_block_batching_guard_v2 import module, matrix_guard, const, arg, pid, coord, add, mul

pytestmark = pytest.mark.backend("native")


def layout_guard(factor=3, *, read_only=False):
    spec = matrix_guard()
    if read_only:
        spec["accesses"] = spec["accesses"][:1]
    start = mul(pid(0), const(factor * 2))
    row = add(start, coord(0, factor * 2))
    endpoint = mul(["grid", 0, 32], const(2))
    base = add(mul(start, arg(4)), mul(pid(1), const(4)))
    local = add(mul(coord(0, factor * 2), arg(4)), coord(1, 4))
    byte_base = mul(base, const(4))
    byte_offset = mul(add(base, local), const(4))
    spec["layout_checks"] = [{
        "version":
        1,
        "access_id":
        i,
        "root":
        access["root"],
        "bytes":
        4,
        "axis":
        0,
        "factor":
        factor,
        "shape": [factor * 2, 4],
        "loops": [],
        "positive": [arg(4)],
        "checks": [start, row, endpoint, ["minsi", 64, arg(2, 32), endpoint], base, local, byte_base, byte_offset],
        "pointer_checks": [byte_base, byte_offset],
    } for i, access in enumerate(spec["accesses"])]
    return spec


@pytest.mark.parametrize("factor", [2, 3, 6])
@pytest.mark.parametrize("stride", [4, 7, 128, 160])
def test_dynamic_stride_and_padded_batch_domain(factor, stride):
    spec = layout_guard(factor)
    assert module.check_guard(spec, [1 << 32, 1 << 45, 9, 4, stride], (5, 1, 1), factor)[0]


def test_mask_does_not_hide_new_i64_overflow_in_inactive_lanes():
    spec = layout_guard(read_only=True)
    arguments = [65536, 0, 2, 4, 1 << 59]
    original = {k: v for k, v in spec.items() if k != "layout_checks"}
    assert module.check_guard(original, arguments, (1, 1, 1), 3)[0]
    assert module.check_guard(spec, arguments, (1, 1, 1), 3) == (False, "INDEX_OVERFLOW")


def test_mask_does_not_hide_new_pointer_wrap_in_inactive_lanes():
    spec = layout_guard(read_only=True)
    arguments = [(1 << 64) - 101, 0, 2, 4, 8]
    original = {k: v for k, v in spec.items() if k != "layout_checks"}
    assert module.check_guard(original, arguments, (1, 1, 1), 3)[0]
    assert module.check_guard(spec, arguments, (1, 1, 1), 3) == (False, "INDEX_OVERFLOW")


def test_zero_read_stride_requires_baseline_for_symbolic_layout():
    spec = layout_guard(read_only=True)
    arguments = [65536, 0, 2, 4, 0]
    original = {k: v for k, v in spec.items() if k != "layout_checks"}
    assert module.check_guard(original, arguments, (1, 1, 1), 3)[0]
    assert module.check_guard(spec, arguments, (1, 1, 1), 3) == (False, "NON_POSITIVE_LAYOUT_STRIDE")


def test_hidden_grid_stays_original_while_pid_uses_physical_grid():
    spec = layout_guard(read_only=True)
    spec["layout_checks"][0]["checks"].append(mul(["grid", 0, 32], const(1 << 62)))
    assert module.check_guard(spec, [65536, 0, 4, 4, 8], (2, 1, 1), 3) == (False, "INDEX_OVERFLOW")
    spec["layout_checks"][0]["checks"][-1] = mul(pid(0), const(1 << 62))
    # Physical grid is one block, so q=0. Original pid values would include 1.
    assert module.check_guard(spec, [65536, 0, 4, 4, 8], (2, 1, 1), 3)[0]


def test_runtime_cache_tracks_additional_layout_scalar_dependencies():
    spec = layout_guard(read_only=True)
    spec["layout_checks"][0]["positive"] = [arg(7)]
    guard = module.RuntimeGuard(spec, 3)
    values = [65536, 0, 2, 4, 8, 0, 0, 1]
    assert guard(values, (1, 1, 1))[0]
    values[7] = 0
    assert guard(values, (1, 1, 1)) == (False, "NON_POSITIVE_LAYOUT_STRIDE")
    values[7] = 1
    assert guard(values, (1, 1, 1))[0]
    values[4] = 1 << 59
    assert guard(values, (1, 1, 1)) == (False, "INDEX_OVERFLOW")


@pytest.mark.parametrize("field,value", [
    ("version", 2),
    ("axis", 1),
    ("factor", 2),
    ("factor", True),
    ("access_id", -1),
    ("root", 1),
    ("bytes", 8),
    ("shape", []),
    ("positive", []),
    ("checks", []),
    ("pointer_checks", []),
])
def test_layout_guard_schema_is_fail_closed(field, value):
    spec = layout_guard(read_only=True)
    spec["layout_checks"][0][field] = value
    assert not module.check_guard(spec, [65536, 0, 2, 4, 8], (1, 1, 1), 3)[0]


def test_layout_guard_does_not_bypass_original_write_alias_checks():
    spec = layout_guard()
    assert module.check_guard(spec, [65536, 65536, 9, 4, 8], (5, 1, 1), 3) == (False, "WRITE_ALIAS")


def test_layout_guard_is_owned_by_runtime_cache():
    spec = layout_guard(read_only=True)
    guard = module.RuntimeGuard(spec, 3)
    spec["layout_checks"].clear()
    assert guard([65536, 0, 2, 4, 1 << 59], (1, 1, 1)) == (False, "INDEX_OVERFLOW")
