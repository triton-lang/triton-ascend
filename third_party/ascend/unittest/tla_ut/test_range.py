"""Triton loop iterators in TLA kernels."""

import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline


@tla.jit
def _loops(n, AXIS: tla.constexpr):
    tla.static_assert(AXIS == 0)
    tla.device_print("program", tla.program_id(AXIS))
    for i in tla.range(1, n, 2, loop_unroll_factor=2):
        tla.device_print("runtime", i)
    for i in tla.static_range(4, 0, -2):
        tla.device_print("unrolled", i)


def test_range_iterators():
    assert tla.constexpr is tl.constexpr
    assert tla.program_id is tl.program_id
    assert tla.static_assert is tl.static_assert
    assert tla.range is tl.range
    assert tla.static_range is tl.static_range

    module, _ = compile_offline(_loops, {"n": "i32"}, {"AXIS": 0})
    text = str(module)
    assert "tt.get_program_id" in text
    assert text.count("scf.for") == 1
    assert "tt.loop_unroll_factor = 2" in text
    assert text.count("unrolled") == 2
