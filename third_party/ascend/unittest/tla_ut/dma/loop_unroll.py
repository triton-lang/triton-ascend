"""TLA applies explicit loop factors and preserves the remaining iterations."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def copy_loop(src, dst, count: tl.constexpr, factor: tl.constexpr):
    with tla.scope.vector():
        tile = tla.alloc((32, ), tl.float16)
        for index in tl.range(count, loop_unroll_factor=factor):
            tla.load_gm_to_ub(tla.view(src + index * 32, (32, )), tile)
            tla.sync.barrier(tla.sync.PIPE.ALL)
            tla.store_ub_to_gm(tile, tla.view(dst + index * 32, (32, )))
            tla.sync.barrier(tla.sync.PIPE.ALL)


CASES = ((15, None, 1, 1), (15, 1, 1, 1), (16, 4, 4, 1), (15, 4, 5, 2))


@pytest.mark.parametrize("count,factor,copies,loops", CASES)
def test_ir(count, factor, copies, loops):
    module, context = compile_offline(copy_loop, {"src": "*fp16", "dst": "*fp16"}, {
        "count": count,
        "factor": factor,
    })
    text = str(lower_offline(module, context))
    assert text.count("hivm.hir.load ") == copies
    assert text.count("hivm.hir.store ") == copies
    assert len(re.findall(r"scf\.for [^\n]+", text)) == loops


if __name__ == "__main__" and get_stages()[0]:
    for case in CASES:
        test_ir(*case)
