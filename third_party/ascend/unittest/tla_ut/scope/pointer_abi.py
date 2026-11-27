"""Normalize boolean pointer arguments in the TLA entry ABI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def bool_pointer_abi(B, X, Y):
    with tla.scope.vector():
        ub = tla.alloc((64, ), tl.float32)
        tla.load_gm_to_ub(tla.view(X, (64, )), ub)
        tla.store_ub_to_gm(ub, tla.view(Y, (64, )))


def test_bool_pointer_abi():
    module, ctx = compile_offline(bool_pointer_abi, {"B": "*i1", "X": "*fp32", "Y": "*fp32"}, {})
    lowered = str(lower_offline(module, ctx))
    entry = next(line for line in lowered.splitlines() if "func.func @bool_pointer_abi(" in line)
    assert "%B: memref<?xi8>" in entry
    assert "memref<?xi1>" not in lowered
    print("IR PASS: TLA i1 pointer ABI")


def main():
    if get_stages()[0]:
        test_bool_pointer_abi()


if __name__ == "__main__":
    main()
