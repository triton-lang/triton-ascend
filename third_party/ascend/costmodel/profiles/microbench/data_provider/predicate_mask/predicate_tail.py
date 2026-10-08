"""Tail-mask geometry probe; every input/output allocation holds BLOCK elements.

MODE bits: 1 masks loads (other=0), 2 masks the store; 0 is the unmasked control.
N is a runtime bound. OP selects AND/OR/XOR. U, V and OUT2 preserve the existing
simulator host's pointer ABI and are unused by this probe.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def probe(A, B, C, D, X, Y, U, V, OUT, OUT2, N, MODE: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    valid = i < N
    if MODE & 1:
        a = tl.load(A + i, valid, other=0)
        b = tl.load(B + i, valid, other=0)
        c = tl.load(C + i, valid, other=0)
        d = tl.load(D + i, valid, other=0)
        x = tl.load(X + i, valid, other=0)
        y = tl.load(Y + i, valid, other=0)
    else:
        a, b = tl.load(A + i), tl.load(B + i)
        c, d = tl.load(C + i), tl.load(D + i)
        x, y = tl.load(X + i), tl.load(Y + i)
    p, q = a < b, c < d
    if OP == 0:
        mask = p & q
    elif OP == 1:
        mask = p | q
    else:
        mask = p ^ q
    result = tl.where(mask, x, y)
    if MODE & 2:
        tl.store(OUT + i, result, valid)
    else:
        tl.store(OUT + i, result)
