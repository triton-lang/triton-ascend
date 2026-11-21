"""Small stage-chain programs using the normal Triton compilation route."""
import triton
import triton.language as tl


@triton.jit
def stage_chain(A, B0, B1, B2, Output, SEGMENTS: tl.constexpr,
                REPEATS: tl.constexpr):
    rows = tl.arange(0, 16)
    inner0 = tl.arange(0, 16)
    cols0 = tl.arange(0, 32)
    cols1 = tl.arange(0, 64)
    cols2 = tl.arange(0, 16)
    left = tl.load(A + rows[:, None] * 16 + inner0[None, :])
    if SEGMENTS == 1:
        result = tl.full((16, 32), 0, tl.float16)
    elif SEGMENTS == 2:
        result = tl.full((16, 64), 0, tl.float16)
    else:
        result = tl.full((16, 16), 0, tl.float16)
    for tile in tl.range(0, REPEATS, loop_unroll_factor=1):
        right0 = tl.load(B0 + tile * 32 * 16 + cols0[:, None] * 16 + inner0[None, :])
        matrix0 = tl.dot(left, tl.trans(right0))
        value0 = tl.exp(matrix0 * 0.125).to(tl.float16)
        if SEGMENTS == 1:
            result = value0
        else:
            right1 = tl.load(B1 + tile * 64 * 32 + cols1[:, None] * 32 + cols0[None, :])
            matrix1 = tl.dot(value0, tl.trans(right1))
            value1 = tl.exp(matrix1 * 0.125).to(tl.float16)
            if SEGMENTS == 2:
                result = value1
            else:
                right2 = tl.load(B2 + tile * 16 * 64 + cols2[:, None] * 64 + cols1[None, :])
                matrix2 = tl.dot(value1, tl.trans(right2))
                result = tl.exp(matrix2 * 0.125).to(tl.float16)
    if SEGMENTS == 1:
        tl.store(Output + rows[:, None] * 32 + cols0[None, :], result)
    elif SEGMENTS == 2:
        tl.store(Output + rows[:, None] * 64 + cols1[None, :], result)
    else:
        tl.store(Output + rows[:, None] * 16 + cols2[None, :], result)
