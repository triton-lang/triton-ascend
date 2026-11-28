import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def compensated_rstd_example(SQ_ptr, O_ptr, N: tl.constexpr, HIDDEN: tl.constexpr):
    sq = tla.view(SQ_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    sq_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(sq, sq_ub)

    with tla.scope.simd():
        # square_sum is an f32 vector register, usually broadcast to all lanes.
        square_sum = tla.simd.vlds(sq_ub[0], dist="BRC_B32")

        # eps must be at least the smallest normal FP32 value, 2^-126.
        rstd = tla.compensated_rstd(square_sum, HIDDEN, 1e-6)
        tla.simd.vsts(o_ub[0], rstd, dist="NORM_B32")

    tla.copy(o_ub, o)
