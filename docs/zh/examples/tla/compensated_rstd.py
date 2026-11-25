import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def compensated_rstd_example(SQ_ptr, O_ptr, N: tl.constexpr, HIDDEN: tl.constexpr):
    sq = tla.view(SQ_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    sq_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(sq, sq_ub)

    with tla.scope.simd():
        # square_sum 为 f32 向量寄存器，通常已广播到所有 lane。
        square_sum = tla.simd.vlds(sq_ub[0], dist="BRC_B32")

        # eps 必须不小于最小 FP32 正规数 2^-126。
        rstd = tla.compensated_rstd(square_sum, HIDDEN, 1e-6)
        tla.simd.vsts(o_ub[0], rstd, dist="NORM_B32")

    tla.copy(o_ub, o)
