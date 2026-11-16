import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def math_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        # tla.sqrt supports both f32 scalars and f16/f32 SIMD vectors.
        tla.simd.vsts(o_ub[0], tla.sqrt(v), dist="NORM_B32")

    # Integer remainder truncated toward zero; the result has the dividend's sign. Here it is always 1.
    tla.assume(tla.truncmod(7, 3) == 1)

    tla.copy(o_ub, o)
