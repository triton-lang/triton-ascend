import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def device_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    # device_print is only allowed outside tla.scope and only supports UB values and scalars.
    tla.device_print("x[0] =", x_ub[0])

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], v, dist="NORM_B32")

    # cond must be a scalar int1; it only takes effect when the debug option is enabled.
    tla.device_assert(N > 0, "N must be positive")
    tla.copy(o_ub, o)
