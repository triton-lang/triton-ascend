import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def scope_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    # The vector scope hosts non-SIMD operations such as copy and control flow.
    with tla.scope.vector():
        tla.copy(x, x_ub)

        # A SIMD scope can be nested inside a vector scope.
        with tla.scope.simd():
            v = tla.simd.vlds(x_ub[0])
            tla.simd.vsts(o_ub[0], v)

        tla.copy(o_ub, o)
