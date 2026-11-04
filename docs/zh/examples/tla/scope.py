import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def scope_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    # vector 作用域承载 copy、控制流等非 SIMD 操作。
    with tla.scope.vector():
        tla.copy(x, x_ub)

        # SIMD 作用域可以嵌套在 vector 作用域中。
        with tla.scope.simd():
            v = tla.simd.vlds(x_ub[0])
            tla.simd.vsts(o_ub[0], v)

        tla.copy(o_ub, o)
