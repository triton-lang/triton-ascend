import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def device_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    # device_print 只能在 tla.scope 外使用，且只支持 UB 值和标量。
    tla.device_print("x[0] =", x_ub[0])

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], v, dist="NORM_B32")

    # cond 必须是标量 int1；仅在开启 debug 选项时生效。
    tla.device_assert(N > 0, "N must be positive")
    tla.copy(o_ub, o)
