import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_membar_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])

        # Insert a vector-store -> vector-load barrier for write-then-read.
        tla.simd.vsts(x_ub[0], v, dist="NORM_B32")
        tla.simd.membar("VST_VLD")

        reloaded = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], reloaded, dist="NORM_B32")

    tla.copy(o_ub, o)
