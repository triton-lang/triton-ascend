import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_load_store_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        # Load a full vector register from UB.
        v = tla.simd.vlds(x_ub[0], dist="NORM")
        # Store back to UB with the default NORM_B32 distribution and an all-true mask.
        tla.simd.vsts(o_ub[0], v, dist="NORM_B32")

    tla.copy(o_ub, o)
