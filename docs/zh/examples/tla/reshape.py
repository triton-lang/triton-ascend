import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def reshape_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))

    x_ub = tla.alloc_buffer((N // 64, 64), tl.float32, space="ub")
    tla.copy(x, x_ub)

    # A contiguous row-major buffer with zero offset supports a pure shape transform.
    flat = tla.reshape(x_ub, (N, ))

    with tla.scope.simd():
        v = tla.simd.vlds(flat[0])
        tla.simd.vsts(flat[0], v)

    tla.copy(x_ub, o)
