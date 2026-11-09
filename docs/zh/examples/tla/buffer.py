import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def buffer_example(X_ptr, O_ptr, N: tl.constexpr):
    # Pointer entry: convert a GM pointer into a tla.buffer.
    x = tla.view(X_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    # Allocate a UB buffer; versions=2 declares double buffering.
    tla.assume(N > 0)
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub", versions=2)

    tla.copy(x, x_ub)

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], v)

    tla.copy(o_ub, o)
