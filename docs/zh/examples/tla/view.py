import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def view_example(X_ptr, O_ptr, N: tl.constexpr):
    # Pointer entry: specify the final memref shape / strides / offsets.
    x = tla.view(X_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    tla.copy(x, x_ub)

    # Buffer slice: subview preserves the source strides; offsets are static or scalar integers.
    tile = tla.subview(x_ub, offsets=(0, ), sizes=(64, ))

    with tla.scope.simd():
        v = tla.simd.vlds(tile[0])
        tla.simd.vsts(tile[0], v)

    tla.copy(x_ub, o)
