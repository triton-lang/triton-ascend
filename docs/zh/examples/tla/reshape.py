import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def reshape_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))

    x_ub = tla.alloc_buffer((N // 64, 64), tl.float32, space="ub")
    tla.copy(x, x_ub)

    # 连续、行主序、无偏移的 buffer 可以做纯形状变换。
    flat = tla.reshape(x_ub, (N, ))

    with tla.scope.simd():
        v = tla.simd.vlds(flat[0])
        tla.simd.vsts(flat[0], v)

    tla.copy(x_ub, o)
