import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def view_example(X_ptr, O_ptr, N: tl.constexpr):
    # 指针入口：指定最终 memref 的 shape / strides / offsets。
    x = tla.view(X_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    tla.copy(x, x_ub)

    # buffer 切片：strides 相对于源视图，offset 为静态或标量整数。
    tile = tla.view(x_ub, (64, ), strides=(1, ), offsets=(0, ))

    with tla.scope.simd():
        v = tla.simd.vlds(tile[0])
        tla.simd.vsts(tile[0], v)

    tla.copy(x_ub, o)
