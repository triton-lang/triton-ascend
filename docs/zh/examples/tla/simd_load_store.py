import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def simd_load_store_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        # 从 UB 加载一个满向量寄存器。
        v = tla.simd.vlds(x_ub[0], dist="NORM")
        # 默认使用 NORM_B32 分布和全真 mask 写回 UB。
        tla.simd.vsts(o_ub[0], v, dist="NORM_B32")

    tla.copy(o_ub, o)
