import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def dma_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    # 显式 DMA 路由；tla.copy 会根据源/目标地址空间自动分派。
    tla.load_gm_to_ub(x, x_ub)

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], v, dist="NORM_B32")

    tla.store_ub_to_gm(o_ub, o)
