import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def math_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        # tla.sqrt 同时支持 f32 标量和 f16/f32 SIMD 向量。
        tla.simd.vsts(o_ub[0], tla.sqrt(v), dist="NORM_B32")

    # 整数取余向零截断，结果符号与被除数一致；结果恒为 1。
    tla.assume(tla.truncmod(7, 3) == 1)

    tla.copy(o_ub, o)
