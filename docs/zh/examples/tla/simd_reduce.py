import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def simd_reduce_example(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a = tla.view(A_ptr, (N, ), strides=(1, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0])
        bv = tla.simd.vlds(b_ub[0])

        # 水平规约，结果放在 lane 0。
        total = tla.simd.vreduce_sum(av)

        # 多寄存器全局最大值，结果广播到所有 lane。
        global_max = tla.simd.vreduce_max([av, bv])

        # 广播低半部分到整个寄存器。
        splat = tla.simd.vbroadcast(total, low=True)

        # 结果可直接用于 vcmp。
        mask = tla.simd.vcmp(splat, global_max, "le")
        result = tla.simd.vsel(mask, splat, global_max)
        tla.simd.vsts(o_ub[0], result, dist="NORM_B32")

    tla.copy(o_ub, o)
